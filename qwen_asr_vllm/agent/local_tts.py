from __future__ import annotations

import io
import shutil
import wave
from importlib.util import find_spec as default_find_spec
from typing import Any, Callable

import numpy as np

from qwen_asr_vllm.agent.qwen_tts_streaming import (
    run_with_codec_frame_batch_hook,
    run_with_codec_frame_hook,
    stream_decode_codec_frame_batches,
    stream_decode_codec_frames,
)
from qwen_asr_vllm.agent.qwen_tts_fast_predictor import (
    install_batched_fast_code_predictor,
)
from qwen_asr_vllm.agent.qwen_tts_cuda_graph_predictor import (
    install_cuda_graph_code_predictor,
)
from qwen_asr_vllm.agent.qwen_tts_outer_engine import (
    install_explicit_talker_step_engine,
)
from qwen_asr_vllm.agent.qwen_tts_outer_graph_engine import (
    install_cuda_graph_outer_talker,
)
from qwen_asr_vllm.agent.qwen_tts_outer_static_engine import (
    install_static_outer_talker,
)
from qwen_asr_vllm.agent.qwen_tts_static_predictor import (
    install_static_code_predictor,
)


class QwenTtsDependencyError(RuntimeError):
    """Raised when the local Qwen-TTS runtime is not importable."""


def _group_stream_batch_texts(wrapper: Any, texts: list[str]):
    """Group stream requests by prompt length to avoid left-pad drift."""

    tokenize = getattr(wrapper, "_tokenize_texts", None)
    build_text = getattr(wrapper, "_build_assistant_text", None)
    if not callable(tokenize) or not callable(build_text):
        return [(list(range(len(texts))), list(texts))]

    tokenized = tokenize([build_text(text) for text in texts])
    groups: dict[int, tuple[list[int], list[str]]] = {}
    for index, (text, input_ids) in enumerate(zip(texts, tokenized)):
        length = int(input_ids.shape[-1])
        group = groups.get(length)
        if group is None:
            group = ([], [])
            groups[length] = group
        group[0].append(index)
        group[1].append(text)
    return list(groups.values())


def install_active_prefix_outer_talker(talker: Any, *, max_cache_len: int) -> bool:
    from qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache import (
        install_active_prefix_outer_talker as install,
    )

    return install(talker, max_cache_len=max_cache_len)


def check_qwen_tts_runtime(
    *,
    find_spec: Callable[[str], object | None] = default_find_spec,
    which: Callable[[str], str | None] = shutil.which,
) -> None:
    missing = []
    if find_spec("qwen_tts") is None:
        missing.append("qwen-tts")
    if find_spec("onnxruntime") is None:
        missing.append("onnxruntime")
    if which("sox") is None:
        missing.append("sox")
    if not missing:
        return

    raise QwenTtsDependencyError(
        "Local Qwen-TTS runtime is incomplete: missing "
        + ", ".join(missing)
        + ". Install Python deps into the nano-vLLM environment with "
        "`/opt/conda/envs/nano-vllm/bin/python -m pip install --no-deps "
        "qwen-tts==0.1.1 onnxruntime`; install the system `sox` binary separately."
    )


class QwenTtsBackend:
    """Local Qwen3-TTS backend using the `qwen_tts` package and local checkpoint."""

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cuda:1",
        language: str = "chinese",
        speaker: str = "",
        dtype=None,
        warmup: bool = True,
        check_runtime: bool = True,
        streaming: bool = False,
        stream_chunk_size: int = 8,
        stream_first_chunk_size: int | None = None,
        stream_left_context_size: int = 4,
        stream_batch_exact_parity: bool = True,
        stream_batch_parity_gate: bool = True,
        fast_code_predictor: bool = False,
        static_code_predictor: bool = False,
        cuda_graph_code_predictor: bool = False,
        cuda_graph_fixed_slots: int = 1,
        cuda_graph_batch_window_ms: float = 0.0,
        fast_code_predictor_batch_window_ms: float = 0.0,
        fast_code_predictor_max_batch_size: int = 8,
        outer_static_talker_engine: bool = False,
        outer_active_prefix_talker_engine: bool = False,
        outer_graph_max_cache_len: int = 16384,
        outer_cuda_graph_talker_engine: bool = False,
        outer_graph_fixed_slots: int = 2,
        outer_graph_fuse_qkv: bool = False,
        outer_graph_fuse_attention: bool = False,
        outer_graph_fused_projection_kernel: bool = False,
        outer_graph_fusion_policy: str = "allow",
        outer_graph_fusion_max_abs_error: float = 2.0e-3,
        outer_graph_fusion_max_relative_l2: float = 2.0e-4,
        explicit_talker_step_engine: bool = False,
        compile_step_engine: bool = False,
        compile_step_engine_mode: str = "reduce-overhead",
        do_sample: bool | None = None,
        subtalker_dosample: bool | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
    ):
        if eos_token_id is not None and int(eos_token_id) < 0:
            raise ValueError(
                "eos_token_id must be non-negative; use max_new_tokens to cap generation"
            )
        if outer_active_prefix_talker_engine and (
            outer_static_talker_engine
            or outer_cuda_graph_talker_engine
            or explicit_talker_step_engine
        ):
            raise ValueError(
                "active-prefix, fixed-static, and explicit outer engines are "
                "mutually exclusive"
            )
        outer_engines = sum(
            bool(value)
            for value in (
                outer_static_talker_engine,
                outer_cuda_graph_talker_engine,
                outer_active_prefix_talker_engine,
                explicit_talker_step_engine,
            )
        )
        if outer_engines > 1:
            raise ValueError("outer talker engines are mutually exclusive")
        if check_runtime:
            check_qwen_tts_runtime()
        import torch
        from qwen_tts import Qwen3TTSModel

        self._language = language
        self._streaming = bool(streaming)
        self._stream_chunk_size = max(1, int(stream_chunk_size))
        self._stream_first_chunk_size = (
            max(1, int(stream_first_chunk_size))
            if stream_first_chunk_size is not None
            else None
        )
        self._stream_left_context_size = max(0, int(stream_left_context_size))
        self._stream_batch_exact_parity = bool(stream_batch_exact_parity)
        self._stream_batch_parity_gate = bool(stream_batch_parity_gate)
        self._stream_batch_gate_stats = {
            "native_batches": 0,
            "scalar_fallback_batches": 0,
            "scalar_fallback_requests": 0,
        }
        self._fast_code_predictor = bool(fast_code_predictor)
        self._static_code_predictor = bool(static_code_predictor)
        self._cuda_graph_code_predictor = bool(cuda_graph_code_predictor)
        self._cuda_graph_fixed_slots = max(1, int(cuda_graph_fixed_slots))
        self._cuda_graph_batch_window_ms = max(
            0.0,
            float(cuda_graph_batch_window_ms),
        )
        self._fast_code_predictor_batch_window_ms = max(
            0.0,
            float(fast_code_predictor_batch_window_ms),
        )
        self._fast_code_predictor_max_batch_size = max(
            1,
            int(fast_code_predictor_max_batch_size),
        )
        self._outer_static_talker_engine = bool(outer_static_talker_engine)
        self._outer_active_prefix_talker_engine = bool(
            outer_active_prefix_talker_engine
        )
        self._outer_graph_max_cache_len = max(1, int(outer_graph_max_cache_len))
        self._outer_cuda_graph_talker_engine = bool(outer_cuda_graph_talker_engine)
        self._outer_graph_fixed_slots = max(1, int(outer_graph_fixed_slots))
        self._outer_graph_fuse_qkv = bool(outer_graph_fuse_qkv)
        self._outer_graph_fuse_attention = bool(outer_graph_fuse_attention)
        self._outer_graph_fused_projection_kernel = bool(
            outer_graph_fused_projection_kernel
        )
        self._outer_graph_fusion_policy = str(outer_graph_fusion_policy)
        self._outer_graph_fusion_max_abs_error = float(
            outer_graph_fusion_max_abs_error
        )
        self._outer_graph_fusion_max_relative_l2 = float(
            outer_graph_fusion_max_relative_l2
        )
        self._explicit_talker_step_engine = bool(explicit_talker_step_engine)
        self._compile_step_engine = bool(compile_step_engine)
        self._compile_step_engine_mode = str(compile_step_engine_mode)
        self._generation_kwargs = {}
        if do_sample is not None:
            self._generation_kwargs["do_sample"] = bool(do_sample)
        if subtalker_dosample is not None:
            self._generation_kwargs["subtalker_dosample"] = bool(subtalker_dosample)
        if temperature is not None:
            self._generation_kwargs["temperature"] = float(temperature)
        if max_new_tokens is not None:
            if int(max_new_tokens) <= 0:
                raise ValueError("max_new_tokens must be positive")
            self._generation_kwargs["max_new_tokens"] = int(max_new_tokens)
        if eos_token_id is not None:
            self._generation_kwargs["eos_token_id"] = int(eos_token_id)
        self.supports_streaming_tts = self._streaming
        dtype = dtype or torch.float16
        self._model = Qwen3TTSModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device,
        )
        if self._cuda_graph_code_predictor:
            install_cuda_graph_code_predictor(
                self._model.model.talker,
                fixed_slot_count=self._cuda_graph_fixed_slots,
                batch_window_ms=self._cuda_graph_batch_window_ms,
            )
        elif self._static_code_predictor:
            install_static_code_predictor(self._model.model.talker)
        elif self._fast_code_predictor:
            install_batched_fast_code_predictor(
                self._model.model.talker,
                batch_window_ms=self._fast_code_predictor_batch_window_ms,
                max_batch_size=self._fast_code_predictor_max_batch_size,
                compile_step=self._compile_step_engine,
                compile_mode=self._compile_step_engine_mode,
            )
        if self._outer_static_talker_engine:
            install_static_outer_talker(
                self._model.model.talker,
                max_cache_len=self._outer_graph_max_cache_len,
            )
        if self._outer_cuda_graph_talker_engine:
            install_cuda_graph_outer_talker(
                self._model.model.talker,
                max_cache_len=self._outer_graph_max_cache_len,
                max_graph_batch_size=self._outer_graph_fixed_slots,
                fuse_qkv=self._outer_graph_fuse_qkv,
                fuse_attention=self._outer_graph_fuse_attention,
                fused_projection_kernel=self._outer_graph_fused_projection_kernel,
                fusion_policy=self._outer_graph_fusion_policy,
                fusion_max_abs_error=self._outer_graph_fusion_max_abs_error,
                fusion_max_relative_l2=self._outer_graph_fusion_max_relative_l2,
            )
        if self._outer_active_prefix_talker_engine:
            install_active_prefix_outer_talker(
                self._model.model.talker,
                max_cache_len=self._outer_graph_max_cache_len,
            )
        if self._explicit_talker_step_engine:
            install_explicit_talker_step_engine(
                self._model.model.talker,
                compile_step=self._compile_step_engine,
                compile_mode=self._compile_step_engine_mode,
            )
        speakers = self._model.get_supported_speakers() or []
        self._speaker = speaker or (speakers[0] if speakers else "")
        if warmup:
            warmup_kwargs = {}
            if (
                self._outer_static_talker_engine
                or self._outer_cuda_graph_talker_engine
                or self._outer_active_prefix_talker_engine
            ):
                warmup_kwargs["max_new_tokens"] = min(
                    64,
                    max(1, self._outer_graph_max_cache_len // 2),
                )
            warmup_kwargs.update(self._generation_parameters())
            self._model.generate_custom_voice(
                text="你好",
                speaker=self._speaker,
                language=self._language,
                **warmup_kwargs,
            )
            predictor = getattr(self._model.model.talker, "code_predictor", None)
            predictor_generate = getattr(predictor, "generate", None)
            predictor_engine = getattr(predictor_generate, "_qav_engine", None)
            mark_steady_state = getattr(predictor_engine, "mark_steady_state", None)
            if callable(mark_steady_state):
                mark_steady_state()

    def synthesize(self, text: str) -> bytes | None:
        return self.synthesize_with_mode(text, non_streaming_mode=True)

    def synthesize_with_mode(
        self, text: str, *, non_streaming_mode: bool
    ) -> bytes | None:
        audios, sample_rate = self._model.generate_custom_voice(
            text=text,
            speaker=self._speaker,
            language=self._language,
            non_streaming_mode=non_streaming_mode,
            **self._generation_parameters(),
        )
        if not audios or len(audios[0]) == 0:
            return None
        return self._numpy_to_wav(audios[0], sample_rate)

    def synthesize_stream(self, text: str):
        if not self._streaming:
            audio = self.synthesize(text)
            if audio:
                yield audio
            return

        streaming_model = getattr(self._model, "model", self._model)
        eos_token_id = self._generation_parameters().get(
            "eos_token_id",
            getattr(
                getattr(getattr(streaming_model, "config", None), "talker_config", None),
                "codec_eos_token_id",
                None,
            ),
        )

        def run_generate(on_codec_frame):
            talker = streaming_model.talker
            return run_with_codec_frame_hook(
                talker,
                lambda: self._run_custom_voice_generate_only(text),
                on_codec_frame,
            )

        for chunk in stream_decode_codec_frames(
            run_generate,
            streaming_model.speech_tokenizer,
            chunk_size=self._stream_chunk_size,
            first_chunk_size=self._stream_first_chunk_size,
            left_context_size=self._stream_left_context_size,
            eos_token_id=eos_token_id,
        ):
            if chunk.wav_bytes:
                yield chunk.wav_bytes

    def synthesize_stream_batch(self, texts: list[str]):
        indexed_texts = [
            (index, str(text))
            for index, text in enumerate(texts)
            if str(text).strip()
        ]
        if not indexed_texts:
            return
        normalized_texts = [text for _index, text in indexed_texts]
        if not self._streaming:
            for local_index, audio in enumerate(self.synthesize_batch(normalized_texts)):
                if audio:
                    yield indexed_texts[local_index][0], audio
            return

        streaming_model = getattr(self._model, "model", self._model)
        eos_token_id = self._generation_parameters().get(
            "eos_token_id",
            getattr(
                getattr(getattr(streaming_model, "config", None), "talker_config", None),
                "codec_eos_token_id",
                None,
            ),
        )

        if getattr(self, "_stream_batch_exact_parity", True):
            # Preserve the legacy scalar decoder and its exact codec/audio path.
            for request_id, text in indexed_texts:
                for chunk in self.synthesize_stream(text):
                    if chunk:
                        yield request_id, chunk
            return

        groups = _group_stream_batch_texts(self._model, normalized_texts)
        for local_indices, group_texts in groups:
            request_ids = [indexed_texts[index][0] for index in local_indices]
            if not self._native_stream_batch_allowed(group_texts):
                self._stream_batch_gate_stats["scalar_fallback_batches"] += 1
                self._stream_batch_gate_stats["scalar_fallback_requests"] += len(
                    group_texts
                )
                for request_id, text in zip(request_ids, group_texts):
                    for chunk in self.synthesize_stream(text):
                        if chunk:
                            yield request_id, chunk
                continue
            self._stream_batch_gate_stats["native_batches"] += 1

            def run_generate(on_codec_frame_batch, group_texts=group_texts):
                talker = streaming_model.talker
                generation_text = (
                    group_texts[0] if len(group_texts) == 1 else group_texts
                )
                return run_with_codec_frame_batch_hook(
                    talker,
                    lambda: self._run_custom_voice_generate_only(generation_text),
                    on_codec_frame_batch,
                )

            for index, chunk in stream_decode_codec_frame_batches(
                run_generate,
                streaming_model.speech_tokenizer,
                request_ids=request_ids,
                chunk_size=self._stream_chunk_size,
                first_chunk_size=self._stream_first_chunk_size,
                left_context_size=self._stream_left_context_size,
                eos_token_id=eos_token_id,
            ):
                if chunk.wav_bytes:
                    yield index, chunk.wav_bytes

    def stream_batch_compatibility_key(self, text: str):
        """Return the service-level cohort key used before native generation."""

        if getattr(self, "_stream_batch_exact_parity", True) or not self._native_stream_batch_allowed(
            [str(text), str(text)]
        ):
            return ("scalar", object())
        wrapper = self._model
        tokenize = getattr(wrapper, "_tokenize_texts", None)
        build_text = getattr(wrapper, "_build_assistant_text", None)
        if not callable(tokenize) or not callable(build_text):
            return ("native", None)
        input_ids = tokenize([build_text(str(text))])[0]
        return ("native", int(input_ids.shape[-1]))

    def synthesize_batch(self, texts: list[str]) -> list[bytes | None]:
        if not texts:
            return []
        audios, sample_rate = self._model.generate_custom_voice(
            text=list(texts),
            speaker=self._speaker,
            language=self._language,
            non_streaming_mode=True,
            **self._generation_parameters(),
        )
        output: list[bytes | None] = []
        for audio in audios:
            if audio is None or len(audio) == 0:
                output.append(None)
            else:
                output.append(self._numpy_to_wav(audio, sample_rate))
        return output

    def close(self) -> None:
        talker = self._model.model.talker
        predictor_generate = talker.code_predictor.generate
        predictor_runtime = getattr(predictor_generate, "_qav_scheduler", None)
        if predictor_runtime is None:
            predictor_runtime = getattr(predictor_generate, "_qav_engine", None)

        outer_engine = getattr(talker.generate, "_qav_outer_engine", None)
        outer_runtime = getattr(outer_engine, "step_runtime", None)
        first_error: BaseException | None = None
        for runtime in (predictor_runtime, outer_runtime):
            close = getattr(runtime, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def runtime_metrics(self) -> dict[str, Any]:
        talker = self._model.model.talker
        predictor_generate = talker.code_predictor.generate
        predictor_runtime = getattr(predictor_generate, "_qav_scheduler", None)
        if predictor_runtime is None:
            predictor_runtime = getattr(predictor_generate, "_qav_engine", None)
        outer_engine = getattr(talker.generate, "_qav_outer_engine", None)
        result = {
            "code_predictor": _runtime_metrics_snapshot(predictor_runtime),
            "outer_talker": _runtime_metrics_snapshot(outer_engine),
        }
        if hasattr(self, "_stream_batch_gate_stats"):
            result["stream_batch_gate"] = dict(self._stream_batch_gate_stats)
        return result

    def _native_stream_batch_allowed(self, group_texts: list[str]) -> bool:
        """Apply the cheap runtime parity gate before native outer batching.

        Native batched generation is only parity-safe for deterministic
        requests. Prompt-length grouping is performed by
        ``_group_stream_batch_texts``. Sampling requests are routed through the
        scalar path because a batch-local RNG/state change cannot be validated
        without running a second generation.
        """

        if not self._stream_batch_parity_gate:
            return True
        if len(group_texts) <= 1:
            return False
        if self._generation_kwargs.get("do_sample") is True:
            return False
        if self._generation_kwargs.get("subtalker_dosample") is True:
            return False
        return True

    @staticmethod
    def _numpy_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
        if audio.dtype in (np.float32, np.float64):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(audio.tobytes())
        return buffer.getvalue()

    def _run_custom_voice_generate_only(self, text: str) -> Any:
        wrapper = self._model
        model = getattr(wrapper, "model", None)
        if model is None or not all(
            hasattr(wrapper, name)
            for name in (
                "_ensure_list",
                "_validate_languages",
                "_validate_speakers",
                "_tokenize_texts",
                "_build_assistant_text",
                "_merge_generate_kwargs",
            )
        ):
            return wrapper.generate_custom_voice(
                text=text,
                speaker=self._speaker_values(text),
                language=self._language_values(text),
                non_streaming_mode=True,
                **self._generation_parameters(),
            )

        texts = wrapper._ensure_list(text)
        languages = (
            wrapper._ensure_list(self._language)
            if isinstance(self._language, list)
            else (
                [self._language] * len(texts)
                if self._language is not None
                else ["Auto"] * len(texts)
            )
        )
        speakers = wrapper._ensure_list(self._speaker)
        instructs = [""] * len(texts)

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(speakers) != len(texts) or len(languages) != len(texts):
            raise ValueError("text/language/speaker batch sizes do not match")

        wrapper._validate_languages(languages)
        wrapper._validate_speakers(speakers)
        input_ids = wrapper._tokenize_texts(
            [wrapper._build_assistant_text(item) for item in texts]
        )
        instruct_ids = [None for _ in instructs]
        gen_kwargs = wrapper._merge_generate_kwargs(**self._generation_parameters())
        return model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=True,
            **gen_kwargs,
        )

    def _generation_parameters(self) -> dict[str, object]:
        # Keep lightweight test doubles and older in-process instances compatible.
        return dict(getattr(self, "_generation_kwargs", {}))

    def _speaker_values(self, text: str | list[str]) -> str | list[str]:
        if isinstance(text, list):
            return [self._speaker] * len(text)
        return self._speaker

    def _language_values(self, text: str | list[str]) -> str | list[str]:
        if isinstance(text, list):
            return [self._language] * len(text)
        return self._language


def _runtime_metrics_snapshot(runtime: Any) -> dict[str, Any]:
    snapshot = getattr(runtime, "metrics_snapshot", None)
    if not callable(snapshot):
        return {}
    metrics = snapshot()
    if not isinstance(metrics, dict):
        raise TypeError("runtime metrics_snapshot() must return a dictionary")
    return _copy_plain_runtime_data(metrics)


def _copy_plain_runtime_data(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        copied = {}
        for key, item in value.items():
            if not isinstance(key, (bool, int, float, str)) and key is not None:
                raise TypeError("runtime metrics must contain only plain data")
            copied[key] = _copy_plain_runtime_data(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [_copy_plain_runtime_data(item) for item in value]
    raise TypeError("runtime metrics must contain only plain data")
