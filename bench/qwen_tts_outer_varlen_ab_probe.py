"""Full-codec ActivePrefix versus request-id packed varlen A/B probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache import (
    install_active_prefix_outer_talker,
)
from qwen_asr_vllm.agent.qwen_tts_outer_static_engine import (
    _sample_outer_token,
    select_text_condition,
)
from qwen_asr_vllm.agent.qwen_tts_outer_varlen import (
    OuterVarlenRequest,
    QwenOuterVarlenAdapter,
    QwenOuterVarlenRunner,
    build_packed_step,
)


@dataclass
class _CodecState:
    request_id: str
    request: OuterVarlenRequest
    logits: Any
    generation_step: int = 0
    first_codebook_history: list[Any] = field(default_factory=list)
    codec_frames: list[Any] = field(default_factory=list)
    active: bool = True


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _slice_prepared(prepared: dict[str, Any], row: int, batch_size: int) -> dict[str, Any]:
    result = {}
    for key, value in prepared.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            result[key] = value[row : row + 1].clone()
        else:
            result[key] = value.clone() if torch.is_tensor(value) else value
    return result


def _raw_codec_hash(codec: Any) -> str:
    return hashlib.sha256(codec.detach().to(device="cpu").contiguous().numpy().tobytes()).hexdigest()


def _codec_parity_details(active: Any, candidate: Any) -> dict[str, Any]:
    frame_count = min(int(active.shape[1]), int(candidate.shape[1]))
    frame_exact = [
        bool(torch.equal(active[:, frame], candidate[:, frame]))
        for frame in range(frame_count)
    ]
    first_divergence = next(
        (frame for frame, exact in enumerate(frame_exact) if not exact),
        None,
    )
    return {
        "shape_equal": tuple(active.shape) == tuple(candidate.shape),
        "frame_exact": frame_exact,
        "first_divergent_frame": first_divergence,
    }


def _sample_first(logits: Any, state: _CodecState, prepared: dict[str, Any]) -> Any:
    return _sample_outer_token(
        logits[:, -1, :] if logits.ndim == 3 else logits,
        state.first_codebook_history,
        do_sample=bool(prepared.get("do_sample", False)),
        top_p=prepared.get("top_p"),
        top_k=prepared.get("top_k"),
        temperature=prepared.get("temperature"),
        repetition_penalty=prepared.get("repetition_penalty"),
        suppress_tokens=prepared.get("suppress_tokens"),
    )


def _predictor_kwargs(prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        "do_sample": prepared.get("subtalker_dosample"),
        "top_p": prepared.get("subtalker_top_p"),
        "top_k": prepared.get("subtalker_top_k"),
        "temperature": prepared.get("subtalker_temperature"),
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }


def _run_active_prefix(
    wrapper: Any,
    prepared: dict[str, Any],
    *,
    device: str,
    max_new_tokens: int,
) -> tuple[list[Any], float, float]:
    talker = wrapper.model.talker
    batch_size = int(prepared["inputs_embeds"].shape[0])
    original_generate = talker.generate
    installed = install_active_prefix_outer_talker(
        talker,
        max_cache_len=int(prepared["inputs_embeds"].shape[1]) + max_new_tokens + 2,
    )
    if not installed:
        raise RuntimeError("ActivePrefix installer did not install")
    results = []
    generation_started = time.perf_counter()
    try:
        for row in range(batch_size):
            row_prepared = _slice_prepared(prepared, row, batch_size)
            row_prepared["max_new_tokens"] = int(max_new_tokens)
            _synchronize(device)
            output = talker.generate(**row_prepared)
            _synchronize(device)
            histories = getattr(output, "hidden_states", None)
            frames = [history[-1] for history in histories if history[-1] is not None]
            if not frames:
                raise RuntimeError(f"ActivePrefix request {row} produced no codec frames")
            results.append(torch.stack(frames, dim=1))
    finally:
        engine = getattr(talker.generate, "_qav_outer_engine", None)
        runtime = getattr(engine, "step_runtime", None)
        if runtime is not None:
            runtime.close()
        talker.generate = original_generate
    generation_ms = (time.perf_counter() - generation_started) * 1000.0
    decode_started = time.perf_counter()
    _decode_audio(wrapper, results, device=device)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0
    return results, generation_ms, decode_ms


def _decode_audio(wrapper: Any, codecs: Sequence[Any], *, device: str) -> list[Any]:
    eos = int(wrapper.model.config.talker_config.codec_eos_token_id)
    items = []
    for codec in codecs:
        first = codec[0, :, 0]
        matches = (first == eos).nonzero(as_tuple=False)
        length = int(matches[0].item()) if matches.numel() else int(codec.shape[1])
        items.append(codec[0, :length])
    _synchronize(device)
    audios, _ = wrapper.model.speech_tokenizer.decode([{"audio_codes": item} for item in items])
    _synchronize(device)
    return audios


def _make_varlen_states(
    wrapper: Any,
    prepared: dict[str, Any],
    *,
    adapter: QwenOuterVarlenAdapter,
    runner: QwenOuterVarlenRunner,
    device: str,
) -> list[_CodecState]:
    talker = wrapper.model.talker
    embeds = prepared["inputs_embeds"]
    attention_mask = prepared["attention_mask"]
    position_ids, rope_deltas = talker.get_rope_index(attention_mask)
    rope_deltas = rope_deltas - (1 - attention_mask).sum(-1, keepdim=True)
    states = []
    for row in range(int(embeds.shape[0])):
        request_id = ("short", "long")[row]
        prompt = embeds[row : row + 1]
        table = runner.allocate_request(
            request_id,
            max_seq_len=int(prompt.shape[1]) + int(prepared["max_new_tokens"]) + 2,
        )
        last_hidden = runner.prefill(
            request_id,
            inputs_embeds=prompt,
            position_ids=position_ids[:, row : row + 1],
            attention_mask=attention_mask[row : row + 1],
        )
        request = OuterVarlenRequest(
            request_id=request_id,
            page_table=table,
            seq_len=int(prompt.shape[1]),
            max_seq_len=table.max_seq_len,
            rope_delta=rope_deltas[row : row + 1],
            past_hidden=last_hidden[:, -1:],
            trailing_text_hidden=prepared["trailing_text_hidden"][row : row + 1],
            tts_pad_embed=prepared["tts_pad_embed"][row : row + 1],
        )
        logits = talker.codec_head(last_hidden)
        states.append(_CodecState(request_id, request, logits))
    return states


def _run_varlen(
    wrapper: Any,
    prepared: dict[str, Any],
    *,
    device: str,
    page_size: int,
    max_pages: int,
    stagger_steps: int,
    use_paged_attention_kernel: bool,
    use_fused_qkv: bool,
    paged_attention_block_n: int,
    paged_attention_num_warps: int,
) -> tuple[list[Any], float, float, dict[str, Any]]:
    talker = wrapper.model.talker
    adapter = QwenOuterVarlenAdapter(
        talker,
        max_cache_len=int(prepared["inputs_embeds"].shape[1]) + int(prepared["max_new_tokens"]) + 2,
        page_size=page_size,
        max_pages=max_pages,
        use_paged_attention_kernel=use_paged_attention_kernel,
        use_fused_qkv=use_fused_qkv,
        paged_attention_block_n=paged_attention_block_n,
        paged_attention_num_warps=paged_attention_num_warps,
    )
    runner = adapter.create_runner()
    states = []
    generation_started = time.perf_counter()
    try:
        states = _make_varlen_states(
            wrapper,
            prepared,
            adapter=adapter,
            runner=runner,
            device=device,
        )
        max_decode_steps = int(prepared["max_new_tokens"]) - 1
        for step_index in range(max_decode_steps + int(stagger_steps)):
            selected = [
                state
                for state in states
                if state.active
                and state.generation_step < max_decode_steps
                and (step_index >= stagger_steps or state.request_id == "short")
            ]
            if not selected:
                continue
            first_ids = torch.cat(
                [_sample_first(state.logits, state, prepared) for state in selected], dim=0
            )
            first_hidden = talker.get_input_embeddings()(first_ids)
            predictor_inputs = torch.cat(
                [torch.cat((state.request.past_hidden, first_hidden[row : row + 1]), dim=1)
                 for row, state in enumerate(selected)],
                dim=0,
            )
            predictor_result = talker.code_predictor.generate(
                inputs_embeds=predictor_inputs,
                max_new_tokens=int(talker.config.num_code_groups) - 1,
                **_predictor_kwargs(prepared),
            )
            codec_ids = torch.cat((first_ids, predictor_result.sequences), dim=-1)
            condition = select_text_condition([state.request for state in selected])
            step = build_packed_step([state.request for state in selected])
            output = runner.decode_packed(
                step,
                codec_ids=codec_ids,
                condition=condition,
            )
            for row, state in enumerate(selected):
                state.first_codebook_history.append(codec_ids[row : row + 1, :1])
                state.codec_frames.append(codec_ids[row : row + 1].unsqueeze(1))
                state.request.past_hidden = output.past_hidden[row : row + 1]
                state.request.seq_len += 1
                state.generation_step += 1
                state.logits = output.logits[row : row + 1]
        _synchronize(device)
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        codecs = [torch.cat(state.codec_frames, dim=1) for state in states]
        decode_started = time.perf_counter()
        audios = _decode_audio(wrapper, codecs, device=device)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        metrics = adapter.metrics()
        metrics["post_release_live_pages"] = None
        metrics["post_release_live_requests"] = None
        cleanup_metrics = metrics
        return codecs, generation_ms, decode_ms, {
            "audio_sample_counts": [int(torch.as_tensor(audio).numel()) for audio in audios],
            "adapter": cleanup_metrics,
            "stagger_steps": int(stagger_steps),
            "paged_attention_kernel": bool(use_paged_attention_kernel),
        }
    finally:
        for state in states:
            try:
                adapter.cache.release(state.request_id)
            except Exception:
                pass
        # Record the ownership invariant after all request rows are released.
        # The timing path itself is already complete at this point.
        if states:
            snapshot = adapter.cache.snapshot()
            if "cleanup_metrics" in locals():
                cleanup_metrics["post_release_live_pages"] = int(snapshot["live_pages"])
                cleanup_metrics["post_release_live_requests"] = int(snapshot["live_requests"])
        runner.close()
        adapter.close()


def run_ab_probe(
    *,
    model_path: str,
    device: str = "cuda:0",
    texts: Sequence[str] = ("你好，这是一个实时语音测试。", "请确认第二个会话也能正确合批。"),
    max_new_tokens: int = 5,
    page_size: int = 16,
    max_pages: int = 256,
    stagger_steps: int = 1,
    language: str = "chinese",
    speaker: str = "",
    warmup: int = 1,
    repeats: int = 2,
    use_paged_attention_kernel: bool = False,
    use_fused_qkv: bool = False,
    paged_attention_block_n: int = 128,
    paged_attention_num_warps: int = 4,
) -> dict[str, Any]:
    if len(texts) != 2:
        raise ValueError("full codec A/B probe requires exactly two texts")
    if int(max_new_tokens) < 2:
        raise ValueError("max_new_tokens must be at least 2 for codec decode")
    if int(warmup) < 0 or int(repeats) <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    from qwen_tts import Qwen3TTSModel
    from bench.qwen_tts_active_prefix_parity_probe import _prepare_talker_inputs_once

    wrapper = Qwen3TTSModel.from_pretrained(model_path, dtype=torch.float16, device_map=device)
    supported = wrapper.get_supported_speakers() or []
    backend = SimpleNamespace(_language=language, _speaker=speaker or supported[0])
    args = SimpleNamespace(max_new_tokens=int(max_new_tokens))
    prepared = _prepare_talker_inputs_once(wrapper, list(texts), backend, args)
    prepared["max_new_tokens"] = int(max_new_tokens)
    for _ in range(int(warmup)):
        _run_active_prefix(wrapper, prepared, device=device, max_new_tokens=max_new_tokens)
        _run_varlen(
            wrapper,
            prepared,
            device=device,
            page_size=int(page_size),
            max_pages=int(max_pages),
            stagger_steps=int(stagger_steps),
            use_paged_attention_kernel=bool(use_paged_attention_kernel),
            use_fused_qkv=bool(use_fused_qkv),
            paged_attention_block_n=int(paged_attention_block_n),
            paged_attention_num_warps=int(paged_attention_num_warps),
        )

    samples = []
    active_codecs = varlen_codecs = None
    varlen_metrics = {}
    for repeat_index in range(int(repeats)):
        if repeat_index % 2 == 0:
            active_codecs, active_generation_ms, active_decode_ms = _run_active_prefix(
                wrapper, prepared, device=device, max_new_tokens=max_new_tokens
            )
            varlen_codecs, varlen_generation_ms, varlen_decode_ms, varlen_metrics = _run_varlen(
                wrapper,
                prepared,
                device=device,
                page_size=int(page_size),
                max_pages=int(max_pages),
                stagger_steps=int(stagger_steps),
                use_paged_attention_kernel=bool(use_paged_attention_kernel),
                use_fused_qkv=bool(use_fused_qkv),
                paged_attention_block_n=int(paged_attention_block_n),
                paged_attention_num_warps=int(paged_attention_num_warps),
            )
        else:
            varlen_codecs, varlen_generation_ms, varlen_decode_ms, varlen_metrics = _run_varlen(
                wrapper,
                prepared,
                device=device,
                page_size=int(page_size),
                max_pages=int(max_pages),
                stagger_steps=int(stagger_steps),
                use_paged_attention_kernel=bool(use_paged_attention_kernel),
                use_fused_qkv=bool(use_fused_qkv),
                paged_attention_block_n=int(paged_attention_block_n),
                paged_attention_num_warps=int(paged_attention_num_warps),
            )
            active_codecs, active_generation_ms, active_decode_ms = _run_active_prefix(
                wrapper, prepared, device=device, max_new_tokens=max_new_tokens
            )
        samples.append(
            {
                "active_prefix_generation": active_generation_ms,
                "varlen_generation": varlen_generation_ms,
                "active_prefix_audio_decode": active_decode_ms,
                "varlen_audio_decode": varlen_decode_ms,
                "active_prefix_total": active_generation_ms + active_decode_ms,
                "varlen_total": varlen_generation_ms + varlen_decode_ms,
            }
        )
    parity = [
        torch.equal(active, candidate)
        for active, candidate in zip(active_codecs, varlen_codecs)
    ]
    parity_details = [
        _codec_parity_details(active, candidate)
        for active, candidate in zip(active_codecs, varlen_codecs)
    ]
    median = {
        key: statistics.median(sample[key] for sample in samples)
        for key in samples[0]
    }
    return {
        "schema": "qwen_tts_outer_varlen_full_codec_ab",
        "device": device,
        "physical_device_hint": "set CUDA_VISIBLE_DEVICES explicitly when mapping device=cuda:0",
        "model_path": model_path,
        "request_ids": ["short", "long"],
        "prompt_lengths": [
            int(prepared["attention_mask"][row].to(dtype=torch.long).sum().item())
            for row in range(2)
        ],
        "prompt_width": int(prepared["inputs_embeds"].shape[1]),
        "max_new_tokens": int(max_new_tokens),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "paged_attention_kernel": bool(use_paged_attention_kernel),
        "fused_qkv": bool(use_fused_qkv),
        "paged_attention_block_n": int(paged_attention_block_n),
        "paged_attention_num_warps": int(paged_attention_num_warps),
        "codec_frames": {
            "active_prefix": [int(codec.shape[1]) for codec in active_codecs],
            "varlen": [int(codec.shape[1]) for codec in varlen_codecs],
        },
        "codec_sha256": {
            "active_prefix": [_raw_codec_hash(codec) for codec in active_codecs],
            "varlen": [_raw_codec_hash(codec) for codec in varlen_codecs],
        },
        "first_codebook_ids": {
            "active_prefix": [codec[0, :, 0].detach().to(device="cpu").tolist() for codec in active_codecs],
            "varlen": [codec[0, :, 0].detach().to(device="cpu").tolist() for codec in varlen_codecs],
        },
        "parity": {
            "per_request_codec_exact": parity,
            "all_codec_exact": bool(all(parity)),
            "per_request": parity_details,
        },
        "timing_ms": {
            **median,
            "generation_speedup": median["active_prefix_generation"] / median["varlen_generation"]
            if median["varlen_generation"]
            else None,
            "total_speedup": median["active_prefix_total"] / median["varlen_total"]
            if median["varlen_total"]
            else None,
            "samples": samples,
        },
        "varlen": varlen_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-pages", type=int, default=256)
    parser.add_argument("--stagger-steps", type=int, default=1)
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--speaker", default="")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--paged-attention-kernel",
        action="store_true",
        help="use the Triton request-paged outer decode attention kernel",
    )
    parser.add_argument(
        "--fuse-qkv",
        action="store_true",
        help="fuse Q/K/V projection into one inference-only linear operation",
    )
    parser.add_argument("--paged-attention-block-n", type=int, default=128)
    parser.add_argument("--paged-attention-num-warps", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("results/qwen_tts_outer_varlen_full_codec_ab_cuda2.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    texts = tuple(args.texts) if args.texts else (
        "你好，这是一个实时语音测试。",
        "请确认第二个会话也能正确合批。",
    )
    report = run_ab_probe(
        model_path=args.model_path,
        device=args.device,
        texts=texts,
        max_new_tokens=args.max_new_tokens,
        page_size=args.page_size,
        max_pages=args.max_pages,
        stagger_steps=args.stagger_steps,
        language=args.language,
        speaker=args.speaker,
        warmup=args.warmup,
        repeats=args.repeats,
        use_paged_attention_kernel=args.paged_attention_kernel,
        use_fused_qkv=args.fuse_qkv,
        paged_attention_block_n=args.paged_attention_block_n,
        paged_attention_num_warps=args.paged_attention_num_warps,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["parity"]["all_codec_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
