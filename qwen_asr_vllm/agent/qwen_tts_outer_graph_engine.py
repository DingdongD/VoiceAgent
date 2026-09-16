"""Fixed-shape CUDA Graph runtime for the Qwen-TTS outer talker step."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Any

from qwen_asr_vllm.agent.qwen_tts_outer_static_engine import (
    OuterTalkerStaticEngine,
    OuterRuntimeLease,
    OuterStepOutput,
    PreparedOuterDecode,
)
from qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache import (
    ActivePrefixCache,
)


class OuterGraphEngineError(RuntimeError):
    """Raised when a fixed-shape outer CUDA Graph cannot be used."""


@dataclass(frozen=True)
class OuterGraphKey:
    batch_size: int
    device: str
    dtype: str
    hidden_size: int
    num_code_groups: int
    max_cache_len: int
    attention_implementation: str


class CUDAGraphOuterStepRuntime:
    """Acquire shape-keyed graph leases for the static outer engine.

    The runtime deliberately has no eager fallback. A graph lease owns one
    active-prefix cache and fixed-shape bundles; callers must opt into a
    separate eager runtime when graph constraints are not acceptable.
    """

    def __init__(
        self,
        talker: Any,
        *,
        max_graph_batch_size: int = 2,
        bundle_factory=None,
        device_context_factory=None,
        allow_non_cuda_for_testing: bool = False,
    ):
        self._talker = talker
        self._max_graph_batch_size = max(1, int(max_graph_batch_size))
        self._bundle_factory = bundle_factory or _create_outer_graph_bundle
        self._device_context_factory = device_context_factory or _device_context
        self._allow_non_cuda_for_testing = bool(allow_non_cuda_for_testing)
        self._bundles: dict[OuterGraphKey, Any] = {}
        self._bundles_lock = Lock()
        self._metrics_lock = Lock()
        self.graph_captures = 0
        self.graph_replays = 0
        self.graph_errors = 0
        self._attention_patch = None
        self.fusion_calibration = None

    def acquire(self, batch_size, device, dtype, max_cache_len) -> OuterRuntimeLease:
        batch_size = int(batch_size)
        if batch_size <= 0 or batch_size > self._max_graph_batch_size:
            raise OuterGraphEngineError(
                f"outer graph batch size {batch_size} exceeds configured limit "
                f"{self._max_graph_batch_size}"
            )
        device = _as_device(device)
        if device.type != "cuda" and not self._allow_non_cuda_for_testing:
            raise OuterGraphEngineError("outer CUDA Graph runtime requires CUDA")
        key = self._key(batch_size, device, dtype, int(max_cache_len))
        bundle = self._get_bundle(key)
        bundle.lock.acquire()
        return _OuterGraphLease(self, bundle)

    def build_attention_mask(
        self, prompt_attention_mask, cache_position, max_cache_len, dtype
    ):
        del max_cache_len, dtype
        import torch

        prompt_length = prompt_attention_mask.shape[1]
        live_length = int(cache_position) + 1
        if live_length < prompt_length:
            raise OuterGraphEngineError(
                "outer graph active-prefix mask is shorter than the prompt"
            )
        if bool(prompt_attention_mask.to(dtype=torch.bool).all().item()):
            # This is the same mask-skip case used by upstream SDPA. Keeping
            # it as None also avoids a GPU scalar read during graph capture.
            return None
        generated_mask = torch.ones(
            (prompt_attention_mask.shape[0], live_length - prompt_length),
            dtype=torch.bool,
            device=prompt_attention_mask.device,
        )
        allowed = torch.cat(
            (prompt_attention_mask.to(dtype=torch.bool), generated_mask), dim=1
        )
        # Query length is one and cache_position is the last live token, so
        # every live key is causally visible. This is the 4D mask produced by
        # Transformers for the padded SDPA path, materialized before capture.
        return allowed[:, None, None, :]

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            result = {
                "graph_captures": int(self.graph_captures),
                "graph_replays": int(self.graph_replays),
                "graph_errors": int(self.graph_errors),
                "bundles": len(self._bundles),
            }
            if self.fusion_calibration is not None:
                result["fusion_calibration"] = self.fusion_calibration
            return result

    def _key(self, batch_size, device, dtype, max_cache_len) -> OuterGraphKey:
        config = getattr(self._talker, "config", None)
        return OuterGraphKey(
            batch_size=batch_size,
            device=str(device),
            dtype=str(dtype),
            hidden_size=int(getattr(config, "hidden_size", 0)),
            num_code_groups=int(getattr(config, "num_code_groups", 0)),
            max_cache_len=max_cache_len,
            attention_implementation=str(
                getattr(config, "_attn_implementation", "eager")
            ),
        )

    def _get_bundle(self, key: OuterGraphKey):
        with self._bundles_lock:
            bundle = self._bundles.get(key)
            if bundle is None:
                try:
                    bundle = self._bundle_factory(
                        talker=self._talker,
                        batch_size=key.batch_size,
                        device=_as_device(key.device),
                        dtype=_as_dtype(key.dtype),
                        max_cache_len=key.max_cache_len,
                        capture_callback=self._record_capture,
                        device_context_factory=self._device_context_factory,
                    )
                except Exception:
                    with self._metrics_lock:
                        self.graph_errors += 1
                    raise
                self._bundles[key] = bundle
            return bundle

    def _record_replay(self):
        with self._metrics_lock:
            self.graph_replays += 1

    def _record_capture(self):
        with self._metrics_lock:
            self.graph_captures += 1

    def _record_error(self):
        with self._metrics_lock:
            self.graph_errors += 1

    def close(self) -> None:
        patch = self._attention_patch
        self._attention_patch = None
        if patch is not None:
            patch.close()
        with self._bundles_lock:
            bundles = list(self._bundles.values())
            self._bundles.clear()
        for bundle in bundles:
            close = getattr(bundle, "close", None)
            if callable(close):
                close()


class _OuterGraphLease(OuterRuntimeLease):
    def __init__(self, runtime: CUDAGraphOuterStepRuntime, bundle: Any):
        super().__init__(bundle.cache)
        self._runtime = runtime
        self._bundle = bundle
        self._released = False

    def run(self, prepared_decode: PreparedOuterDecode) -> OuterStepOutput:
        if self._released:
            raise OuterGraphEngineError("outer graph lease is released")
        try:
            result = self._bundle.run(prepared_decode)
        except Exception:
            self._runtime._record_error()
            raise
        self._runtime._record_replay()
        return result

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._bundle.lock.release()


class _OuterGraphBundle:
    def __init__(
        self,
        talker,
        batch_size,
        device,
        dtype,
        max_cache_len,
        capture_callback=None,
        device_context_factory=None,
    ):
        import torch
        self.talker = talker
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self._capture_callback = capture_callback
        self._device_context_factory = device_context_factory or _device_context
        self.max_cache_len = int(max_cache_len)
        self.cache = ActivePrefixCache(
            num_hidden_layers=talker.config.num_hidden_layers,
            max_cache_len=self.max_cache_len,
        )
        self.lock = RLock()
        self._graphs: dict[tuple[bool, int, bool], _CapturedOuterGraph] = {}

    def close(self) -> None:
        self._graphs.clear()
        release = getattr(self.cache, "release", None)
        if callable(release):
            release()

    def reset(self):
        self.cache.reset()

    def run(self, prepared: PreparedOuterDecode) -> OuterStepOutput:
        self._validate_prepared(prepared)
        cache_position = int(prepared.cache_position[0].item())
        has_padding_mask = prepared.attention_mask is not None
        captured = self._graphs.get(
            (bool(prepared.output_hidden_states), cache_position, has_padding_mask)
        )
        if captured is None:
            captured = self._capture(prepared)
            self._graphs[
                (bool(prepared.output_hidden_states), cache_position, has_padding_mask)
            ] = captured
        captured.codec_ids.copy_(prepared.codec_ids)
        captured.condition.copy_(prepared.condition)
        if captured.attention_mask is not None:
            captured.attention_mask.copy_(prepared.attention_mask)
        captured.position_ids.copy_(prepared.position_ids)
        captured.cache_position.copy_(prepared.cache_position)
        with self._device_context_factory(self.device):
            captured.graph.replay()
        self._set_active_cache_length(cache_position + 1)
        return OuterStepOutput(
            logits=captured.logits.clone(),
            past_hidden=captured.past_hidden.clone(),
            hidden_states=_clone_tree(captured.hidden_states),
        )

    def _set_active_cache_length(self, length: int) -> None:
        # Cache.update runs during capture, but its Python length bookkeeping
        # is not replayed by CUDA Graph. Replay the state transition here.
        for layer in getattr(self.cache, "layers", ()):
            if hasattr(layer, "active_length"):
                layer.active_length = int(length)

    def _validate_prepared(self, prepared):
        if prepared.codec_ids.shape != (
            self.batch_size,
            self.talker.config.num_code_groups,
        ):
            raise OuterGraphEngineError("codec_ids shape does not match graph bundle")
        hidden_size = int(self.talker.config.hidden_size)
        if prepared.condition.shape != (self.batch_size, 1, hidden_size):
            raise OuterGraphEngineError("condition shape does not match graph bundle")
        expected_cache_position = int(prepared.cache_position[0].item())
        if prepared.attention_mask is not None:
            if prepared.attention_mask.shape != (
                self.batch_size,
                1,
                1,
                expected_cache_position + 1,
            ):
                raise OuterGraphEngineError(
                    "attention mask shape does not match graph bundle"
                )
        if prepared.position_ids.shape != (3, self.batch_size, 1):
            raise OuterGraphEngineError("position IDs shape does not match graph bundle")
        if prepared.cache_position.shape != (1,):
            raise OuterGraphEngineError("cache position shape does not match graph bundle")

    def _capture(self, prepared):
        import torch

        static_codec_ids = prepared.codec_ids.clone()
        static_condition = prepared.condition.clone()
        static_attention_mask = (
            prepared.attention_mask.clone()
            if prepared.attention_mask is not None
            else None
        )
        static_position_ids = prepared.position_ids.clone()
        static_cache_position = prepared.cache_position.clone()
        prefix_length = int(prepared.cache_position[0].item())
        snapshot = _snapshot_cache_prefix(self.cache, prefix_length)
        graph = torch.cuda.CUDAGraph()
        try:
            with self._device_context_factory(self.device):
                for _ in range(2):
                    self._forward(
                        static_codec_ids,
                        static_condition,
                        static_attention_mask,
                        static_position_ids,
                        static_cache_position,
                        prepared.output_hidden_states,
                    )
                    _restore_cache_prefix(snapshot)
                torch.cuda.synchronize(self.device)
                with torch.cuda.graph(graph):
                    outputs = self._forward(
                        static_codec_ids,
                        static_condition,
                        static_attention_mask,
                        static_position_ids,
                        static_cache_position,
                        prepared.output_hidden_states,
                    )
                    hidden = outputs.last_hidden_state
                    logits = self.talker.codec_head(hidden)
                    past_hidden = hidden[:, -1:, :]
                    hidden_states = getattr(outputs, "hidden_states", None)
                _restore_cache_prefix(snapshot)
                torch.cuda.synchronize(self.device)
        except Exception as error:
            with self._device_context_factory(self.device):
                _restore_cache_prefix(snapshot)
            raise OuterGraphEngineError(
                f"outer CUDA Graph capture failed: {type(error).__name__}: {error}"
            ) from error
        if self._capture_callback is not None:
            self._capture_callback()
        return _CapturedOuterGraph(
            graph=graph,
            codec_ids=static_codec_ids,
            condition=static_condition,
            attention_mask=static_attention_mask,
            position_ids=static_position_ids,
            cache_position=static_cache_position,
            logits=logits,
            past_hidden=past_hidden,
            hidden_states=hidden_states,
        )

    def _forward(
        self,
        codec_ids,
        condition,
        attention_mask,
        position_ids,
        cache_position,
        output_hidden_states,
    ):
        import torch

        codec_hiddens = [self.talker.get_input_embeddings()(codec_ids[:, :1])]
        predictor_embeddings = self.talker.code_predictor.get_input_embeddings()
        codec_hiddens.extend(
            predictor_embeddings[index](codec_ids[:, index + 1 : index + 2])
            for index in range(self.talker.config.num_code_groups - 1)
        )
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
        inputs_embeds = inputs_embeds + condition
        return self.talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=output_hidden_states,
        )


@dataclass
class _CapturedOuterGraph:
    graph: Any
    codec_ids: Any
    condition: Any
    attention_mask: Any
    position_ids: Any
    cache_position: Any
    logits: Any
    past_hidden: Any
    hidden_states: Any


def _create_outer_graph_bundle(**kwargs):
    return _OuterGraphBundle(**kwargs)


def install_cuda_graph_outer_talker(
    talker: Any,
    *,
    max_cache_len: int = 16384,
    max_graph_batch_size: int = 2,
    fuse_qkv: bool = False,
    fuse_attention: bool = False,
    fused_projection_kernel: bool = False,
    fusion_policy: str = "allow",
    fusion_max_abs_error: float = 2.0e-3,
    fusion_max_relative_l2: float = 2.0e-4,
) -> bool:
    """Install the fixed-shape outer engine and optionally fused static attention."""

    if fusion_policy not in {"allow", "strict"}:
        raise OuterGraphEngineError(
            f"unsupported outer fusion policy: {fusion_policy}"
        )
    if getattr(talker.generate, "_qav_static_outer_talker", False):
        return False
    from qwen_asr_vllm.agent.qwen_tts_paged_attention import (
        install_qwen_paged_decode_attention,
    )

    attention_patch = None
    if fused_projection_kernel:
        fuse_qkv = True
    if fuse_qkv:
        attention_patch = install_qwen_paged_decode_attention(
            talker,
            fuse_qkv=True,
            patch_static_cache=True,
            fuse_static_attention=fuse_attention,
            fused_projection_kernel=fused_projection_kernel,
        )
    elif fuse_attention:
        attention_patch = install_qwen_paged_decode_attention(
            talker,
            patch_static_cache=True,
            fuse_static_attention=True,
        )
    runtime = CUDAGraphOuterStepRuntime(
        talker,
        max_graph_batch_size=max_graph_batch_size,
    )
    runtime._attention_patch = attention_patch
    if fused_projection_kernel or (fusion_policy == "strict" and fuse_qkv):
        from qwen_asr_vllm.agent.qwen_tts_paged_attention import calibrate_fused_qkv

        backend = "triton" if fused_projection_kernel else "torch"
        try:
            calibration = calibrate_fused_qkv(talker, backend=backend)
            runtime.fusion_calibration = calibration
            if fusion_policy == "strict" and (
                calibration["max_abs"] > float(fusion_max_abs_error)
                or calibration["max_relative_l2"] > float(fusion_max_relative_l2)
            ):
                raise OuterGraphEngineError(
                    "fused projection calibration exceeded tolerance: "
                    f"max_abs={calibration['max_abs']:.6g}, "
                    f"relative_l2={calibration['max_relative_l2']:.6g}"
                )
        except BaseException:
            runtime.close()
            raise
    engine = OuterTalkerStaticEngine(
        talker,
        max_cache_len=max_cache_len,
        step_runtime=runtime,
    )
    original_generate = talker.generate

    def graph_generate(**kwargs):
        return engine.generate(**kwargs)

    graph_generate._qav_static_outer_talker = True  # type: ignore[attr-defined]
    graph_generate._qav_cuda_graph_outer_talker = True  # type: ignore[attr-defined]
    graph_generate._qav_original_generate = original_generate  # type: ignore[attr-defined]
    graph_generate._qav_outer_engine = engine  # type: ignore[attr-defined]
    talker.generate = graph_generate
    return True


def _clone_tree(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return value.clone()


def _snapshot_cache_prefix(cache, prefix_length):
    snapshots = []
    for layer in getattr(cache, "layers", ()):
        if not getattr(layer, "is_initialized", False):
            continue
        snapshots.append(
            (
                layer,
                layer.keys[..., :prefix_length, :].clone(),
                layer.values[..., :prefix_length, :].clone(),
                int(getattr(layer, "active_length", prefix_length)),
            )
        )
    return snapshots


def _restore_cache_prefix(snapshots):
    for layer, keys, values, active_length in snapshots:
        if hasattr(layer, "active_length"):
            layer.active_length = 0
        layer.keys[..., : keys.shape[-2], :].copy_(keys)
        layer.values[..., : values.shape[-2], :].copy_(values)
        if hasattr(layer, "active_length"):
            layer.active_length = active_length


def _as_device(device):
    import torch

    return device if isinstance(device, torch.device) else torch.device(device)


def _as_dtype(dtype):
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).split(".")[-1]
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise OuterGraphEngineError(f"unsupported graph dtype: {dtype}")
    return value


@contextmanager
def _device_context(device):
    import torch

    device = _as_device(device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            yield
    else:
        yield
