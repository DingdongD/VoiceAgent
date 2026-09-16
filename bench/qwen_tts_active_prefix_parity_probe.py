"""Exact upstream versus active-prefix Qwen-TTS boundary parity probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ParityProbeError(RuntimeError):
    """Raised when an installed model API cannot satisfy the probe contract."""


class ArgumentParseFailure(RuntimeError):
    """Raised instead of argparse's parity-colliding SystemExit(2)."""


class _AuditableArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ArgumentParseFailure(message)


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    contiguous: bool
    dtype: str
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "contiguous": self.contiguous,
            "dtype": self.dtype,
            "device": self.device,
        }


@dataclass(frozen=True)
class TensorSnapshot:
    name: str
    tensor: Any
    layout: TensorLayout

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "layout": self.layout.to_dict()}


@dataclass(frozen=True)
class TensorComparison:
    name: str
    bitwise_equal: bool
    shape_equal: bool
    dtype_equal: bool
    max_abs: float | None
    first_difference: tuple[int, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bitwise_equal": self.bitwise_equal,
            "shape_equal": self.shape_equal,
            "dtype_equal": self.dtype_equal,
            "max_abs": self.max_abs,
            "first_difference": (
                list(self.first_difference)
                if self.first_difference is not None
                else None
            ),
        }


@dataclass(frozen=True)
class LayerKvCapture:
    layer_index: int
    key: TensorSnapshot
    value: TensorSnapshot

    def metadata(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "key": self.key.metadata(),
            "value": self.value.metadata(),
        }


@dataclass(frozen=True)
class BoundaryCapture:
    call_index: int
    phase: str
    layers: tuple[LayerKvCapture, ...]
    last_hidden: TensorSnapshot
    attention_evidence: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "phase": self.phase,
            "layers": [layer.metadata() for layer in self.layers],
            "last_hidden": self.last_hidden.metadata(),
            "attention_evidence": _plain_copy(self.attention_evidence),
        }


@dataclass(frozen=True)
class CodecCapture:
    raw_codec: TensorSnapshot
    first_codebook_ids: TensorSnapshot
    predictor_sequences: TensorSnapshot
    raw_sha256: str
    combined_sha256: str
    item_sha256: tuple[str, ...]
    item_shapes: tuple[tuple[int, ...], ...]
    item_dtypes: tuple[str, ...]
    eos_positions: tuple[int | None, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "raw_codec": self.raw_codec.metadata(),
            "first_codebook_ids": self.first_codebook_ids.metadata(),
            "predictor_sequences": self.predictor_sequences.metadata(),
            "raw_sha256": self.raw_sha256,
            "combined_sha256": self.combined_sha256,
            "item_sha256": list(self.item_sha256),
            "item_shapes": [list(shape) for shape in self.item_shapes],
            "item_dtypes": list(self.item_dtypes),
            "eos_positions": list(self.eos_positions),
            "hash_scheme": (
                "SHA-256 of ordered CPU-contiguous tensor bytes; per-item "
                "hashes preserve tensor boundaries"
            ),
        }


@dataclass(frozen=True)
class AudioCapture:
    sample_rate: int
    sample_counts: tuple[int, ...]
    finite: tuple[bool, ...]
    nonempty: tuple[bool, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "sample_counts": list(self.sample_counts),
            "finite": list(self.finite),
            "nonempty": list(self.nonempty),
        }


@dataclass(frozen=True)
class ModeCapture:
    mode: str
    batch_size: int
    boundaries: tuple[BoundaryCapture, ...]
    raw_codec_logits: tuple[TensorSnapshot, ...]
    processed_outer_logits: tuple[TensorSnapshot, ...]
    sampled_first_codebook: TensorSnapshot
    codec: CodecCapture
    audio: AudioCapture
    attention_evidence: dict[str, Any]
    runtime_metrics: dict[str, Any]
    timings_ms: dict[str, float]
    resolved_config: dict[str, Any]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "batch_size": self.batch_size,
            "boundaries": [boundary.metadata() for boundary in self.boundaries],
            "raw_codec_logits": [item.metadata() for item in self.raw_codec_logits],
            "processed_outer_logits": [
                item.metadata() for item in self.processed_outer_logits
            ],
            "sampled_first_codebook": self.sampled_first_codebook.metadata(),
            "codec": self.codec.metadata(),
            "audio": self.audio.to_dict(),
            "attention_evidence": _plain_copy(self.attention_evidence),
            "runtime_metrics": _plain_copy(self.runtime_metrics),
            "timings_ms": _plain_copy(self.timings_ms),
            "resolved_config": _plain_copy(self.resolved_config),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ParitySummary:
    parity_passed: bool
    failed_reasons: tuple[str, ...]
    boundary_comparisons: tuple[dict[str, Any], ...]
    codec_comparisons: dict[str, Any]
    first_codec_divergence: dict[str, int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parity_passed": self.parity_passed,
            "failed_reasons": list(self.failed_reasons),
            "boundary_comparisons": _plain_copy(self.boundary_comparisons),
            "codec_comparisons": _plain_copy(self.codec_comparisons),
            "first_codec_divergence": _plain_copy(self.first_codec_divergence),
        }


@dataclass
class HookRecorder:
    boundaries: list[BoundaryCapture] = field(default_factory=list)
    raw_codec_logits: list[TensorSnapshot] = field(default_factory=list)
    processed_outer_logits: list[TensorSnapshot] = field(default_factory=list)
    predictor_sequences: list[TensorSnapshot] = field(default_factory=list)
    sampled_first_codebook: list[TensorSnapshot] = field(default_factory=list)
    attention_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeDependencies:
    execute: (
        Callable[
            [argparse.Namespace], Sequence[tuple[int, ModeCapture, ModeCapture]]
        ]
        | None
    ) = None
    environment: Callable[[argparse.Namespace], dict[str, Any]] | None = None
    parse_args: Callable[[list[str] | None], argparse.Namespace] | None = None
    stability_execute: Callable[[argparse.Namespace], "StabilityExecution"] | None = None


@dataclass(frozen=True)
class StabilityExecution:
    request_results: tuple[dict[str, Any], ...]
    elapsed_seconds: float
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    primary_error: BaseException | None = None
    primary_traceback: Any = None


@dataclass(frozen=True)
class _AttributeState:
    had_instance_value: bool
    instance_value: Any


@dataclass
class _FailureAccumulator:
    error: BaseException | None = None
    traceback: Any = None

    def record(self, error: BaseException, *, label: str | None = None) -> None:
        if self.error is None:
            self.error = error
            self.traceback = error.__traceback__
            return
        if label is None:
            label = "secondary operation"
        self.error.add_note(
            f"{label} also failed: {type(error).__name__}: {error}"
        )

    def run(self, label: str, operation: Callable[[], Any]) -> None:
        try:
            operation()
        except BaseException as error:
            self.record(error, label=label)

    def raise_if_present(self) -> None:
        if self.error is not None:
            raise self.error.with_traceback(self.traceback)


@dataclass
class _ProbeWarningCapture:
    python_warnings: list[Any]
    log_messages: list[str] = field(default_factory=list)

    def messages(self) -> list[str]:
        messages = []
        for captured in self.python_warnings:
            message = (
                f"python.warnings: {captured.category.__name__}: "
                f"{str(captured.message).strip()}"
            )
            if message not in messages:
                messages.append(message)
        for message in self.log_messages:
            if message not in messages:
                messages.append(message)
        return messages


class _ProbeLogHandler(logging.Handler):
    def __init__(self, destination: list[str]):
        super().__init__(level=logging.WARNING)
        self._destination = destination

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._destination.append(
                f"python.logging: {record.name}: {record.levelname}: "
                f"{record.getMessage()}"
            )
        except Exception:
            self.handleError(record)


class _PreparedInputsCaptured(Exception):
    pass


def capture_tensor(name: str, tensor: Any) -> TensorSnapshot:
    """Capture tensor layout before making a detached CPU instrumentation copy."""

    import torch

    if not torch.is_tensor(tensor):
        raise ParityProbeError(f"{name} is not a tensor")
    layout = TensorLayout(
        shape=tuple(int(size) for size in tensor.shape),
        stride=tuple(int(size) for size in tensor.stride()),
        contiguous=bool(tensor.is_contiguous()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
    )
    copied = tensor.detach().to(device="cpu").clone()
    return TensorSnapshot(name=name, tensor=copied, layout=layout)


def compare_tensors(name: str, reference: Any, candidate: Any) -> TensorComparison:
    """Compare tensors by exact dtype, shape, and element bit representation."""

    import torch

    reference_tensor = _snapshot_tensor(reference)
    candidate_tensor = _snapshot_tensor(candidate)
    if not torch.is_tensor(reference_tensor) or not torch.is_tensor(candidate_tensor):
        raise TypeError("compare_tensors requires tensor inputs")

    shape_equal = tuple(reference_tensor.shape) == tuple(candidate_tensor.shape)
    dtype_equal = reference_tensor.dtype == candidate_tensor.dtype
    bitwise_equal = False
    first_difference = None
    max_abs = None

    if shape_equal:
        max_abs = _safe_max_abs(reference_tensor, candidate_tensor)
    if shape_equal and dtype_equal:
        reference_bytes = _tensor_bytes(reference_tensor)
        candidate_bytes = _tensor_bytes(candidate_tensor)
        byte_equal = reference_bytes.equal(candidate_bytes)
        bitwise_equal = bool(byte_equal)
        if not bitwise_equal:
            element_size = max(1, int(reference_tensor.element_size()))
            changed = reference_bytes.reshape(-1, element_size).ne(
                candidate_bytes.reshape(-1, element_size)
            ).any(dim=1)
            flat_index = int(changed.nonzero(as_tuple=False)[0, 0])
            first_difference = _unravel_index(flat_index, tuple(reference_tensor.shape))

    return TensorComparison(
        name=name,
        bitwise_equal=bitwise_equal,
        shape_equal=shape_equal,
        dtype_equal=dtype_equal,
        max_abs=max_abs,
        first_difference=first_difference,
    )


def first_codec_divergence(reference: Any, candidate: Any) -> dict[str, int] | None:
    reference_tensor = _snapshot_tensor(reference)
    candidate_tensor = _snapshot_tensor(candidate)
    comparison = compare_tensors("raw_codec", reference_tensor, candidate_tensor)
    if comparison.bitwise_equal:
        return None
    if (
        not comparison.shape_equal
        or not comparison.dtype_equal
        or comparison.first_difference is None
        or len(comparison.first_difference) != 3
    ):
        return None
    batch_index, frame_index, codebook_index = comparison.first_difference
    return {
        "batch_index": batch_index,
        "frame_index": frame_index,
        "codebook_index": codebook_index,
        "reference_id": int(reference_tensor[batch_index, frame_index, codebook_index]),
        "candidate_id": int(candidate_tensor[batch_index, frame_index, codebook_index]),
    }


def build_codec_capture(
    raw_codec: Any,
    *,
    predictor_sequences: Any,
    decoded_codec_items: Sequence[Any],
    eos_token_id: int,
) -> CodecCapture:
    import torch

    if not torch.is_tensor(raw_codec) or raw_codec.ndim != 3:
        raise ParityProbeError("raw codec tensor must have shape [batch, frames, groups]")
    if raw_codec.shape[0] != len(decoded_codec_items):
        raise ParityProbeError("decoded codec item count does not match raw codec batch")
    first_codebook = raw_codec[:, :, 0]
    raw_cpu = raw_codec.detach().to(device="cpu")
    raw_sha256 = hashlib.sha256(
        _tensor_raw_bytes(raw_cpu.contiguous())
    ).hexdigest()
    eos_positions = []
    for row in raw_cpu[:, :, 0]:
        matches = row.eq(int(eos_token_id)).nonzero(as_tuple=False)
        eos_positions.append(int(matches[0, 0]) if matches.numel() else None)

    combined = hashlib.sha256()
    item_hashes = []
    item_shapes = []
    item_dtypes = []
    for index, item in enumerate(decoded_codec_items):
        if not torch.is_tensor(item):
            item = torch.as_tensor(item)
        cpu_item = item.detach().to(device="cpu").contiguous()
        raw_bytes = _tensor_raw_bytes(cpu_item)
        combined.update(raw_bytes)
        item_hashes.append(hashlib.sha256(raw_bytes).hexdigest())
        item_shapes.append(tuple(int(size) for size in cpu_item.shape))
        item_dtypes.append(str(cpu_item.dtype))

    return CodecCapture(
        raw_codec=capture_tensor("raw_codec", raw_codec),
        first_codebook_ids=capture_tensor("first_codebook_ids", first_codebook),
        predictor_sequences=capture_tensor(
            "predictor_sequences", predictor_sequences
        ),
        raw_sha256=raw_sha256,
        combined_sha256=combined.hexdigest(),
        item_sha256=tuple(item_hashes),
        item_shapes=tuple(item_shapes),
        item_dtypes=tuple(item_dtypes),
        eos_positions=tuple(eos_positions),
    )


def build_audio_capture(audios: Sequence[Any], *, sample_rate: int) -> AudioCapture:
    arrays = [np.asarray(audio).reshape(-1) for audio in audios]
    return AudioCapture(
        sample_rate=int(sample_rate),
        sample_counts=tuple(int(array.size) for array in arrays),
        finite=tuple(bool(np.isfinite(array).all()) for array in arrays),
        nonempty=tuple(bool(array.size) for array in arrays),
    )


def clone_prepared_inputs(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_prepared_inputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_prepared_inputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_prepared_inputs(item) for item in value)
    return value


@contextmanager
def _capture_probe_warnings() -> Iterator[_ProbeWarningCapture]:
    """Capture Python warnings and relevant logger warnings without leaking handlers."""

    logger_names = ("", "transformers", "qwen_tts", "qwen_asr_vllm")
    logger_states = []
    with warnings.catch_warnings(record=True) as python_warnings:
        warnings.simplefilter("always")
        capture = _ProbeWarningCapture(python_warnings=python_warnings)
        handler = _ProbeLogHandler(capture.log_messages)
        try:
            for name in logger_names:
                logger = logging.getLogger(name)
                logger_states.append((logger, logger.level, logger.disabled))
                logger.addHandler(handler)
                logger.disabled = False
                if logger.getEffectiveLevel() > logging.WARNING:
                    logger.setLevel(logging.WARNING)
            yield capture
        finally:
            for logger, level, disabled in reversed(logger_states):
                logger.removeHandler(handler)
                logger.setLevel(level)
                logger.disabled = disabled
            handler.close()


@contextmanager
def _capture_talker_hooks(
    talker: Any,
    *,
    mode: str,
    recorder: HookRecorder,
    sampler_module: Any | None = None,
    active_engine: Any | None = None,
) -> Iterator[None]:
    """Install all boundary hooks and restore each mutation in ``finally``."""

    if mode not in ("upstream", "active_prefix"):
        raise ValueError(f"unsupported probe mode: {mode}")
    model = _required_member(talker, "model")
    codec_head = _required_member(talker, "codec_head")
    predictor = _required_member(talker, "code_predictor")
    original_model_forward = _required_callable(model, "forward")
    original_head_forward = _required_callable(codec_head, "forward")
    original_predictor_generate = _required_callable(predictor, "generate")
    original_logits_processor = _required_callable(talker, "_get_logits_processor")
    original_talker_forward = (
        _required_callable(talker, "forward") if mode == "upstream" else None
    )
    states: list[tuple[str, Any, str, _AttributeState]] = []

    def replace(label: str, owner: Any, name: str, value: Any) -> None:
        states.append(
            (label, owner, name, _capture_attribute_state(owner, name))
        )
        setattr(owner, name, value)

    def model_forward(*args, **kwargs):
        outputs = original_model_forward(*args, **kwargs)
        if len(recorder.boundaries) < 5:
            cache = getattr(outputs, "past_key_values", None)
            if cache is None:
                cache = kwargs.get("past_key_values")
            hidden = getattr(outputs, "last_hidden_state", None)
            recorder.boundaries.append(
                _capture_cache_boundary(
                    cache,
                    hidden,
                    call_index=len(recorder.boundaries),
                    attention_evidence=recorder.attention_evidence,
                )
            )
        return outputs

    def head_forward(*args, **kwargs):
        outputs = original_head_forward(*args, **kwargs)
        if len(recorder.raw_codec_logits) < 5:
            recorder.raw_codec_logits.append(
                capture_tensor(f"raw_codec_logits.{len(recorder.raw_codec_logits)}", outputs)
            )
        return outputs

    def predictor_generate(*args, **kwargs):
        outputs = original_predictor_generate(*args, **kwargs)
        sequences = getattr(outputs, "sequences", None)
        recorder.predictor_sequences.append(
            capture_tensor(
                f"predictor_sequences.{len(recorder.predictor_sequences)}",
                sequences,
            )
        )
        return outputs

    def talker_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        inputs_embeds = kwargs.get("inputs_embeds")
        is_prefill = inputs_embeds is not None and getattr(inputs_embeds, "shape", (0, 0))[1] > 1
        if input_ids is not None and not is_prefill:
            recorder.sampled_first_codebook.append(
                capture_tensor(
                    f"sampled_first_codebook.{len(recorder.sampled_first_codebook)}",
                    input_ids,
                )
            )
        return original_talker_forward(*args, **kwargs)

    if original_talker_forward is not None:
        talker_forward = wraps(original_talker_forward)(talker_forward)

    def get_logits_processor(*args, **kwargs):
        processors = original_logits_processor(*args, **kwargs)
        append = getattr(processors, "append", None)
        if not callable(append):
            raise ParityProbeError(
                "talker._get_logits_processor did not return a public processor list"
            )

        def capture_processed(input_ids, scores):
            if len(recorder.processed_outer_logits) < 5:
                recorder.processed_outer_logits.append(
                    capture_tensor(
                        f"processed_outer_logits.{len(recorder.processed_outer_logits)}",
                        scores,
                    )
                )
            return scores

        append(capture_processed)
        return processors

    failure = _FailureAccumulator()
    try:
        replace("talker.model.forward", model, "forward", model_forward)
        replace(
            "talker.codec_head.forward",
            codec_head,
            "forward",
            head_forward,
        )
        replace(
            "talker.code_predictor.generate",
            predictor,
            "generate",
            predictor_generate,
        )
        replace(
            "talker._get_logits_processor",
            talker,
            "_get_logits_processor",
            get_logits_processor,
        )
        if mode == "upstream":
            replace("talker.forward", talker, "forward", talker_forward)

        if mode == "active_prefix":
            if sampler_module is None:
                import qwen_asr_vllm.agent.qwen_tts_outer_static_engine as sampler_module
            original_sampler = _required_callable(sampler_module, "_sample_next_token")

            def sample_next_token(scores, *args, **kwargs):
                if len(recorder.processed_outer_logits) < 5:
                    recorder.processed_outer_logits.append(
                        capture_tensor(
                            f"processed_outer_logits.{len(recorder.processed_outer_logits)}",
                            scores,
                        )
                    )
                return original_sampler(scores, *args, **kwargs)

            replace(
                "sampler_module._sample_next_token",
                sampler_module,
                "_sample_next_token",
                sample_next_token,
            )
            if active_engine is None:
                active_generate = _required_callable(talker, "generate")
                active_engine = getattr(active_generate, "_qav_outer_engine", None)
            if active_engine is None:
                raise ParityProbeError("active-prefix outer engine metadata is missing")
            original_decode_step = _required_callable(active_engine, "_decode_step")

            @wraps(original_decode_step)
            def decode_step(*args, **kwargs):
                result = original_decode_step(*args, **kwargs)
                try:
                    codec_ids = result[0]
                    final_first_ids = codec_ids[:, :1]
                except (IndexError, TypeError) as error:
                    raise ParityProbeError(
                        "active-prefix _decode_step did not return final codec IDs"
                    ) from error
                recorder.sampled_first_codebook.append(
                    capture_tensor(
                        f"sampled_first_codebook.{len(recorder.sampled_first_codebook)}",
                        final_first_ids,
                    )
                )
                return result

            replace(
                "active_engine._decode_step",
                active_engine,
                "_decode_step",
                decode_step,
            )

        yield
    except BaseException as error:
        failure.record(error)
    finally:
        for label, owner, name, state in reversed(states):
            failure.run(
                f"{label} restoration",
                lambda owner=owner, name=name, state=state: (
                    _restore_attribute_state(owner, name, state)
                ),
            )
    failure.raise_if_present()


def summarize_parity(reference: ModeCapture, candidate: ModeCapture) -> ParitySummary:
    reasons: list[str] = []
    boundary_reports = []
    required_count = 5
    if reference.mode != "upstream" or candidate.mode != "active_prefix":
        _add_reason(reasons, "mode_label_mismatch")
    if reference.batch_size != candidate.batch_size:
        _add_reason(reasons, "mode_batch_size_mismatch")
    if len(reference.boundaries) != required_count or len(candidate.boundaries) != required_count:
        _add_reason(reasons, "required_boundary_count_mismatch")
    if (
        len(reference.raw_codec_logits) != required_count
        or len(candidate.raw_codec_logits) != required_count
    ):
        _add_reason(reasons, "raw_codec_logits_count_mismatch")
    if (
        len(reference.processed_outer_logits) != required_count
        or len(candidate.processed_outer_logits) != required_count
    ):
        _add_reason(reasons, "processed_outer_logits_count_mismatch")

    common_boundaries = min(
        required_count, len(reference.boundaries), len(candidate.boundaries)
    )
    for index in range(common_boundaries):
        left = reference.boundaries[index]
        right = candidate.boundaries[index]
        report: dict[str, Any] = {
            "call_index": index,
            "phase": {"reference": left.phase, "candidate": right.phase},
            "reference_attention_evidence": _plain_copy(left.attention_evidence),
            "candidate_attention_evidence": _plain_copy(right.attention_evidence),
            "layers": [],
        }
        if left.call_index != right.call_index or left.phase != right.phase:
            _add_reason(reasons, "boundary_structure_mismatch")
        if len(left.layers) != len(right.layers):
            _add_reason(reasons, "boundary_structure_mismatch")
        hidden = compare_tensors(
            f"boundary.{index}.last_hidden", left.last_hidden, right.last_hidden
        )
        report["last_hidden"] = hidden.to_dict()
        if not hidden.bitwise_equal:
            _add_reason(reasons, "boundary_bitwise_mismatch")
        for layer_index in range(min(len(left.layers), len(right.layers))):
            left_layer = left.layers[layer_index]
            right_layer = right.layers[layer_index]
            if left_layer.layer_index != right_layer.layer_index:
                _add_reason(reasons, "boundary_structure_mismatch")
            key = compare_tensors(
                f"boundary.{index}.layer.{layer_index}.key",
                left_layer.key,
                right_layer.key,
            )
            value = compare_tensors(
                f"boundary.{index}.layer.{layer_index}.value",
                left_layer.value,
                right_layer.value,
            )
            report["layers"].append(
                {
                    "layer_index": layer_index,
                    "key": key.to_dict(),
                    "value": value.to_dict(),
                    "reference_layout": {
                        "key": left_layer.key.layout.to_dict(),
                        "value": left_layer.value.layout.to_dict(),
                    },
                    "candidate_layout": {
                        "key": right_layer.key.layout.to_dict(),
                        "value": right_layer.value.layout.to_dict(),
                    },
                }
            )
            if not key.bitwise_equal or not value.bitwise_equal:
                _add_reason(reasons, "boundary_bitwise_mismatch")

        if index < len(reference.raw_codec_logits) and index < len(candidate.raw_codec_logits):
            raw_logits = compare_tensors(
                f"boundary.{index}.raw_codec_logits",
                reference.raw_codec_logits[index],
                candidate.raw_codec_logits[index],
            )
            report["raw_codec_logits"] = raw_logits.to_dict()
            if not raw_logits.bitwise_equal:
                _add_reason(reasons, "boundary_bitwise_mismatch")
        if (
            index < len(reference.processed_outer_logits)
            and index < len(candidate.processed_outer_logits)
        ):
            processed = compare_tensors(
                f"boundary.{index}.processed_outer_logits",
                reference.processed_outer_logits[index],
                candidate.processed_outer_logits[index],
            )
            report["processed_outer_logits"] = processed.to_dict()
            if not processed.bitwise_equal:
                _add_reason(reasons, "boundary_bitwise_mismatch")
        boundary_reports.append(report)

    raw_codec = compare_tensors(
        "raw_codec", reference.codec.raw_codec, candidate.codec.raw_codec
    )
    first_codebook = compare_tensors(
        "first_codebook_ids",
        reference.codec.first_codebook_ids,
        candidate.codec.first_codebook_ids,
    )
    predictor = compare_tensors(
        "predictor_sequences",
        reference.codec.predictor_sequences,
        candidate.codec.predictor_sequences,
    )
    sampled_first_codebook = compare_tensors(
        "sampled_first_codebook",
        reference.sampled_first_codebook,
        candidate.sampled_first_codebook,
    )
    if not raw_codec.bitwise_equal:
        _add_reason(reasons, "raw_codec_mismatch")
    if not first_codebook.bitwise_equal:
        _add_reason(reasons, "first_codebook_mismatch")
    if not predictor.bitwise_equal:
        _add_reason(reasons, "predictor_sequence_mismatch")
    if not sampled_first_codebook.bitwise_equal:
        _add_reason(reasons, "sampled_first_codebook_mismatch")
    if (
        reference.codec.raw_sha256 != candidate.codec.raw_sha256
        or reference.codec.combined_sha256 != candidate.codec.combined_sha256
        or reference.codec.item_sha256 != candidate.codec.item_sha256
    ):
        _add_reason(reasons, "codec_hash_mismatch")
    if (
        reference.codec.item_shapes != candidate.codec.item_shapes
        or reference.codec.item_dtypes != candidate.codec.item_dtypes
    ):
        _add_reason(reasons, "codec_metadata_mismatch")
    if reference.codec.eos_positions != candidate.codec.eos_positions:
        _add_reason(reasons, "eos_position_mismatch")

    for mode_capture in (reference, candidate):
        expected_predictor = mode_capture.codec.raw_codec.tensor[:, :, 1:]
        if not compare_tensors(
            f"{mode_capture.mode}.predictor_raw_consistency",
            expected_predictor,
            mode_capture.codec.predictor_sequences,
        ).bitwise_equal:
            _add_reason(reasons, "predictor_raw_codec_inconsistent")
        if not compare_tensors(
            f"{mode_capture.mode}.sampled_first_codebook_raw_consistency",
            mode_capture.codec.first_codebook_ids,
            mode_capture.sampled_first_codebook,
        ).bitwise_equal:
            _add_reason(reasons, "sampled_first_codebook_raw_inconsistent")

    for mode_capture in (reference, candidate):
        audio_lengths = (
            len(mode_capture.audio.sample_counts),
            len(mode_capture.audio.finite),
            len(mode_capture.audio.nonempty),
        )
        if any(length != mode_capture.batch_size for length in audio_lengths):
            _add_reason(reasons, "audio_batch_size_mismatch")
    if reference.audio.sample_counts != candidate.audio.sample_counts:
        _add_reason(reasons, "audio_sample_count_mismatch")
    if reference.audio.sample_rate != candidate.audio.sample_rate:
        _add_reason(reasons, "audio_sample_rate_mismatch")
    if not all(reference.audio.finite) or not all(candidate.audio.finite):
        _add_reason(reasons, "audio_not_finite")
    if not all(reference.audio.nonempty) or not all(candidate.audio.nonempty):
        _add_reason(reasons, "audio_empty")
    if reference.warnings or candidate.warnings:
        _add_reason(reasons, "warnings_present")
    if reference.errors or candidate.errors:
        _add_reason(reasons, "errors_present")

    codec_report = {
        "raw_codec": raw_codec.to_dict(),
        "first_codebook_ids": first_codebook.to_dict(),
        "predictor_sequences": predictor.to_dict(),
        "sampled_first_codebook": sampled_first_codebook.to_dict(),
        "hashes_equal": (
            reference.codec.raw_sha256 == candidate.codec.raw_sha256
            and reference.codec.combined_sha256 == candidate.codec.combined_sha256
            and reference.codec.item_sha256 == candidate.codec.item_sha256
        ),
        "shapes_equal": reference.codec.item_shapes == candidate.codec.item_shapes,
        "dtypes_equal": reference.codec.item_dtypes == candidate.codec.item_dtypes,
        "eos_positions_equal": (
            reference.codec.eos_positions == candidate.codec.eos_positions
        ),
        "audio_sample_counts_equal": (
            reference.audio.sample_counts == candidate.audio.sample_counts
        ),
    }
    return ParitySummary(
        parity_passed=not reasons,
        failed_reasons=tuple(reasons),
        boundary_comparisons=tuple(boundary_reports),
        codec_comparisons=codec_report,
        first_codec_divergence=first_codec_divergence(
            reference.codec.raw_codec, candidate.codec.raw_codec
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _AuditableArgumentParser(
        description="Compare upstream DynamicCache and active-prefix Qwen-TTS exactly."
    )
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="english")
    parser.add_argument("--speaker", default="")
    parser.add_argument("--texts", nargs="+", default=["I'm here to help."])
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--requests", type=int, default=0)
    parser.add_argument("--duration-minutes", type=float, default=0.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/qwen_tts_active_prefix_parity.json"),
    )
    return parser


def main(
    argv: list[str] | None = None,
    dependencies: ProbeDependencies | None = None,
) -> int:
    dependencies = dependencies or ProbeDependencies()
    parse_args = dependencies.parse_args or build_parser().parse_args
    try:
        args = parse_args(argv)
    except SystemExit as caught:
        if caught.code in (None, 0):
            raise
        parse_error = ArgumentParseFailure(
            f"argument parser exited with status {caught.code}"
        )
        return _write_argument_failure(
            argv, parse_error, out=_report_path_from_argv(argv)
        )
    except ArgumentParseFailure as parse_error:
        return _write_argument_failure(
            argv, parse_error, out=_report_path_from_argv(argv)
        )

    if _stability_enabled(args):
        return _main_stability(args, dependencies)

    execute = dependencies.execute or _execute_probe
    environment_fn = dependencies.environment or _collect_environment
    configured_batch_sizes: list[int] = []
    batch_reports: dict[int, dict[str, Any]] = {}
    environment: dict[str, Any] = {}
    error: Exception | None = None
    error_traceback = None

    try:
        _validate_args(args)
        configured_batch_sizes = _parse_batch_sizes(args.batch_sizes)
        environment = environment_fn(args)
        for batch_size, reference, candidate in execute(args):
            batch_size = int(batch_size)
            if batch_size not in configured_batch_sizes:
                raise ParityProbeError(f"unexpected result batch size {batch_size}")
            if batch_size in batch_reports:
                raise ParityProbeError(
                    f"duplicate result for configured batch size {batch_size}"
                )
            for capture in (reference, candidate):
                if capture.batch_size != batch_size:
                    raise ParityProbeError(
                        f"{capture.mode} capture batch size {capture.batch_size} "
                        f"does not match configured batch size {batch_size}"
                    )
            summary = summarize_parity(reference, candidate)
            batch_reports[batch_size] = {
                "batch_size": batch_size,
                "upstream": reference.metadata(),
                "active_prefix": candidate.metadata(),
                "summary": summary.to_dict(),
            }
        missing = [
            size for size in configured_batch_sizes if size not in batch_reports
        ]
        if missing:
            missing_text = ",".join(str(size) for size in missing)
            raise ParityProbeError(
                f"missing result for configured batch size {missing_text}"
            )
    except Exception as caught:
        error = caught
        error_traceback = caught.__traceback__

    batches = [
        batch_reports[size]
        for size in configured_batch_sizes
        if size in batch_reports
    ]

    parity_passed = bool(batches) and all(
        bool(batch["summary"]["parity_passed"]) for batch in batches
    )
    exception_lines = _exception_report_lines(error)
    error_warnings = _exception_warning_notes(error)
    report = _plain_copy({
        "schema": "qwen_tts_active_prefix_parity",
        "schema_version": 1,
        "status": "failure" if error is not None else "completed",
        "failure_kind": "runtime" if error is not None else None,
        "parity_passed": parity_passed if error is None else False,
        "config": _config_report(args),
        "environment": _plain_copy(environment),
        "batches": batches,
        "warnings": [
            *[
                warning
                for batch in batches
                for mode in ("upstream", "active_prefix")
                for warning in batch[mode]["warnings"]
            ],
            *error_warnings,
        ],
        "errors": exception_lines,
        "exit_contract": _exit_contract(),
    })
    try:
        _write_report(args.out, report)
    except Exception as write_error:
        if error is None:
            raise
        error.add_note(f"report write also failed: {write_error}")
        raise error.with_traceback(error_traceback)
    if error is not None:
        raise error.with_traceback(error_traceback)
    return 0 if parity_passed else 2


def _main_stability(
    args: argparse.Namespace,
    dependencies: ProbeDependencies,
) -> int:
    execute = dependencies.stability_execute or _execute_stability
    environment_fn = dependencies.environment or _collect_environment
    environment: dict[str, Any] = {}
    execution: StabilityExecution | None = None
    error: BaseException | None = None
    error_traceback = None

    try:
        _validate_args(args)
        environment = environment_fn(args)
        execution = execute(args)
        if execution.primary_error is not None:
            error = execution.primary_error
            error_traceback = execution.primary_traceback
    except BaseException as caught:
        error = caught
        error_traceback = caught.__traceback__

    request_results = list(execution.request_results) if execution else []
    execution_warnings = execution.warnings if execution else ()
    execution_errors = execution.errors if execution else ()
    request_warnings = tuple(
        str(warning)
        for request in request_results
        for warning in request.get("warnings", ())
    )
    request_errors = tuple(
        str(request_error)
        for request in request_results
        for request_error in request.get("errors", ())
    )
    warnings_report = _merge_warning_messages(
        execution_warnings,
        request_warnings,
        _exception_warning_notes(error),
    )
    errors_report = _merge_warning_messages(
        execution_errors,
        request_errors,
        _exception_report_lines(error),
    )
    quality = _stability_quality_report(
        args,
        request_results,
        elapsed_seconds=(execution.elapsed_seconds if execution is not None else 0.0),
    )
    parity_passed = bool(request_results) and all(
        bool(request.get("parity_passed")) for request in request_results
    )
    parity_passed = parity_passed and quality["passed"] and not warnings_report and not errors_report
    runtime_failed = error is not None or bool(errors_report)
    report = _plain_copy(
        {
            "schema": "qwen_tts_active_prefix_parity",
            "schema_version": 1,
            "mode": "stability",
            "status": "failure" if runtime_failed else "completed",
            "failure_kind": "runtime" if runtime_failed else None,
            "parity_passed": parity_passed,
            "config": _config_report(args),
            "environment": _plain_copy(environment),
            "batches": [],
            "stability": {
                "target_requests": int(args.requests),
                "target_duration_seconds": float(
                    math.ceil(float(args.duration_minutes) * 60.0)
                ),
                "completed_requests": len(request_results),
                "elapsed_seconds": (
                    execution.elapsed_seconds if execution is not None else 0.0
                ),
                "requests": request_results,
                "quality": quality,
            },
            "warnings": warnings_report,
            "errors": errors_report,
            "exit_contract": _exit_contract(),
        }
    )
    try:
        _write_report(args.out, report)
    except BaseException as write_error:
        if error is None:
            raise
        error.add_note(f"report write also failed: {write_error}")
        raise error.with_traceback(error_traceback)
    if error is not None:
        raise error.with_traceback(error_traceback)
    if runtime_failed:
        return 1
    return 0 if parity_passed else 2


def _write_argument_failure(
    argv: list[str] | None,
    error: ArgumentParseFailure,
    *,
    out: Path,
) -> int:
    args = build_parser().parse_args([])
    args.out = out
    report = _plain_copy(
        {
            "schema": "qwen_tts_active_prefix_parity",
            "schema_version": 1,
            "status": "failure",
            "failure_kind": "argument_parse",
            "parity_passed": False,
            "config": {
                **_config_report(args),
                "argv": list(sys.argv[1:] if argv is None else argv),
            },
            "environment": {},
            "batches": [],
            "warnings": [],
            "errors": [f"{type(error).__name__}: {error}"],
            "exit_contract": _exit_contract(),
        }
    )
    try:
        _write_report(out, report)
    except Exception as write_error:
        error.add_note(f"report write also failed: {write_error}")
        raise error
    return 64


def _report_path_from_argv(argv: list[str] | None) -> Path:
    values = list(sys.argv[1:] if argv is None else argv)
    out = Path("results/qwen_tts_active_prefix_parity.json")
    for index, value in enumerate(values):
        if value.startswith("--out="):
            candidate = value.partition("=")[2]
            if candidate:
                out = Path(candidate)
        elif value == "--out" and index + 1 < len(values):
            candidate = values[index + 1]
            if candidate and not candidate.startswith("-"):
                out = Path(candidate)
    return out


def _exception_report_lines(error: Exception | None) -> list[str]:
    if error is None:
        return []
    return [
        f"{type(error).__name__}: {error}",
        *[str(note) for note in getattr(error, "__notes__", ())],
    ]


def _exception_warning_notes(error: Exception | None) -> list[str]:
    if error is None:
        return []
    prefix = "probe warning: "
    return [
        str(note)[len(prefix) :]
        for note in getattr(error, "__notes__", ())
        if str(note).startswith(prefix)
    ]


def _attach_warning_notes(error: BaseException, messages: Sequence[str]) -> None:
    existing = set(str(note) for note in getattr(error, "__notes__", ()))
    for message in messages:
        note = f"probe warning: {message}"
        if note not in existing:
            error.add_note(note)
            existing.add(note)


def _exit_contract() -> dict[str, Any]:
    return {
        "pass": 0,
        "completed_parity_mismatch": 2,
        "runtime_failure": 1,
        "runtime_failure_behavior": (
            "failure JSON is written, then the exception is re-raised"
        ),
        "argument_parse_failure": 64,
    }


def _execute_stability(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[..., Any] | None = None,
    active_installer: Callable[..., bool] | None = None,
    prepare_fn: Callable[..., dict[str, Any]] | None = None,
    run_mode_fn: Callable[..., ModeCapture] | None = None,
    memory_snapshot_fn: Callable[[str], dict[str, int]] | None = None,
    clock: Callable[[], float] | None = None,
) -> StabilityExecution:
    if backend_factory is None:
        from qwen_asr_vllm.agent.local_tts import QwenTtsBackend

        backend_factory = QwenTtsBackend
    if active_installer is None:
        from qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache import (
            install_active_prefix_outer_talker,
        )

        active_installer = install_active_prefix_outer_talker
    prepare_fn = prepare_fn or _prepare_talker_inputs_once
    run_mode_fn = run_mode_fn or _run_mode
    memory_snapshot_fn = memory_snapshot_fn or _memory_snapshot
    clock = clock or time.monotonic

    request_target = int(args.requests)
    duration_seconds = float(math.ceil(float(args.duration_minutes) * 60.0))
    started = None
    elapsed = 0.0
    text_cursor = 0
    request_results: list[dict[str, Any]] = []
    request_warning_messages: set[str] = set()
    global_failure = _FailureAccumulator()
    backend = None
    active_generate = None
    active_engine = None
    active_runtime = None

    with _capture_probe_warnings() as lifetime_warning_capture:
        try:
            backend = backend_factory(
                model_path=args.model_path,
                device=args.device,
                language=args.language,
                speaker=args.speaker,
                warmup=False,
            )
            wrapper = backend._model
            talker = wrapper.model.talker
            original_generate_state = _capture_attribute_state(talker, "generate")
            original_generate = _required_callable(talker, "generate")
            original_rope_state = _capture_attribute_state(talker, "rope_deltas")
            eos_token_id = int(
                wrapper.model.config.talker_config.codec_eos_token_id
            )
            # Model loading is setup, not steady-state serving time.
            started = float(clock())

            while _stability_target_pending(
                len(request_results),
                elapsed,
                request_target=request_target,
                duration_seconds=duration_seconds,
            ):
                request_index = len(request_results)
                batch_size = 1 if request_index % 2 == 0 else 2
                texts, text_indices, text_cursor = _stability_request_texts(
                    args.texts,
                    cursor=text_cursor,
                    batch_size=batch_size,
                )
                request_failure = _FailureAccumulator()
                control_error: BaseException | None = None
                reference = None
                candidate = None
                summary = None
                active_cache = {"metrics": {}, "backing_pointers": []}
                memory = {
                    "memory_allocated": 0,
                    "memory_reserved": 0,
                    "max_memory_allocated": 0,
                    "max_memory_reserved": 0,
                }

                with _capture_probe_warnings() as request_warning_capture:
                    prepared = None
                    try:
                        prepared = prepare_fn(wrapper, texts, backend, args)
                        if batch_size == 2:
                            prepared = _left_pad_prepared_inputs(prepared)
                        resolved_config = _resolved_generation_config(
                            wrapper,
                            backend,
                            prepared,
                            eos_token_id=eos_token_id,
                        )
                    except BaseException as error:
                        if _is_control_exception(error):
                            control_error = error
                        else:
                            request_failure.record(error)

                    if (
                        prepared is not None
                        and request_failure.error is None
                        and control_error is None
                    ):
                        try:
                            reference = run_mode_fn(
                                wrapper,
                                prepared=clone_prepared_inputs(prepared),
                                mode="upstream",
                                seed=args.seed,
                                eos_token_id=eos_token_id,
                                device=args.device,
                                resolved_config=resolved_config,
                            )
                        except BaseException as error:
                            if _is_control_exception(error):
                                control_error = error
                            else:
                                request_failure.record(error)
                        _record_restoration(
                            request_failure,
                            talker,
                            original_generate_state,
                            original_generate,
                            original_rope_state,
                        )

                    if (
                        prepared is not None
                        and request_failure.error is None
                        and control_error is None
                    ):
                        try:
                            if active_generate is None:
                                installed = active_installer(
                                    talker, max_cache_len=args.max_cache_len
                                )
                                if not installed:
                                    raise ParityProbeError(
                                        "active-prefix installer did not install"
                                    )
                                active_generate = _required_callable(
                                    talker, "generate"
                                )
                                active_engine = getattr(
                                    active_generate, "_qav_outer_engine", None
                                )
                                active_runtime = getattr(
                                    active_engine, "step_runtime", None
                                )
                                if active_runtime is None:
                                    raise ParityProbeError(
                                        "active-prefix runtime metadata is missing"
                                    )
                            else:
                                setattr(talker, "generate", active_generate)
                            candidate = run_mode_fn(
                                wrapper,
                                prepared=clone_prepared_inputs(prepared),
                                mode="active_prefix",
                                seed=args.seed,
                                eos_token_id=eos_token_id,
                                device=args.device,
                                resolved_config=resolved_config,
                            )
                        except BaseException as error:
                            if _is_control_exception(error):
                                control_error = error
                            else:
                                request_failure.record(error)
                        _record_restoration(
                            request_failure,
                            talker,
                            original_generate_state,
                            original_generate,
                            original_rope_state,
                        )

                    try:
                        memory = _plain_copy(memory_snapshot_fn(args.device))
                    except BaseException as error:
                        if _is_control_exception(error):
                            control_error = error
                        else:
                            request_failure.record(error, label="memory snapshot")
                    if active_engine is not None:
                        try:
                            active_cache = _active_cache_snapshot(active_engine)
                        except BaseException as error:
                            if _is_control_exception(error):
                                control_error = error
                            else:
                                request_failure.record(
                                    error, label="active cache snapshot"
                                )

                request_warnings = tuple(request_warning_capture.messages())
                if control_error is not None:
                    raise control_error
                if request_failure.error is not None:
                    _attach_warning_notes(
                        request_failure.error, request_warnings
                    )
                if reference is not None:
                    reference = replace(
                        reference,
                        warnings=_merge_warning_messages(
                            reference.warnings, request_warnings
                        ),
                    )
                if candidate is not None:
                    candidate = replace(
                        candidate,
                        warnings=_merge_warning_messages(
                            candidate.warnings, request_warnings
                        ),
                    )
                if reference is not None and candidate is not None:
                    try:
                        summary = summarize_parity(reference, candidate)
                    except BaseException as error:
                        request_failure.record(error, label="parity summary")

                all_request_warnings = _merge_warning_messages(
                    request_warnings,
                    reference.warnings if reference is not None else (),
                    candidate.warnings if candidate is not None else (),
                    _exception_warning_notes(request_failure.error),
                )
                request_warning_messages.update(all_request_warnings)
                request_errors = _exception_report_lines(request_failure.error)
                parity_passed = bool(
                    summary is not None
                    and summary.parity_passed
                    and not all_request_warnings
                    and not request_errors
                )
                request_results.append(
                    _plain_copy(
                        {
                            "request_index": request_index,
                            "batch_size": batch_size,
                            "text_indices": text_indices,
                            "left_padded": batch_size == 2,
                            "left_padding_evidence": (
                                _left_padding_evidence(prepared)
                                if prepared is not None and batch_size == 2
                                else {}
                            ),
                            "parity_passed": parity_passed,
                            "parity_summary": (
                                summary.to_dict() if summary is not None else None
                            ),
                            "exact_hashes": _exact_hashes(reference, candidate),
                            "timings_ms": {
                                "upstream": (
                                    reference.timings_ms
                                    if reference is not None
                                    else None
                                ),
                                "active_prefix": (
                                    candidate.timings_ms
                                    if candidate is not None
                                    else None
                                ),
                            },
                            "active_cache": active_cache,
                            "memory": memory,
                            "warnings": all_request_warnings,
                            "errors": request_errors,
                        }
                    )
                )
                if control_error is not None:
                    raise control_error
                elapsed = max(0.0, float(clock()) - float(started))
        except BaseException as error:
            global_failure.record(error)
        finally:
            if active_runtime is not None:
                global_failure.run("active runtime close", active_runtime.close)
            if backend is not None:
                global_failure.run("backend close", backend.close)

    lifetime_warnings = tuple(lifetime_warning_capture.messages())
    global_warnings = tuple(
        message
        for message in lifetime_warnings
        if message not in request_warning_messages
    )
    if global_failure.error is not None:
        _attach_warning_notes(global_failure.error, global_warnings)
    return StabilityExecution(
        request_results=tuple(request_results),
        elapsed_seconds=elapsed,
        warnings=global_warnings,
        errors=tuple(_exception_report_lines(global_failure.error)),
        primary_error=global_failure.error,
        primary_traceback=global_failure.traceback,
    )


def _is_control_exception(error: BaseException) -> bool:
    return isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit))


def _stability_quality_report(
    args: argparse.Namespace,
    request_results: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    request_target = int(args.requests)
    raw_duration = float(args.duration_minutes)
    duration_target = (
        float(math.ceil(raw_duration * 60.0))
        if math.isfinite(raw_duration)
        else 0.0
    )
    completed = len(request_results)
    target_satisfied = (
        (request_target <= 0 or completed >= request_target)
        and (duration_target <= 0 or elapsed_seconds >= duration_target)
    )
    failures: list[str] = []
    if not target_satisfied:
        failures.append("stability_targets_not_satisfied")
    acceptance_ready = request_target >= 100 and duration_target >= 600
    if not acceptance_ready:
        failures.append("stability_acceptance_requires_100_requests_and_10_minutes")
    if not request_results:
        failures.append("no_requests_completed")
    expected_batches = {1, 2}
    observed_batches = {int(item.get("batch_size", -1)) for item in request_results}
    if not expected_batches.issubset(observed_batches):
        failures.append("missing_batch_size_1_or_2")

    left_padding_evidence = [
        item.get("left_padding_evidence", {})
        for item in request_results
        if int(item.get("batch_size", -1)) == 2
    ]
    if not any(
        any(int(value) > 0 for value in evidence.get("left_padding_tokens", ()))
        for evidence in left_padding_evidence
    ):
        failures.append("batch_2_has_no_observed_left_padding")

    for item in request_results:
        if item.get("warnings") or item.get("errors"):
            failures.append(f"request_{item.get('request_index')}_reported_failure")
        metrics = item.get("active_cache", {}).get("metrics", {})
        required_metrics = {"errors", "cache_overflows"}
        if not required_metrics.issubset(metrics):
            failures.append(f"request_{item.get('request_index')}_missing_cache_metrics")
        if int(metrics.get("errors", 0)) or int(metrics.get("cache_overflows", 0)):
            failures.append(f"request_{item.get('request_index')}_cache_error")

    pointer_by_slot: dict[tuple[Any, Any], tuple[Any, Any]] = {}
    pointer_stability = True
    for item in request_results:
        pointers = item.get("active_cache", {}).get("backing_pointers", ())
        if not pointers:
            pointer_stability = False
            continue
        # A new batch size may allocate its pool once. Existing layer backing
        # addresses must never change after they have been observed.
        for pointer in pointers:
            slot = (pointer.get("cache_identity"), pointer.get("layer_index"))
            address = (pointer.get("key"), pointer.get("value"))
            previous = pointer_by_slot.get(slot)
            if previous is not None and previous != address:
                pointer_stability = False
            pointer_by_slot.setdefault(slot, address)
    if not pointer_stability:
        failures.append("active_cache_backing_pointers_changed")

    memory = [item.get("memory", {}) for item in request_results]
    memory_delta = 0
    reserved_delta = 0
    memory_stable = True
    if memory:
        initial = memory[0]
        final = memory[-1]
        memory_delta = int(final.get("memory_allocated", 0)) - int(initial.get("memory_allocated", 0))
        reserved_delta = int(final.get("memory_reserved", 0)) - int(initial.get("memory_reserved", 0))
        baseline = int(initial.get("memory_reserved", 0))
        allocated_limit = max(512 * 1024 * 1024, int(max(initial.get("memory_allocated", 0), 1) * 0.10))
        reserved_limit = max(512 * 1024 * 1024, int(baseline * 0.10))
        memory_stable = (
            memory_delta <= allocated_limit
            and reserved_delta <= reserved_limit
            and all(
                int(item.get("memory_allocated", 0)) - int(initial.get("memory_allocated", 0))
                <= allocated_limit
                for item in memory
            )
        )
    else:
        memory_stable = False
        failures.append("missing_cuda_memory_snapshots")
    if not memory_stable:
        failures.append("cuda_reserved_memory_growth_exceeded_10pct_or_512MiB")
    return {
        "passed": not failures,
        "failures": list(dict.fromkeys(failures)),
        "targets_satisfied": target_satisfied,
        "acceptance_ready": acceptance_ready,
        "observed_batches": sorted(observed_batches),
        "left_padding_observed": bool(
            any(
                any(int(value) > 0 for value in evidence.get("left_padding_tokens", ()))
                for evidence in left_padding_evidence
            )
        ),
        "cache_backing_pointers_stable_by_batch": pointer_stability,
        "reserved_memory_delta_bytes": reserved_delta,
        "allocated_memory_delta_bytes": memory_delta,
        "reserved_memory_growth_gate": memory_stable,
    }


def _record_restoration(
    failure: _FailureAccumulator,
    talker: Any,
    generate_state: _AttributeState,
    expected_generate: Callable[..., Any],
    rope_state: _AttributeState,
) -> None:
    failure.run(
        "talker.generate restoration",
        lambda: _restore_attribute_state(talker, "generate", generate_state),
    )
    failure.run(
        "talker.rope_deltas restoration",
        lambda: _restore_attribute_state(talker, "rope_deltas", rope_state),
    )
    failure.run(
        "talker.generate restoration validation",
        lambda: _validate_restored_attribute(
            talker,
            "generate",
            generate_state,
            expected_callable=expected_generate,
        ),
    )
    failure.run(
        "talker.rope_deltas restoration validation",
        lambda: _validate_restored_attribute(talker, "rope_deltas", rope_state),
    )


def _execute_probe(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[..., Any] | None = None,
    active_installer: Callable[..., bool] | None = None,
    prepare_fn: Callable[..., dict[str, Any]] | None = None,
    run_mode_fn: Callable[..., ModeCapture] | None = None,
) -> Sequence[tuple[int, ModeCapture, ModeCapture]]:
    if backend_factory is None:
        from qwen_asr_vllm.agent.local_tts import QwenTtsBackend

        backend_factory = QwenTtsBackend
    if active_installer is None:
        from qwen_asr_vllm.agent.qwen_tts_outer_active_prefix_cache import (
            install_active_prefix_outer_talker,
        )

        active_installer = install_active_prefix_outer_talker
    prepare_fn = prepare_fn or _prepare_talker_inputs_once
    run_mode_fn = run_mode_fn or _run_mode

    failure = _FailureAccumulator()
    backend = None
    results = []
    with _capture_probe_warnings() as lifetime_warning_capture:
        try:
            backend = backend_factory(
                model_path=args.model_path,
                device=args.device,
                language=args.language,
                speaker=args.speaker,
                warmup=False,
            )
            wrapper = backend._model
            talker = wrapper.model.talker
            for batch_size in _parse_batch_sizes(args.batch_sizes):
                texts = _texts_for_batch(args.texts, batch_size)
                prepared = prepare_fn(wrapper, texts, backend, args)
                eos_token_id = int(
                    wrapper.model.config.talker_config.codec_eos_token_id
                )
                resolved_config = _resolved_generation_config(
                    wrapper,
                    backend,
                    prepared,
                    eos_token_id=eos_token_id,
                )
                original_generate_state = _capture_attribute_state(talker, "generate")
                original_generate = _required_callable(talker, "generate")
                original_rope_state = _capture_attribute_state(talker, "rope_deltas")

                try:
                    reference = run_mode_fn(
                        wrapper,
                        prepared=clone_prepared_inputs(prepared),
                        mode="upstream",
                        seed=args.seed,
                        eos_token_id=eos_token_id,
                        device=args.device,
                        resolved_config=resolved_config,
                    )
                except BaseException as error:
                    failure.record(error)
                failure.run(
                    "talker.generate restoration",
                    lambda: _restore_attribute_state(
                        talker, "generate", original_generate_state
                    ),
                )
                failure.run(
                    "talker.rope_deltas restoration",
                    lambda: _restore_attribute_state(
                        talker, "rope_deltas", original_rope_state
                    ),
                )
                failure.run(
                    "talker.generate restoration validation",
                    lambda: _validate_restored_attribute(
                        talker,
                        "generate",
                        original_generate_state,
                        expected_callable=original_generate,
                    ),
                )
                failure.run(
                    "talker.rope_deltas restoration validation",
                    lambda: _validate_restored_attribute(
                        talker, "rope_deltas", original_rope_state
                    ),
                )
                if failure.error is not None:
                    break

                active_runtime = None
                try:
                    installed = active_installer(
                        talker, max_cache_len=args.max_cache_len
                    )
                    if not installed:
                        raise ParityProbeError("active-prefix installer did not install")
                    active_generate = _required_callable(talker, "generate")
                    engine = getattr(active_generate, "_qav_outer_engine", None)
                    active_runtime = getattr(engine, "step_runtime", None)
                    if active_runtime is None:
                        raise ParityProbeError("active-prefix runtime metadata is missing")
                    candidate = run_mode_fn(
                        wrapper,
                        prepared=clone_prepared_inputs(prepared),
                        mode="active_prefix",
                        seed=args.seed,
                        eos_token_id=eos_token_id,
                        device=args.device,
                        resolved_config=resolved_config,
                    )
                except BaseException as error:
                    failure.record(error)
                if active_runtime is not None:
                    failure.run("active runtime close", active_runtime.close)
                failure.run(
                    "talker.generate restoration",
                    lambda: _restore_attribute_state(
                        talker, "generate", original_generate_state
                    ),
                )
                failure.run(
                    "talker.rope_deltas restoration",
                    lambda: _restore_attribute_state(
                        talker, "rope_deltas", original_rope_state
                    ),
                )
                failure.run(
                    "talker.generate restoration validation",
                    lambda: _validate_restored_attribute(
                        talker,
                        "generate",
                        original_generate_state,
                        expected_callable=original_generate,
                    ),
                )
                failure.run(
                    "talker.rope_deltas restoration validation",
                    lambda: _validate_restored_attribute(
                        talker, "rope_deltas", original_rope_state
                    ),
                )
                if failure.error is not None:
                    break

                results.append((batch_size, reference, candidate))
        except BaseException as error:
            failure.record(error)
        finally:
            if backend is not None:
                failure.run("backend close", backend.close)

    lifetime_warnings = tuple(lifetime_warning_capture.messages())
    if failure.error is not None:
        _attach_warning_notes(failure.error, lifetime_warnings)
        failure.raise_if_present()

    return [
        (
            batch_size,
            replace(
                reference,
                warnings=_merge_warning_messages(
                    reference.warnings, lifetime_warnings
                ),
            ),
            replace(
                candidate,
                warnings=_merge_warning_messages(
                    candidate.warnings, lifetime_warnings
                ),
            ),
        )
        for batch_size, reference, candidate in results
    ]


def _prepare_talker_inputs_once(
    wrapper: Any,
    texts: Sequence[str],
    backend: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    required = (
        "_build_assistant_text",
        "_tokenize_texts",
        "_validate_languages",
        "_validate_speakers",
        "_merge_generate_kwargs",
    )
    for name in required:
        _required_callable(wrapper, name)
    model = _required_member(wrapper, "model")
    talker = _required_member(model, "talker")
    languages = [backend._language] * len(texts)
    speakers = [backend._speaker] * len(texts)
    wrapper._validate_languages(languages)
    wrapper._validate_speakers(speakers)
    input_ids = wrapper._tokenize_texts(
        [wrapper._build_assistant_text(text) for text in texts]
    )
    instruct_ids = [None] * len(texts)
    generation_kwargs = wrapper._merge_generate_kwargs(
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=args.max_new_tokens,
    )
    captured: dict[str, Any] = {}
    generate_state = _capture_attribute_state(talker, "generate")
    original_rope_state = _capture_attribute_state(talker, "rope_deltas")

    def intercept_generate(**kwargs):
        if captured:
            raise ParityProbeError("talker.generate was called more than once during preparation")
        captured.update(kwargs)
        raise _PreparedInputsCaptured

    failure = _FailureAccumulator()
    setattr(talker, "generate", intercept_generate)
    try:
        try:
            model.generate(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                languages=languages,
                speakers=speakers,
                non_streaming_mode=True,
                **generation_kwargs,
            )
        except _PreparedInputsCaptured:
            pass
        except BaseException as error:
            failure.record(error)
    finally:
        failure.run(
            "talker.generate restoration",
            lambda: _restore_attribute_state(talker, "generate", generate_state),
        )
        failure.run(
            "talker.rope_deltas restoration",
            lambda: _restore_attribute_state(
                talker, "rope_deltas", original_rope_state
            ),
        )
    failure.raise_if_present()
    if not captured:
        raise ParityProbeError("installed model did not call talker.generate during preparation")
    if "eos_token_id" not in captured:
        raise ParityProbeError("prepared talker inputs are missing required field: eos_token_id")
    captured["temperature"] = None
    captured["subtalker_temperature"] = None
    captured["pad_token_id"] = int(captured["eos_token_id"])
    required_prepared = {
        "inputs_embeds",
        "attention_mask",
        "do_sample",
        "eos_token_id",
        "min_new_tokens",
        "pad_token_id",
        "repetition_penalty",
        "subtalker_dosample",
        "subtalker_temperature",
        "suppress_tokens",
        "trailing_text_hidden",
        "tts_pad_embed",
    }
    missing = sorted(required_prepared.difference(captured))
    if missing:
        raise ParityProbeError(
            f"prepared talker inputs are missing required fields: {', '.join(missing)}"
        )
    return captured


def _resolved_generation_config(
    wrapper: Any,
    backend: Any,
    prepared: Mapping[str, Any],
    *,
    eos_token_id: int,
) -> dict[str, Any]:
    suppress_tokens = [int(token) for token in prepared.get("suppress_tokens", ())]
    suppress_payload = ",".join(str(token) for token in suppress_tokens).encode("ascii")
    generation_keys = (
        "max_new_tokens",
        "min_new_tokens",
        "do_sample",
        "top_k",
        "top_p",
        "temperature",
        "subtalker_dosample",
        "subtalker_top_k",
        "subtalker_top_p",
        "subtalker_temperature",
        "eos_token_id",
        "pad_token_id",
        "repetition_penalty",
    )
    return {
        "speaker": getattr(backend, "_speaker", None),
        "language": getattr(backend, "_language", None),
        "generation_defaults": _plain_copy(
            getattr(wrapper, "generate_defaults", {})
        ),
        "generation": {
            key: _plain_copy(prepared.get(key)) for key in generation_keys
        },
        "resolved_eos_token_id": int(eos_token_id),
        "suppress_tokens": suppress_tokens,
        "suppress_token_count": len(suppress_tokens),
        "suppress_tokens_sha256": hashlib.sha256(suppress_payload).hexdigest(),
    }


def _run_mode(
    wrapper: Any,
    *,
    prepared: dict[str, Any],
    mode: str,
    seed: int,
    eos_token_id: int,
    device: str,
    resolved_config: Mapping[str, Any] | None = None,
) -> ModeCapture:
    talker = wrapper.model.talker
    recorder = HookRecorder(attention_evidence=_attention_evidence(talker))
    _set_seed(seed)
    failure = _FailureAccumulator()
    with _capture_probe_warnings() as warning_capture:
        try:
            generation_started = time.perf_counter()
            _synchronize(device)
            with _capture_talker_hooks(talker, mode=mode, recorder=recorder):
                generation_result = talker.generate(**prepared)
            _synchronize(device)
            generation_ms = (time.perf_counter() - generation_started) * 1000.0

            raw_codec = _raw_codec_from_generation(generation_result)
            decoded_items = _decoded_codec_items(raw_codec, eos_token_id)
            predictor_sequences = _stack_predictor_sequences(recorder, raw_codec)
            sampled_first_codebook = _stack_sampled_first_codebook(
                recorder, raw_codec
            )
            codec = build_codec_capture(
                raw_codec,
                predictor_sequences=predictor_sequences,
                decoded_codec_items=decoded_items,
                eos_token_id=eos_token_id,
            )
            decode_started = time.perf_counter()
            _synchronize(device)
            audios, sample_rate = wrapper.model.speech_tokenizer.decode(
                [{"audio_codes": item} for item in decoded_items]
            )
            _synchronize(device)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            runtime_metrics = {}
            generate = _required_callable(talker, "generate")
            engine = getattr(generate, "_qav_outer_engine", None)
            snapshot = getattr(engine, "metrics_snapshot", None)
            if callable(snapshot):
                runtime_metrics = _plain_copy(snapshot())
        except BaseException as error:
            failure.record(error)

    mode_warnings = warning_capture.messages()
    if failure.error is not None:
        _attach_warning_notes(failure.error, mode_warnings)
        failure.raise_if_present()

    return ModeCapture(
        mode=mode,
        batch_size=int(raw_codec.shape[0]),
        boundaries=tuple(recorder.boundaries),
        raw_codec_logits=tuple(recorder.raw_codec_logits),
        processed_outer_logits=tuple(recorder.processed_outer_logits),
        sampled_first_codebook=capture_tensor(
            "sampled_first_codebook", sampled_first_codebook
        ),
        codec=codec,
        audio=build_audio_capture(audios, sample_rate=int(sample_rate)),
        attention_evidence=_plain_copy(recorder.attention_evidence),
        runtime_metrics=runtime_metrics,
        timings_ms={
            "generation": round(generation_ms, 3),
            "decode": round(decode_ms, 3),
        },
        resolved_config=_plain_copy(resolved_config or {}),
        warnings=tuple(mode_warnings),
        errors=(),
    )


def _capture_cache_boundary(
    cache: Any,
    hidden: Any,
    *,
    call_index: int,
    attention_evidence: Mapping[str, Any],
) -> BoundaryCapture:
    if cache is None:
        raise ParityProbeError("talker.model output did not expose a Cache")
    layers = getattr(cache, "layers", None)
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not isinstance(layers, (list, tuple)) or not callable(get_seq_length):
        raise ParityProbeError("past_key_values does not expose public Cache layers")
    if not layers:
        raise ParityProbeError("past_key_values Cache has no layers")
    captured_layers = []
    for layer_index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None or keys.ndim != 4 or values.ndim != 4:
            raise ParityProbeError(
                f"Cache layer {layer_index} does not expose rank-4 keys and values"
            )
        seq_length = int(get_seq_length(layer_index))
        if seq_length <= 0 or seq_length > keys.shape[-2] or seq_length > values.shape[-2]:
            raise ParityProbeError(
                f"Cache layer {layer_index} reported invalid sequence length {seq_length}"
            )
        key_prefix = keys[..., :seq_length, :]
        value_prefix = values[..., :seq_length, :]
        captured_layers.append(
            LayerKvCapture(
                layer_index=layer_index,
                key=capture_tensor(
                    f"boundary.{call_index}.layer.{layer_index}.key", key_prefix
                ),
                value=capture_tensor(
                    f"boundary.{call_index}.layer.{layer_index}.value", value_prefix
                ),
            )
        )
    return BoundaryCapture(
        call_index=call_index,
        phase="prefill" if call_index == 0 else "decode",
        layers=tuple(captured_layers),
        last_hidden=capture_tensor(f"boundary.{call_index}.last_hidden", hidden),
        attention_evidence=_plain_copy(attention_evidence),
    )


def _raw_codec_from_generation(generation_result: Any) -> Any:
    import torch

    histories = getattr(generation_result, "hidden_states", None)
    if not isinstance(histories, (list, tuple)):
        raise ParityProbeError("talker.generate did not return hidden-state histories")
    codec_steps = []
    for history in histories:
        if not isinstance(history, (list, tuple)) or len(history) < 2:
            raise ParityProbeError("talker hidden-state history has an unsupported shape")
        codec_ids = history[-1]
        if codec_ids is not None:
            codec_steps.append(codec_ids)
    if not codec_steps:
        raise ParityProbeError("talker.generate returned no raw codec steps")
    return torch.stack(codec_steps, dim=1)


def _decoded_codec_items(raw_codec: Any, eos_token_id: int) -> tuple[Any, ...]:
    first_codebook = raw_codec[:, :, 0].detach().to(device="cpu")
    items = []
    for row_index, row in enumerate(first_codebook):
        matches = row.eq(int(eos_token_id)).nonzero(as_tuple=False)
        length = int(matches[0, 0]) if matches.numel() else int(row.shape[0])
        items.append(raw_codec[row_index, :length])
    return tuple(items)


def _stack_predictor_sequences(recorder: HookRecorder, raw_codec: Any) -> Any:
    import torch

    expected_steps = int(raw_codec.shape[1])
    if len(recorder.predictor_sequences) != expected_steps:
        raise ParityProbeError(
            "code_predictor.generate call count does not match raw codec frame count"
        )
    return torch.stack(
        [snapshot.tensor for snapshot in recorder.predictor_sequences], dim=1
    )


def _stack_sampled_first_codebook(recorder: HookRecorder, raw_codec: Any) -> Any:
    import torch

    expected_steps = int(raw_codec.shape[1])
    if len(recorder.sampled_first_codebook) != expected_steps:
        raise ParityProbeError(
            "sampled first-codebook count does not match raw codec frame count"
        )
    sampled = torch.cat(
        [
            snapshot.tensor.reshape(raw_codec.shape[0], -1)
            for snapshot in recorder.sampled_first_codebook
        ],
        dim=1,
    )
    if tuple(sampled.shape) != tuple(raw_codec[:, :, 0].shape):
        raise ParityProbeError(
            "sampled first-codebook shape does not match raw codec history"
        )
    return sampled


def _attention_evidence(talker: Any) -> dict[str, Any]:
    import torch

    config = getattr(getattr(talker, "model", None), "config", None)
    if config is None:
        config = getattr(talker, "config", None)
    evidence: dict[str, Any] = {
        "evidence_kind": "public_configuration",
        "observed_undocumented_kernel": None,
        "attn_implementation": getattr(config, "_attn_implementation", None),
    }
    cuda_backend = getattr(torch.backends, "cuda", None)
    for report_name, method_name in (
        ("flash_sdp_enabled", "flash_sdp_enabled"),
        ("mem_efficient_sdp_enabled", "mem_efficient_sdp_enabled"),
        ("math_sdp_enabled", "math_sdp_enabled"),
        ("cudnn_sdp_enabled", "cudnn_sdp_enabled"),
    ):
        method = getattr(cuda_backend, method_name, None)
        evidence[report_name] = bool(method()) if callable(method) else None
    return evidence


def _collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dispatch_evidence_label": (
            "Public attention implementation metadata and CUDA SDPA backend flags; "
            "not an observed undocumented kernel"
        ),
    }
    try:
        import transformers

        result["transformers"] = transformers.__version__
    except Exception:
        result["transformers"] = None
    try:
        from importlib.metadata import version

        result["qwen_tts"] = version("qwen-tts")
    except Exception:
        result["qwen_tts"] = None
    if torch.cuda.is_available():
        device = torch.device(args.device)
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "requested_device": args.device,
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(device)),
            "physical_identity": _gpu_physical_identity(properties),
        }
    else:
        result["gpu"] = None
    return result


def _gpu_physical_identity(properties: Any) -> dict[str, Any]:
    return {
        "uuid": _public_property_text(properties, "uuid"),
        "pci_bus_id": _public_property_text(properties, "pci_bus_id"),
        "pci_device_id": getattr(properties, "pci_device_id", None),
        "source": "torch.cuda.get_device_properties public attributes",
    }


def _public_property_text(properties: Any, name: str) -> str | None:
    value = getattr(properties, name, None)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace")
    return str(value)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model_path:
        raise ValueError("--model-path is required")
    _parse_batch_sizes(args.batch_sizes)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.max_new_tokens < 5:
        raise ValueError("--max-new-tokens must be at least 5")
    if args.max_cache_len <= 0:
        raise ValueError("--max-cache-len must be positive")
    if args.requests < 0:
        raise ValueError("--requests must be non-negative")
    if not math.isfinite(args.duration_minutes) or args.duration_minutes < 0:
        raise ValueError("--duration-minutes must be finite and non-negative")
    if args.requests > 0 and args.duration_minutes > 0:
        if args.requests < 100:
            raise ValueError(
                "stability mode with both targets requires at least 100 requests"
            )
        if args.duration_minutes < 10:
            raise ValueError(
                "stability mode with both targets requires at least 10 minutes"
            )
    if _stability_enabled(args) and _parse_batch_sizes(args.batch_sizes) != [1, 2]:
        raise ValueError("stability mode requires --batch-sizes 1,2")
    if not args.texts or any(not str(text).strip() for text in args.texts):
        raise ValueError("--texts must contain non-empty values")


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--batch-sizes must contain positive integers") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    if len(set(sizes)) != len(sizes):
        raise ValueError("--batch-sizes must not contain duplicates")
    return sizes


def _texts_for_batch(texts: Sequence[str], batch_size: int) -> list[str]:
    return [str(texts[index % len(texts)]) for index in range(batch_size)]


def _stability_enabled(args: argparse.Namespace) -> bool:
    return int(args.requests) > 0 or float(args.duration_minutes) > 0


def _stability_request_texts(
    texts: Sequence[str],
    *,
    cursor: int,
    batch_size: int,
) -> tuple[list[str], list[int], int]:
    if not texts:
        raise ValueError("stability text rotation requires at least one text")
    indices = [
        int((cursor + offset) % len(texts)) for offset in range(int(batch_size))
    ]
    values = [str(texts[index]) for index in indices]
    if batch_size == 2 and values[0] == values[1]:
        values[1] = f"{values[1]} Please."
    return values, indices, cursor + batch_size


def _stability_target_pending(
    completed_requests: int,
    elapsed_seconds: float,
    *,
    request_target: int,
    duration_seconds: float,
) -> bool:
    requests_pending = request_target > 0 and completed_requests < request_target
    duration_pending = duration_seconds > 0 and elapsed_seconds < duration_seconds
    return requests_pending or duration_pending


def _left_pad_prepared_inputs(prepared: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    result = clone_prepared_inputs(dict(prepared))
    attention_mask = result.get("attention_mask")
    inputs_embeds = result.get("inputs_embeds")
    if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
        raise ParityProbeError(
            "left-padded stability batch requires a 2D attention_mask"
        )
    if not torch.is_tensor(inputs_embeds) or inputs_embeds.ndim < 3:
        raise ParityProbeError(
            "left-padded stability batch requires batched inputs_embeds"
        )
    if tuple(inputs_embeds.shape[:2]) != tuple(attention_mask.shape):
        raise ParityProbeError(
            "attention_mask must match stability inputs_embeds batch and prompt"
        )
    padded_embeds = inputs_embeds.clone()
    padded_mask = attention_mask.clone()
    for row in range(int(attention_mask.shape[0])):
        valid = attention_mask[row].to(dtype=torch.bool)
        order = torch.cat(
            (
                (~valid).nonzero(as_tuple=False).reshape(-1),
                valid.nonzero(as_tuple=False).reshape(-1),
            )
        )
        padded_embeds[row] = inputs_embeds[row].index_select(0, order)
        padded_mask[row] = attention_mask[row].index_select(0, order)
    result["inputs_embeds"] = padded_embeds
    result["attention_mask"] = padded_mask
    return result


def _left_padding_evidence(prepared: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    mask = prepared.get("attention_mask")
    if not torch.is_tensor(mask) or mask.ndim != 2:
        return {"valid_lengths": [], "left_padding_tokens": [], "mask": []}
    left_padding_tokens = []
    valid_lengths = []
    for row in mask:
        values = row.to(dtype=torch.bool)
        valid = int(values.sum().item())
        first_valid = int(values.nonzero(as_tuple=False)[0].item()) if valid else 0
        valid_lengths.append(valid)
        left_padding_tokens.append(first_valid)
    return {
        "valid_lengths": valid_lengths,
        "left_padding_tokens": left_padding_tokens,
        "mask": mask.detach().to(device="cpu").tolist(),
    }


def _memory_snapshot(device: str, *, cuda_api: Any | None = None) -> dict[str, int]:
    if cuda_api is None:
        import torch

        cuda_api = torch.cuda
    if not str(device).startswith("cuda") or not bool(cuda_api.is_available()):
        return {
            "memory_allocated": 0,
            "memory_reserved": 0,
            "max_memory_allocated": 0,
            "max_memory_reserved": 0,
        }
    return {
        "memory_allocated": int(cuda_api.memory_allocated(device)),
        "memory_reserved": int(cuda_api.memory_reserved(device)),
        "max_memory_allocated": int(cuda_api.max_memory_allocated(device)),
        "max_memory_reserved": int(cuda_api.max_memory_reserved(device)),
    }


def _active_cache_snapshot(engine: Any) -> dict[str, Any]:
    metrics = {}
    metrics_snapshot = getattr(engine, "metrics_snapshot", None)
    if callable(metrics_snapshot):
        metrics = _plain_copy(metrics_snapshot())
    runtime = getattr(engine, "step_runtime", None)
    pointer_snapshot = getattr(runtime, "backing_pointers_snapshot", None)
    if callable(pointer_snapshot):
        pointers = _plain_copy(pointer_snapshot())
    else:
        pointers = []
        caches = getattr(runtime, "_caches", {})
        values = caches.values() if isinstance(caches, Mapping) else ()
        for cache in values:
            for layer_index, layer in enumerate(getattr(cache, "layers", ())):
                keys = getattr(layer, "keys", None)
                values_tensor = getattr(layer, "values", None)
                key_pointer = getattr(keys, "data_ptr", None)
                value_pointer = getattr(values_tensor, "data_ptr", None)
                if callable(key_pointer) and callable(value_pointer):
                    pointers.append(
                        {
                            "cache_identity": int(id(cache)),
                            "layer_index": int(layer_index),
                            "key": int(key_pointer()),
                            "value": int(value_pointer()),
                        }
                    )
    return {"metrics": metrics, "backing_pointers": pointers}


def _exact_hashes(
    reference: ModeCapture | None,
    candidate: ModeCapture | None,
) -> dict[str, Any]:
    def mode_hashes(capture: ModeCapture | None) -> dict[str, Any] | None:
        if capture is None:
            return None
        return {
            "raw_codec": capture.codec.raw_sha256,
            "combined_codec": capture.codec.combined_sha256,
            "codec_items": list(capture.codec.item_sha256),
        }

    upstream = mode_hashes(reference)
    active = mode_hashes(candidate)
    return {
        "upstream": upstream,
        "active_prefix": active,
        "equal": upstream is not None and upstream == active,
    }


def _config_report(args: argparse.Namespace) -> dict[str, Any]:
    try:
        batch_sizes: list[int] | str = _parse_batch_sizes(args.batch_sizes)
    except ValueError:
        batch_sizes = args.batch_sizes
    return {
        "model_path": args.model_path,
        "device": args.device,
        "language": args.language,
        "speaker": args.speaker,
        "texts": list(args.texts),
        "batch_sizes": batch_sizes,
        "seed": int(args.seed),
        "max_new_tokens": int(args.max_new_tokens),
        "max_cache_len": int(args.max_cache_len),
        "requests": int(args.requests),
        "duration_minutes": float(args.duration_minutes),
        "out": str(args.out),
        "outer_sampling": "greedy",
        "predictor_sampling": "greedy",
        "prepared_inputs": "prepared/tokenized once per batch and cloned per mode",
        "captured_boundaries": "prefill plus first four decode calls, every layer",
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    print(payload, end="")


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _synchronize(device: str) -> None:
    import torch

    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _capture_attribute_state(owner: Any, name: str) -> _AttributeState:
    namespace = getattr(owner, "__dict__", {})
    return _AttributeState(name in namespace, namespace.get(name))


def _restore_attribute_state(owner: Any, name: str, state: _AttributeState) -> None:
    if state.had_instance_value:
        setattr(owner, name, state.instance_value)
        return
    namespace = getattr(owner, "__dict__", {})
    if name in namespace:
        delattr(owner, name)


def _validate_restored_attribute(
    owner: Any,
    name: str,
    expected_state: _AttributeState,
    *,
    expected_callable: Callable[..., Any] | None = None,
) -> None:
    actual_state = _capture_attribute_state(owner, name)
    state_matches = actual_state.had_instance_value == expected_state.had_instance_value
    if state_matches and expected_state.had_instance_value:
        state_matches = actual_state.instance_value is expected_state.instance_value
    if not state_matches:
        raise ParityProbeError(f"talker.{name} restoration failed")
    if expected_callable is not None:
        actual_callable = _required_callable(owner, name)
        if actual_callable != expected_callable:
            raise ParityProbeError(f"talker.{name} restoration failed")


def _required_member(owner: Any, name: str) -> Any:
    value = getattr(owner, name, None)
    if value is None:
        raise ParityProbeError(f"installed API is missing {name}")
    return value


def _required_callable(owner: Any, name: str) -> Callable[..., Any]:
    value = getattr(owner, name, None)
    if not callable(value):
        raise ParityProbeError(f"installed API is missing callable {name}")
    return value


def _snapshot_tensor(value: Any) -> Any:
    return value.tensor if isinstance(value, TensorSnapshot) else value


def _tensor_bytes(tensor: Any) -> Any:
    import torch

    return tensor.detach().to(device="cpu").contiguous().view(-1).view(torch.uint8)


def _tensor_raw_bytes(tensor: Any) -> bytes:
    import torch

    byte_tensor = tensor.detach().to(device="cpu").contiguous().view(-1).view(torch.uint8)
    return bytes(byte_tensor.tolist())


def _safe_max_abs(reference: Any, candidate: Any) -> float | None:
    import torch

    if reference.numel() == 0:
        return 0.0
    left = reference.detach().to(device="cpu")
    right = candidate.detach().to(device="cpu")
    if left.is_complex() or right.is_complex():
        left = left.to(torch.complex128)
        right = right.to(torch.complex128)
    else:
        left = left.to(torch.float64)
        right = right.to(torch.float64)
    equal_values = left.eq(right)
    difference = (left - right).abs()
    difference = torch.where(equal_values, torch.zeros_like(difference), difference)
    if not bool(torch.isfinite(difference).all()):
        return None
    return float(difference.max())


def _unravel_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    indices = []
    remaining = flat_index
    for size in reversed(shape):
        indices.append(remaining % size)
        remaining //= size
    return tuple(reversed(indices))


def _plain_copy(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return {"non_finite_float": "nan"}
        if value == float("inf"):
            return {"non_finite_float": "positive_infinity"}
        if value == float("-inf"):
            return {"non_finite_float": "negative_infinity"}
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_copy(item) for item in value]
    raise TypeError(f"report contains non-plain value: {type(value).__name__}")


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _merge_warning_messages(*collections: Sequence[str]) -> tuple[str, ...]:
    merged = []
    for collection in collections:
        for message in collection:
            if message not in merged:
                merged.append(message)
    return tuple(merged)


if __name__ == "__main__":
    raise SystemExit(main())
