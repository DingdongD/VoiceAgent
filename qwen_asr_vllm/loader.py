"""Checkpoint loading.

Qwen3-ASR checkpoints store every tensor under a ``thinker.`` prefix and keep the
attention and MLP projections separate, while this implementation holds them
fused. The loader resolves both differences and refuses to finish quietly if a
parameter was left untouched or a checkpoint tensor found no home -- a silently
unloaded layer produces plausible-looking gibberish that is expensive to debug
later.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterator

import torch
from safetensors import safe_open
from torch import nn

CHECKPOINT_PREFIX = "thinker."

_FUSED_SHARDS = {
    "self_attn.q_proj.": ("self_attn.qkv_proj.", "q"),
    "self_attn.k_proj.": ("self_attn.qkv_proj.", "k"),
    "self_attn.v_proj.": ("self_attn.qkv_proj.", "v"),
    "mlp.gate_proj.": ("mlp.gate_up_proj.", "gate"),
    "mlp.up_proj.": ("mlp.gate_up_proj.", "up"),
}
_FUSED_GROUPS = {
    "self_attn.qkv_proj.": {"q", "k", "v"},
    "mlp.gate_up_proj.": {"gate", "up"},
}


def _iter_checkpoint_tensors(model_path: str) -> Iterator[tuple[str, torch.Tensor]]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            shards = sorted(set(json.load(handle)["weight_map"].values()))
        files = [os.path.join(model_path, shard) for shard in shards]
    else:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors found under {model_path}")

    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                yield name, reader.get_tensor(name)


class _FusedLayout:
    """Row offsets of each checkpoint projection inside a fused parameter."""

    def __init__(self, q_size: int, kv_size: int, intermediate_size: int):
        self.offsets = {
            "q": (0, q_size),
            "k": (q_size, kv_size),
            "v": (q_size + kv_size, kv_size),
            "gate": (0, intermediate_size),
            "up": (intermediate_size, intermediate_size),
        }

    def __getitem__(self, shard: str) -> tuple[int, int]:
        return self.offsets[shard]


TEXT_DECODER_PREFIX = "model.layers."


def _resolve(name: str) -> tuple[str, str | None]:
    """Return ``(param_name, shard_id)``; ``shard_id`` is None for direct copies.

    Only the text decoder stores projections fused. The audio tower uses the same
    ``q_proj``/``k_proj``/``v_proj`` names but keeps them separate, so fusion is
    scoped by prefix instead of matched on the suffix alone.
    """
    if name.startswith(TEXT_DECODER_PREFIX):
        for marker, (fused_marker, shard) in _FUSED_SHARDS.items():
            if marker in name:
                return name.replace(marker, fused_marker), shard
    return name, None


def load_weights(model: nn.Module, model_path: str) -> None:
    text = model.config.text
    head_dim = text.head_dim or text.hidden_size // text.num_attention_heads
    layout = _FusedLayout(
        q_size=text.num_attention_heads * head_dim,
        kv_size=text.num_key_value_heads * head_dim,
        intermediate_size=text.intermediate_size,
    )

    params = dict(model.named_parameters())
    aliases: dict[str, str] = {}
    if "lm_head.weight" not in params:
        # Tied embeddings: named_parameters() reports the shared tensor once.
        aliases["lm_head.weight"] = "model.embed_tokens.weight"

    written: dict[str, set[str | None]] = {}
    orphans: list[str] = []

    for raw_name, tensor in _iter_checkpoint_tensors(model_path):
        if not raw_name.startswith(CHECKPOINT_PREFIX):
            orphans.append(raw_name)
            continue
        name = raw_name[len(CHECKPOINT_PREFIX) :]
        target, shard = _resolve(name)
        target = aliases.get(target, target)

        param = params.get(target)
        if param is None:
            orphans.append(raw_name)
            continue

        with torch.no_grad():
            if shard is None:
                if param.shape != tensor.shape:
                    raise ValueError(
                        f"shape mismatch for {target}: checkpoint {tuple(tensor.shape)} "
                        f"vs model {tuple(param.shape)}"
                    )
                param.copy_(tensor)
            else:
                offset, length = layout[shard]
                if tensor.size(0) != length:
                    raise ValueError(
                        f"shard {shard} of {raw_name} has {tensor.size(0)} rows, expected {length}"
                    )
                param.narrow(0, offset, length).copy_(tensor)

        written.setdefault(target, set()).add(shard)

    if orphans:
        raise RuntimeError(
            f"{len(orphans)} checkpoint tensors had no destination, e.g. {sorted(orphans)[:5]}"
        )

    missing = sorted(set(params) - set(written))
    if missing:
        raise RuntimeError(f"{len(missing)} parameters were never loaded, e.g. {missing[:5]}")

    for target, shards in written.items():
        for fused_marker, required in _FUSED_GROUPS.items():
            if fused_marker in target and shards != required:
                raise RuntimeError(
                    f"{target} is missing fused shards {sorted(required - shards)}"
                )
