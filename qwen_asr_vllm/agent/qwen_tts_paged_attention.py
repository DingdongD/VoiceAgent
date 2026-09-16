"""Triton decode attention that reads request-owned KV pages directly."""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - optional on CPU-only installations.
    triton = None
    tl = None
    HAS_TRITON = False


class PagedAttentionKernelError(RuntimeError):
    """Raised when the paged decode attention contract is invalid."""


if HAS_TRITON:

    @triton.jit
    def _fused_qkv_projection_kernel(
        hidden_ptr,
        q_weight_ptr,
        k_weight_ptr,
        v_weight_ptr,
        q_bias_ptr,
        k_bias_ptr,
        v_bias_ptr,
        q_out_ptr,
        k_out_ptr,
        v_out_ptr,
        m_size,
        hidden_size,
        q_size,
        kv_size,
        hidden_stride_m,
        hidden_stride_k,
        q_weight_stride_m,
        q_weight_stride_k,
        k_weight_stride_m,
        k_weight_stride_k,
        v_weight_stride_m,
        v_weight_stride_k,
        q_out_stride_m,
        q_out_stride_n,
        k_out_stride_m,
        k_out_stride_n,
        v_out_stride_m,
        v_out_stride_n,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute three row-major projections while loading each hidden tile once."""

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = row_offsets < m_size
        q_col_mask = col_offsets < q_size
        kv_col_mask = col_offsets < kv_size
        q_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        v_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, hidden_size, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            hidden = tl.load(
                hidden_ptr
                + row_offsets[:, None] * hidden_stride_m
                + k_offsets[None, :] * hidden_stride_k,
                mask=row_mask[:, None] & (k_offsets[None, :] < hidden_size),
                other=0.0,
            )
            q_weight = tl.load(
                q_weight_ptr
                + col_offsets[:, None] * q_weight_stride_m
                + k_offsets[None, :] * q_weight_stride_k,
                mask=q_col_mask[:, None] & (k_offsets[None, :] < hidden_size),
                other=0.0,
            )
            k_weight = tl.load(
                k_weight_ptr
                + col_offsets[:, None] * k_weight_stride_m
                + k_offsets[None, :] * k_weight_stride_k,
                mask=kv_col_mask[:, None] & (k_offsets[None, :] < hidden_size),
                other=0.0,
            )
            v_weight = tl.load(
                v_weight_ptr
                + col_offsets[:, None] * v_weight_stride_m
                + k_offsets[None, :] * v_weight_stride_k,
                mask=kv_col_mask[:, None] & (k_offsets[None, :] < hidden_size),
                other=0.0,
            )
            q_acc += tl.dot(hidden, tl.trans(q_weight), out_dtype=tl.float32)
            k_acc += tl.dot(hidden, tl.trans(k_weight), out_dtype=tl.float32)
            v_acc += tl.dot(hidden, tl.trans(v_weight), out_dtype=tl.float32)

        if HAS_BIAS:
            q_acc += tl.load(q_bias_ptr + col_offsets, mask=q_col_mask, other=0.0)[None, :]
            k_acc += tl.load(k_bias_ptr + col_offsets, mask=kv_col_mask, other=0.0)[None, :]
            v_acc += tl.load(v_bias_ptr + col_offsets, mask=kv_col_mask, other=0.0)[None, :]
        tl.store(
            q_out_ptr + row_offsets[:, None] * q_out_stride_m + col_offsets[None, :] * q_out_stride_n,
            q_acc,
            mask=row_mask[:, None] & q_col_mask[None, :],
        )
        tl.store(
            k_out_ptr + row_offsets[:, None] * k_out_stride_m + col_offsets[None, :] * k_out_stride_n,
            k_acc,
            mask=row_mask[:, None] & kv_col_mask[None, :],
        )
        tl.store(
            v_out_ptr + row_offsets[:, None] * v_out_stride_m + col_offsets[None, :] * v_out_stride_n,
            v_acc,
            mask=row_mask[:, None] & kv_col_mask[None, :],
        )

    @triton.jit
    def _paged_decode_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        new_k_ptr,
        new_v_ptr,
        page_table_ptr,
        seq_lens_ptr,
        valid_mask_ptr,
        out_ptr,
        q_stride_b,
        q_stride_h,
        q_stride_d,
        kv_stride_p,
        kv_stride_h,
        kv_stride_s,
        kv_stride_d,
        table_stride_b,
        valid_stride_b,
        out_stride_b,
        out_stride_h,
        out_stride_d,
        new_stride_b,
        new_stride_h,
        new_stride_d,
        scale,
        num_heads,
        num_kv_heads,
        page_size,
        num_page_slots,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        MAX_SEQ_LEN: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HAS_NEW_KV: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch = pid // num_heads
        head = pid % num_heads
        group_size = num_heads // num_kv_heads
        kv_head = head // group_size
        seq_len = tl.load(seq_lens_ptr + batch)

        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < HEAD_DIM
        q = tl.load(
            q_ptr + batch * q_stride_b + head * q_stride_h + dim_offsets * q_stride_d,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        if HAS_NEW_KV:
            current_position = seq_len - 1
            current_page = current_position // page_size
            current_offset = current_position % page_size
            current_page_id = tl.load(
                page_table_ptr + batch * table_stride_b + current_page
            )
            new_key = tl.load(
                new_k_ptr
                + batch * new_stride_b
                + kv_head * new_stride_h
                + dim_offsets * new_stride_d,
                mask=dim_mask,
                other=0.0,
            )
            new_value = tl.load(
                new_v_ptr
                + batch * new_stride_b
                + kv_head * new_stride_h
                + dim_offsets * new_stride_d,
                mask=dim_mask,
                other=0.0,
            )
            tl.store(
                k_ptr
                + current_page_id * kv_stride_p
                + kv_head * kv_stride_h
                + current_offset * kv_stride_s
                + dim_offsets * kv_stride_d,
                new_key,
                mask=dim_mask,
            )
            tl.store(
                v_ptr
                + current_page_id * kv_stride_p
                + kv_head * kv_stride_h
                + current_offset * kv_stride_s
                + dim_offsets * kv_stride_d,
                new_value,
                mask=dim_mask,
            )

        running_max = tl.full((), -1.0e20, dtype=tl.float32)
        running_sum = tl.zeros((), dtype=tl.float32)
        running_out = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for block_start in range(0, MAX_SEQ_LEN, BLOCK_N):
            token_offsets = block_start + tl.arange(0, BLOCK_N)
            token_mask = token_offsets < seq_len
            token_mask &= tl.load(
                valid_mask_ptr + batch * valid_stride_b + token_offsets,
                mask=token_offsets < MAX_SEQ_LEN,
                other=0,
            ).to(tl.int1)
            page_indices = token_offsets // page_size
            page_offsets = token_offsets % page_size
            page_ids = tl.load(
                page_table_ptr + batch * table_stride_b + page_indices,
                mask=page_indices < num_page_slots,
                other=0,
            )
            safe_page_ids = tl.where(token_mask, page_ids, 0)

            key_ptrs = (
                k_ptr
                + safe_page_ids[:, None] * kv_stride_p
                + kv_head * kv_stride_h
                + page_offsets[:, None] * kv_stride_s
                + dim_offsets[None, :] * kv_stride_d
            )
            value_ptrs = (
                v_ptr
                + safe_page_ids[:, None] * kv_stride_p
                + kv_head * kv_stride_h
                + page_offsets[:, None] * kv_stride_s
                + dim_offsets[None, :] * kv_stride_d
            )
            key = tl.load(key_ptrs, mask=token_mask[:, None] & dim_mask[None, :], other=0.0).to(
                tl.float32
            )
            value = tl.load(
                value_ptrs,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(key * q[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -1.0e20)

            block_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - next_max)
            new_scale = tl.exp(scores - next_max)
            running_sum = old_scale * running_sum + tl.sum(new_scale, axis=0)
            running_out = old_scale * running_out + tl.sum(
                new_scale[:, None] * value, axis=0
            )
            running_max = next_max

        output = running_out / tl.maximum(running_sum, 1.0e-20)
        tl.store(
            out_ptr
            + batch * out_stride_b
            + head * out_stride_h
            + dim_offsets * out_stride_d,
            output,
            mask=dim_mask,
        )


def _validate_inputs(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    key_valid_mask: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    tensors = {
        "query": query,
        "key_pool": key_pool,
        "value_pool": value_pool,
        "page_table": page_table,
        "seq_lens": seq_lens,
    }
    if any(not torch.is_tensor(value) for value in tensors.values()):
        raise PagedAttentionKernelError("paged attention inputs must be tensors")
    if query.ndim != 4 or query.shape[2] != 1:
        raise PagedAttentionKernelError("query must have shape [batch, heads, 1, head_dim]")
    if key_pool.ndim != 4 or value_pool.shape != key_pool.shape:
        raise PagedAttentionKernelError("KV pool must have shape [pages, kv_heads, page_size, head_dim]")
    if page_table.ndim != 2 or seq_lens.ndim != 1 or page_table.shape[0] != query.shape[0]:
        raise PagedAttentionKernelError("page table and sequence lengths must match query batch")
    if seq_lens.shape[0] != query.shape[0] or seq_lens.dtype != torch.long:
        raise PagedAttentionKernelError("seq_lens must be a torch.long vector")
    if page_table.dtype != torch.long:
        raise PagedAttentionKernelError("page_table must be a torch.long matrix")
    device = query.device
    if any(value.device != device for value in tensors.values()):
        raise PagedAttentionKernelError("paged attention inputs must share a device")
    if key_pool.dtype != query.dtype or value_pool.dtype != query.dtype:
        raise PagedAttentionKernelError("query and KV pool must share a dtype")
    if key_pool.shape[3] != query.shape[3]:
        raise PagedAttentionKernelError("query and KV pool head dimensions must match")
    if query.shape[1] % key_pool.shape[1] != 0:
        raise PagedAttentionKernelError("query heads must be divisible by KV heads")
    if key_valid_mask is not None:
        if (
            key_valid_mask.ndim != 2
            or key_valid_mask.shape[0] != query.shape[0]
            or key_valid_mask.dtype != torch.bool
            or key_valid_mask.device != device
        ):
            raise PagedAttentionKernelError("key_valid_mask must be [batch, max_seq] bool on device")
    if bool((seq_lens <= 0).any().item()):
        raise PagedAttentionKernelError("sequence lengths must be positive")
    if bool((seq_lens > page_table.shape[1] * key_pool.shape[2]).any().item()):
        raise PagedAttentionKernelError("sequence length exceeds page table capacity")
    page_counts = (seq_lens + key_pool.shape[2] - 1) // key_pool.shape[2]
    for row in range(query.shape[0]):
        used = page_table[row, : int(page_counts[row].item())]
        if bool(((used < 0) | (used >= key_pool.shape[0])).any().item()):
            raise PagedAttentionKernelError("page table contains an invalid page table entry")
        if key_valid_mask is not None and key_valid_mask.shape[1] < int(seq_lens[row].item()):
            raise PagedAttentionKernelError("key_valid_mask is shorter than sequence length")
    return (
        int(query.shape[0]),
        int(query.shape[1]),
        int(key_pool.shape[1]),
        int(key_pool.shape[2]),
    )


def _reference_paged_attention(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    key_valid_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    outputs = []
    for row in range(query.shape[0]):
        length = int(seq_lens[row].item())
        positions = torch.arange(length, device=query.device)
        page_ids = page_table[row].index_select(0, positions // key_pool.shape[2])
        offsets = positions % key_pool.shape[2]
        keys = key_pool[page_ids, :, offsets, :].permute(1, 0, 2).unsqueeze(0)
        values = value_pool[page_ids, :, offsets, :].permute(1, 0, 2).unsqueeze(0)
        repeat = query.shape[1] // key_pool.shape[1]
        keys = keys.repeat_interleave(repeat, dim=1)
        values = values.repeat_interleave(repeat, dim=1)
        scores = torch.matmul(query[row : row + 1], keys.transpose(-1, -2)) * scale
        if key_valid_mask is not None:
            scores = scores.masked_fill(
                ~key_valid_mask[row, :length].view(1, 1, 1, length),
                torch.finfo(scores.dtype).min,
            )
        outputs.append(torch.softmax(scores, dim=-1) @ values)
    return torch.cat(outputs, dim=0)


def paged_attention_decode(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    key_valid_mask: torch.Tensor | None = None,
    *,
    new_key: torch.Tensor | None = None,
    new_value: torch.Tensor | None = None,
    kernel_seq_len: int | None = None,
    kernel_page_count: int | None = None,
    block_n: int = 128,
    num_warps: int = 4,
    scale: float | None = None,
    use_triton: bool = True,
    validate: bool = True,
) -> torch.Tensor:
    """Compute one-token GQA attention directly from a physical page pool.

    ``page_table`` maps each request's logical page to a physical page in the
    pool. No dense ``[batch, heads, sequence, dim]`` K/V tensor is materialized
    on the Triton path. ``key_valid_mask`` supports left-padded prompts.
    """
    if validate:
        _, num_heads, num_kv_heads, page_size = _validate_inputs(
            query, key_pool, value_pool, page_table, seq_lens, key_valid_mask
        )
    else:
        num_heads = int(query.shape[1])
        num_kv_heads = int(key_pool.shape[1])
        page_size = int(key_pool.shape[2])
    scale = float(scale if scale is not None else 1.0 / math.sqrt(query.shape[-1]))
    if (new_key is None) != (new_value is None):
        raise PagedAttentionKernelError("new_key and new_value must be provided together")
    if new_key is not None:
        expected_shape = (query.shape[0], num_kv_heads, 1, query.shape[-1])
        if (
            tuple(new_key.shape) != expected_shape
            or tuple(new_value.shape) != expected_shape
            or new_key.dtype != query.dtype
            or new_value.dtype != query.dtype
            or new_key.device != query.device
            or new_value.device != query.device
        ):
            raise PagedAttentionKernelError(
                "new KV must have shape [batch, kv_heads, 1, head_dim] and match query"
            )
    if not use_triton:
        if new_key is not None:
            _write_current_kv(
                key_pool, value_pool, page_table, seq_lens, new_key, new_value
            )
        return _reference_paged_attention(
            query, key_pool, value_pool, page_table, seq_lens, key_valid_mask, scale
        )
    if not HAS_TRITON or query.device.type != "cuda":
        raise PagedAttentionKernelError("Triton paged attention requires CUDA and Triton")
    block_n = int(block_n)
    if block_n <= 0 or block_n & (block_n - 1):
        raise PagedAttentionKernelError("block_n must be a positive power of two")
    num_warps = int(num_warps)
    if num_warps <= 0:
        raise PagedAttentionKernelError("num_warps must be positive")
    if kernel_seq_len is None:
        max_seq_len = int(seq_lens.max().item())
        kernel_seq_len = int(math.ceil(max_seq_len / block_n) * block_n)
    else:
        kernel_seq_len = int(kernel_seq_len)
    if kernel_page_count is None:
        kernel_page_count = int(math.ceil(kernel_seq_len / page_size))
    else:
        kernel_page_count = int(kernel_page_count)
    if kernel_seq_len <= 0 or kernel_page_count <= 0:
        raise PagedAttentionKernelError("kernel metadata dimensions must be positive")
    if page_table.shape[1] < kernel_page_count:
        padded_page_table = torch.zeros(
            (page_table.shape[0], kernel_page_count),
            dtype=page_table.dtype,
            device=page_table.device,
        )
        padded_page_table[:, : page_table.shape[1]] = page_table
        page_table = padded_page_table
    if key_valid_mask is None:
        valid = torch.ones(
            (query.shape[0], kernel_seq_len), dtype=torch.bool, device=query.device
        )
    elif key_valid_mask.shape[1] < kernel_seq_len:
        valid = torch.ones(
            (query.shape[0], kernel_seq_len), dtype=torch.bool, device=query.device
        )
        valid[:, : key_valid_mask.shape[1]] = key_valid_mask
    else:
        valid = key_valid_mask[:, :kernel_seq_len]
    output = torch.empty_like(query)
    block_d = int(triton.next_power_of_2(query.shape[-1]))
    _paged_decode_attention_kernel[(query.shape[0] * query.shape[1],)](
        query,
        key_pool,
        value_pool,
        key_pool if new_key is None else new_key,
        value_pool if new_value is None else new_value,
        page_table,
        seq_lens,
        valid,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(3),
        key_pool.stride(0),
        key_pool.stride(1),
        key_pool.stride(2),
        key_pool.stride(3),
        page_table.stride(0),
        valid.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(3),
        (0 if new_key is None else new_key.stride(0)),
        (0 if new_key is None else new_key.stride(1)),
        (0 if new_key is None else new_key.stride(3)),
        scale,
        num_heads,
        num_kv_heads,
        page_size,
        kernel_page_count,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        MAX_SEQ_LEN=kernel_seq_len,
        HEAD_DIM=query.shape[-1],
        HAS_NEW_KV=new_key is not None,
        num_warps=num_warps,
    )
    return output


def _write_current_kv(
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
) -> None:
    for row in range(new_key.shape[0]):
        position = int(seq_lens[row].item()) - 1
        page = int(page_table[row, position // key_pool.shape[2]].item())
        offset = position % key_pool.shape[2]
        key_pool[page, :, offset, :] = new_key[row, :, 0, :]
        value_pool[page, :, offset, :] = new_value[row, :, 0, :]


def fused_qkv_projection(module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Project Q/K/V with one concatenated linear operation when enabled."""
    weight = getattr(module, "_qav_fused_qkv_weight", None)
    bias = getattr(module, "_qav_fused_qkv_bias", None)
    if weight is None:
        weights = (module.q_proj.weight, module.k_proj.weight, module.v_proj.weight)
        biases = (module.q_proj.bias, module.k_proj.bias, module.v_proj.bias)
        if any(value is not None for value in biases) and not all(
            value is not None for value in biases
        ):
            return module.q_proj(hidden_states), module.k_proj(hidden_states), module.v_proj(hidden_states)
        weight = torch.cat(weights, dim=0).contiguous()
        bias = torch.cat(biases, dim=0).contiguous() if biases[0] is not None else None
        module._qav_fused_qkv_weight = weight
        module._qav_fused_qkv_bias = bias
    projected = torch.nn.functional.linear(hidden_states, weight, bias)
    q_size = module.q_proj.out_features
    k_size = module.k_proj.out_features
    return projected.split((q_size, k_size, module.v_proj.out_features), dim=-1)


def projection_error_report(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    """Summarize projection drift without changing the tensors or their dtype."""

    if not torch.is_tensor(reference) or not torch.is_tensor(candidate):
        raise PagedAttentionKernelError("projection comparison expects tensors")
    if reference.shape != candidate.shape:
        raise PagedAttentionKernelError("projection tensors must have equal shapes")
    reference_float = reference.detach().to(dtype=torch.float32)
    candidate_float = candidate.detach().to(dtype=torch.float32)
    difference = (candidate_float - reference_float).abs()
    reference_norm = torch.linalg.vector_norm(reference_float).clamp_min(1.0e-12)
    relative_l2 = torch.linalg.vector_norm(candidate_float - reference_float) / reference_norm
    if reference.shape and reference.shape[-1] > 0:
        top1_match = torch.argmax(reference_float, dim=-1).eq(
            torch.argmax(candidate_float, dim=-1)
        )
        top1_match_value = bool(top1_match.all().item())
        top1_mismatch_count = int((~top1_match).sum().item())
    else:
        top1_match_value = True
        top1_mismatch_count = 0
    return {
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "relative_l2": float(relative_l2.item()),
        "top1_match": top1_match_value,
        "top1_mismatch_count": top1_mismatch_count,
    }


def calibrate_fused_qkv(
    talker: Any,
    *,
    backend: str = "torch",
    seed: int = 0,
) -> dict[str, Any]:
    """Measure projection-only drift for every Qwen attention layer.

    This is a startup diagnostic, not an exact-codec guarantee. It deliberately
    does not execute the full talker, so it can reject an obviously unstable
    projection backend without adding a reference pass to every decode step.
    """

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    reports = []
    layers = getattr(getattr(talker, "model", None), "layers", None)
    if layers is None:
        raise PagedAttentionKernelError("Qwen talker model does not expose decoder layers")
    if backend not in {"torch", "triton"}:
        raise PagedAttentionKernelError(f"unsupported fused projection backend: {backend}")
    for index, layer in enumerate(layers):
        module = getattr(layer, "self_attn", None)
        if module is None:
            raise PagedAttentionKernelError("Qwen decoder layer does not expose self_attn")
        hidden = torch.randn(
            (2, 1, module.q_proj.in_features),
            generator=generator,
            dtype=module.q_proj.weight.dtype,
        ).to(device=module.q_proj.weight.device)
        reference = torch.cat(
            (
                module.q_proj(hidden),
                module.k_proj(hidden),
                module.v_proj(hidden),
            ),
            dim=-1,
        )
        if backend == "triton":
            candidate = torch.cat(fused_qkv_projection_kernel(module, hidden), dim=-1)
        else:
            candidate = torch.cat(fused_qkv_projection(module, hidden), dim=-1)
        report = projection_error_report(reference, candidate)
        report["layer"] = index
        reports.append(report)
    return {
        "backend": backend,
        "layers": reports,
        "max_abs": max((item["max_abs"] for item in reports), default=0.0),
        "max_relative_l2": max(
            (item["relative_l2"] for item in reports), default=0.0
        ),
        "top1_match": all(item["top1_match"] for item in reports),
    }


def fused_qkv_projection_kernel(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    use_triton: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Run the single-launch Triton QKV projection prototype.

    The kernel uses FP32 accumulation and writes separate Q/K/V tensors. It is
    intentionally explicit because its rounding and performance characteristics
    differ from PyTorch's vendor GEMM path.
    """

    if not use_triton:
        return fused_qkv_projection(module, hidden_states)
    if hidden_states.device.type != "cuda":
        raise PagedAttentionKernelError("custom fused QKV projection requires CUDA")
    if not HAS_TRITON:
        raise PagedAttentionKernelError("custom fused QKV projection requires Triton")
    _prepare_fused_qkv(module)
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise PagedAttentionKernelError("custom fused QKV projection supports floating tensors only")
    if any(weight.dtype != hidden_states.dtype for weight in (module.q_proj.weight, module.k_proj.weight, module.v_proj.weight)):
        raise PagedAttentionKernelError("custom fused QKV projection requires matching dtypes")
    input_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    m_size = hidden_states.numel() // hidden_size
    q_size = module.q_proj.out_features
    kv_size = module.k_proj.out_features
    flat_hidden = hidden_states.reshape(m_size, hidden_size).contiguous()
    q_output = torch.empty((m_size, q_size), device=hidden_states.device, dtype=hidden_states.dtype)
    k_output = torch.empty((m_size, kv_size), device=hidden_states.device, dtype=hidden_states.dtype)
    v_output = torch.empty((m_size, kv_size), device=hidden_states.device, dtype=hidden_states.dtype)
    if any(value is not None for value in (module.q_proj.bias, module.k_proj.bias, module.v_proj.bias)):
        if not all(value is not None for value in (module.q_proj.bias, module.k_proj.bias, module.v_proj.bias)):
            raise PagedAttentionKernelError("custom fused QKV projection requires all biases or none")
        q_bias, k_bias, v_bias = module.q_proj.bias, module.k_proj.bias, module.v_proj.bias
        has_bias = True
    else:
        q_bias = k_bias = v_bias = torch.empty((1,), device=hidden_states.device, dtype=hidden_states.dtype)
        has_bias = False
    block_m = 16
    block_n = 128
    block_k = 32
    grid = (
        triton.cdiv(m_size, block_m),
        triton.cdiv(max(q_size, kv_size), block_n),
    )
    _fused_qkv_projection_kernel[grid](
        flat_hidden,
        module.q_proj.weight,
        module.k_proj.weight,
        module.v_proj.weight,
        q_bias,
        k_bias,
        v_bias,
        q_output,
        k_output,
        v_output,
        m_size,
        hidden_size,
        q_size,
        kv_size,
        flat_hidden.stride(0),
        flat_hidden.stride(1),
        module.q_proj.weight.stride(0),
        module.q_proj.weight.stride(1),
        module.k_proj.weight.stride(0),
        module.k_proj.weight.stride(1),
        module.v_proj.weight.stride(0),
        module.v_proj.weight.stride(1),
        q_output.stride(0),
        q_output.stride(1),
        k_output.stride(0),
        k_output.stride(1),
        v_output.stride(0),
        v_output.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return (
        q_output.reshape(*input_shape, q_size),
        k_output.reshape(*input_shape, kv_size),
        v_output.reshape(*input_shape, kv_size),
    )


def _prepare_fused_qkv(module: Any) -> bool:
    """Materialize an inference-only concatenated QKV weight on one attention module."""
    if getattr(module, "_qav_use_fused_qkv", False) and hasattr(
        module, "_qav_fused_qkv_weight"
    ):
        return True
    try:
        weights = (module.q_proj.weight, module.k_proj.weight, module.v_proj.weight)
        biases = (module.q_proj.bias, module.k_proj.bias, module.v_proj.bias)
    except AttributeError as error:
        raise PagedAttentionKernelError("Qwen attention module is missing QKV projections") from error
    if any(value is not None for value in biases) and not all(
        value is not None for value in biases
    ):
        return False
    module._qav_fused_qkv_weight = torch.cat(weights, dim=0).contiguous()
    module._qav_fused_qkv_bias = (
        torch.cat(biases, dim=0).contiguous() if biases[0] is not None else None
    )
    module._qav_use_fused_qkv = True
    return True


def _project_fused_qkv(module: Any, hidden_states: torch.Tensor):
    if getattr(module, "_qav_fused_projection_kernel", False):
        return fused_qkv_projection_kernel(module, hidden_states)
    return fused_qkv_projection(module, hidden_states)


def install_qwen_paged_decode_attention(
    talker: Any,
    *,
    fuse_qkv: bool = False,
    patch_static_cache: bool = False,
    fuse_static_attention: bool = False,
    fused_projection_kernel: bool = False,
) -> Any:
    """Patch Qwen self-attention decode calls to use the page-pool kernel.

    Prefill and non-Qwen caches continue through the original attention method.
    The returned handle restores every patched module and is idempotent per
    talker instance.
    """
    from types import MethodType

    model = getattr(talker, "model", None)
    layers = getattr(model, "layers", None)
    if layers is None:
        raise PagedAttentionKernelError("Qwen talker model does not expose decoder layers")
    existing = getattr(talker, "_qav_paged_attention_patch", None)
    if existing is not None:
        return existing

    patched = []

    def make_forward(original):
        def forward(
            module,
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            cache_type = type(past_key_values).__name__
            if (
                hidden_states.ndim == 3
                and hidden_states.shape[1] == 1
                and cache_type == "QwenPagedCache"
                and getattr(past_key_values, "use_paged_attention_kernel", False)
            ):
                return _forward_qwen_paged_attention(
                    module,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                )
            if (
                patch_static_cache
                and hidden_states.ndim == 3
                and hidden_states.shape[1] == 1
                and cache_type == "StaticCache"
                and (
                    getattr(module, "_qav_use_fused_qkv", False)
                    or getattr(module, "_qav_use_fused_attention", False)
                )
            ):
                return _forward_qwen_static_attention(
                    module,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_sdpa=fuse_static_attention,
                    **kwargs,
                )
            return original(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        return forward

    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise PagedAttentionKernelError("Qwen decoder layer does not expose self_attn")
        original = attention.forward
        fused_state = {}
        if fuse_qkv or (patch_static_cache and fuse_static_attention):
            for name in (
                "_qav_fused_qkv_weight",
                "_qav_fused_qkv_bias",
                "_qav_use_fused_qkv",
                "_qav_use_fused_attention",
                "_qav_fused_projection_kernel",
            ):
                fused_state[name] = (hasattr(attention, name), getattr(attention, name, None))
            if fuse_qkv:
                _prepare_fused_qkv(attention)
                if fused_projection_kernel:
                    attention._qav_fused_projection_kernel = True
            if patch_static_cache and fuse_static_attention:
                attention._qav_use_fused_attention = True
        attention.forward = MethodType(make_forward(original), attention)
        patched.append((attention, original, fused_state))

    class _PatchHandle:
        def __init__(self, owner, modules):
            self.owner = owner
            self.modules = modules
            self.closed = False

        def close(self):
            if self.closed:
                return
            for attention, original, fused_state in self.modules:
                attention.forward = original
                for name, (had_value, old_value) in fused_state.items():
                    if had_value:
                        setattr(attention, name, old_value)
                    elif hasattr(attention, name):
                        delattr(attention, name)
            self.closed = True
            if getattr(self.owner, "_qav_paged_attention_patch", None) is self:
                delattr(self.owner, "_qav_paged_attention_patch")

    handle = _PatchHandle(talker, patched)
    setattr(talker, "_qav_paged_attention_patch", handle)
    return handle


def _forward_qwen_paged_attention(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: Any,
    past_key_values: Any,
    cache_position: torch.Tensor | None,
) -> tuple[torch.Tensor, None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    if getattr(module, "_qav_use_fused_qkv", False):
        q_projected, k_projected, v_projected = _project_fused_qkv(module, hidden_states)
    else:
        q_projected = module.q_proj(hidden_states)
        k_projected = module.k_proj(hidden_states)
        v_projected = module.v_proj(hidden_states)
    query_states = module.q_norm(q_projected.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(k_projected.view(hidden_shape)).transpose(1, 2)
    value_states = v_projected.view(hidden_shape).transpose(1, 2)
    if position_embeddings is None:
        raise PagedAttentionKernelError("Qwen paged attention requires position embeddings")
    cos, sin = position_embeddings
    from qwen_tts.core.models.modeling_qwen3_tts import (
        apply_multimodal_rotary_pos_emb,
    )

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        module.rope_scaling["mrope_section"],
        module.rope_scaling["interleaved"],
    )
    del cache_position
    attn_output = past_key_values.paged_attention_decode(
        layer_idx=module.layer_idx,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
    )
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return module.o_proj(attn_output), None


def _forward_qwen_static_attention(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: Any,
    attention_mask: torch.Tensor | None,
    past_key_values: Any,
    cache_position: torch.Tensor | None,
    use_sdpa: bool,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """StaticCache decode with one QKV projection and fused SDPA when enabled."""

    import torch.nn.functional as functional

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    if getattr(module, "_qav_use_fused_qkv", False):
        q_projected, k_projected, v_projected = _project_fused_qkv(module, hidden_states)
    else:
        q_projected = module.q_proj(hidden_states)
        k_projected = module.k_proj(hidden_states)
        v_projected = module.v_proj(hidden_states)
    query_states = module.q_norm(q_projected.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(k_projected.view(hidden_shape)).transpose(1, 2)
    value_states = v_projected.view(hidden_shape).transpose(1, 2)
    if position_embeddings is None:
        raise PagedAttentionKernelError("Qwen static attention requires position embeddings")
    cos, sin = position_embeddings
    from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        module.rope_scaling["mrope_section"],
        module.rope_scaling["interleaved"],
    )
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, module.layer_idx, cache_kwargs
    )
    if use_sdpa:
        key_states = key_states.repeat_interleave(module.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(module.num_key_value_groups, dim=1)
        mask = (
            attention_mask[:, :, :, : key_states.shape[-2]]
            if attention_mask is not None
            else None
        )
        attn_output = functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )
    else:
        from qwen_tts.core.models.modeling_qwen3_tts import (
            ALL_ATTENTION_FUNCTIONS,
            eager_attention_forward,
        )

        attention_interface = eager_attention_forward
        if module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]
        attn_output, _ = attention_interface(
            module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not module.training else module.attention_dropout,
            scaling=module.scaling,
            sliding_window=module.sliding_window,
            **kwargs,
        )
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return module.o_proj(attn_output), None
