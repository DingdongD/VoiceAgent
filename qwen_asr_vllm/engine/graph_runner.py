"""CUDA graph capture for the decode path.

A decode step is a few hundred tiny kernels, and on this model the CPU cannot issue
them fast enough to keep the GPU fed: profiling the 0.6B decoder at batch 4 measured
37ms of CPU dispatch per step against 7.9ms of GPU time. Replaying a captured graph
issues the entire step with one launch, so the step collapses toward its GPU cost.

Only decode is captured. Prefill query lengths vary continuously, which would need a
graph per length, and prefill is large enough per kernel that dispatch overhead is
already amortised -- the imbalance is specific to decode's one-token steps.

Capture requires every tensor address to be fixed, so the runner owns static buffers
and copies each step's metadata into them before replaying. Batch size is the one
thing that legitimately varies, handled by capturing a graph per size bucket and
padding up to the next one; padded rows carry a slot mapping of -1, which the KV
write kernel skips, and their outputs are discarded.
"""
from __future__ import annotations

import logging

import torch

from qwen_asr_vllm.config import EngineConfig
from qwen_asr_vllm.layers.context import AttentionContext, attention_context
from qwen_asr_vllm.models.qwen3_asr import Qwen3ASRForConditionalGeneration

logger = logging.getLogger(__name__)

# Bucket sizes to capture. Dense at the low end because a lightly loaded server
# spends most of its time there, and each graph costs memory and capture time.
DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64)


class GraphRunner:
    """Captures and replays decode-only forward passes."""

    def __init__(
        self,
        model: Qwen3ASRForConditionalGeneration,
        config: EngineConfig,
        buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.block_size = config.kvcache_block_size
        # A decode step's key length is whatever is already cached, up to the model
        # limit, so the block table has to be wide enough for the longest sequence.
        self.max_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.buckets = tuple(size for size in buckets if size <= config.max_num_seqs)
        if not self.buckets:
            self.buckets = (config.max_num_seqs,)
        self.max_batch_size = self.buckets[-1]

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self.pool = None
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        size = self.max_batch_size
        as_int32 = {"dtype": torch.int32, "device": self.device}
        self.input_ids = torch.zeros(size, dtype=torch.int64, device=self.device)
        self.positions = torch.zeros(size, dtype=torch.int64, device=self.device)
        self.slot_mapping = torch.full((size,), -1, **as_int32)
        self.cu_seqlens_q = torch.zeros(size + 1, **as_int32)
        self.cu_seqlens_k = torch.zeros(size + 1, **as_int32)
        self.block_tables = torch.zeros(size, self.max_blocks, **as_int32)

        # Staging mirrors in pinned memory. Copying from ordinary host memory is
        # synchronous, which would hand back much of the CPU time the graph saves;
        # from pinned memory the transfers are async and overlap with the replay.
        pinned = {"pin_memory": torch.cuda.is_available()}
        self.host_input_ids = torch.zeros(size, dtype=torch.int64, **pinned)
        self.host_positions = torch.zeros(size, dtype=torch.int64, **pinned)
        self.host_slot_mapping = torch.full((size,), -1, dtype=torch.int32, **pinned)
        self.host_cu_seqlens_k = torch.zeros(size + 1, dtype=torch.int32, **pinned)
        self.host_block_tables = torch.zeros(size, self.max_blocks, dtype=torch.int32, **pinned)

    def _context(self, batch_size: int) -> AttentionContext:
        return AttentionContext(
            cu_seqlens_q=self.cu_seqlens_q[: batch_size + 1],
            cu_seqlens_k=self.cu_seqlens_k[: batch_size + 1],
            # Decode always contributes exactly one query token, so this is exact
            # rather than an upper bound baked in at capture time.
            max_seqlen_q=1,
            # Baked in at the model limit: the real key length travels in
            # cu_seqlens_k, which the kernel reads per sequence, while max_seqlen_k
            # only sizes the kernel's work split.
            max_seqlen_k=self.config.max_model_len,
            slot_mapping=self.slot_mapping[:batch_size],
            block_tables=self.block_tables[:batch_size],
        )

    @torch.inference_mode()
    def capture(self) -> None:
        """Warm up and capture one graph per bucket.

        Captured largest first so every graph shares a memory pool sized once, rather
        than each capture growing it.
        """
        # Plausible decode state for the warmup: one token per sequence, a context
        # long enough to span more than a single block.
        self.cu_seqlens_q.copy_(
            torch.arange(self.max_batch_size + 1, dtype=torch.int32, device=self.device)
        )
        context_len = min(self.block_size + 1, self.config.max_model_len)
        self.cu_seqlens_k.copy_(
            torch.arange(self.max_batch_size + 1, dtype=torch.int32, device=self.device)
            * context_len
        )
        self.slot_mapping.fill_(-1)
        self.block_tables.zero_()

        for batch_size in reversed(self.buckets):
            context = self._context(batch_size)
            input_ids = self.input_ids[:batch_size]
            positions = self.positions[:batch_size]

            with attention_context(context):
                # Warm up outside the capture: the first call at a new shape compiles
                # Triton kernels and lets torch.compile specialise, neither of which
                # may happen while capturing.
                for _ in range(2):
                    self.model(input_ids, positions)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.pool):
                    output = self.model(input_ids, positions)

            self.pool = self.pool or graph.pool()
            self.graphs[batch_size] = graph
            self.outputs[batch_size] = output

        torch.cuda.synchronize()
        logger.info(
            "captured decode CUDA graphs for batch sizes %s", sorted(self.graphs)
        )

    def bucket_for(self, batch_size: int) -> int | None:
        """Smallest captured bucket that fits, or None if the batch is too large."""
        for size in self.buckets:
            if size >= batch_size:
                return size
        return None

    @torch.inference_mode()
    def run(self, requests) -> torch.Tensor:
        """Replay a decode step, returning last-token hidden states for `requests`."""
        batch_size = len(requests)
        bucket = self.bucket_for(batch_size)
        if bucket is None:
            raise ValueError(f"no captured graph for a batch of {batch_size}")

        # Padding rows get a slot mapping of -1, which the KV write kernel skips, and
        # a key length of one so their attention reads stay inside block 0.
        self.host_slot_mapping[:bucket].fill_(-1)
        self.host_block_tables[:bucket].zero_()

        running_keys = 0
        for index, request in enumerate(requests):
            position = len(request) - 1
            block_id = request.block_table[position // self.block_size]
            self.host_input_ids[index] = request.last_token_id
            self.host_positions[index] = position
            self.host_slot_mapping[index] = (
                block_id * self.block_size + position % self.block_size
            )
            table = request.block_table
            self.host_block_tables[index, : len(table)] = torch.tensor(
                table, dtype=torch.int32
            )
            running_keys += len(request)
            self.host_cu_seqlens_k[index + 1] = running_keys

        for index in range(batch_size, bucket):
            running_keys += 1
            self.host_cu_seqlens_k[index + 1] = running_keys

        self.input_ids[:bucket].copy_(self.host_input_ids[:bucket], non_blocking=True)
        self.positions[:bucket].copy_(self.host_positions[:bucket], non_blocking=True)
        self.slot_mapping[:bucket].copy_(self.host_slot_mapping[:bucket], non_blocking=True)
        self.cu_seqlens_k[: bucket + 1].copy_(
            self.host_cu_seqlens_k[: bucket + 1], non_blocking=True
        )
        self.block_tables[:bucket].copy_(self.host_block_tables[:bucket], non_blocking=True)

        self.graphs[bucket].replay()
        return self.outputs[bucket][:batch_size]
