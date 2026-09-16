"""Text decoder stage.

Assembles one flattened forward pass for a batch that may mix prefill and decode.
Each request contributes ``num_tokens_to_compute`` query tokens -- its whole
uncomputed prompt when prefilling, exactly one when decoding -- and declares its
full context length through ``cu_seqlens_k``. FlashAttention's bottom-right causal
alignment then gives each request the right mask without any per-kind branching.
"""

from __future__ import annotations

import torch

from qwen_asr_vllm.audio.batcher import pack_audio_batch
from qwen_asr_vllm.config import EngineConfig
from qwen_asr_vllm.engine.graph_runner import GraphRunner
from qwen_asr_vllm.engine.request import AsrRequest
from qwen_asr_vllm.engine.scheduler import ModelBatch
from qwen_asr_vllm.layers.attention import PagedAttention
from qwen_asr_vllm.layers.context import AttentionContext, attention_context
from qwen_asr_vllm.layers.sampler import Sampler
from qwen_asr_vllm.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from qwen_asr_vllm.profiling import NULL_TIMER


class ModelRunner:
    def __init__(self, model: Qwen3ASRForConditionalGeneration, config: EngineConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.block_size = config.kvcache_block_size
        self.sampler = Sampler()
        self.attention_layers = [
            module for module in model.modules() if isinstance(module, PagedAttention)
        ]
        self.kv_cache: torch.Tensor | None = None
        # Set by capture_graphs, once the KV cache the graphs must reference exists.
        self.graph_runner: GraphRunner | None = None
        # Replaced by a profiler to separate prefill from decode device time.
        self.timer = NULL_TIMER

    def capture_graphs(self) -> None:
        """Capture decode graphs. Must run after the KV cache is allocated.

        A graph bakes in the addresses it touches, and the attention kernels read the
        KV cache directly, so capturing before allocation would freeze the wrong
        pointers.
        """
        if self.config.enforce_eager:
            return
        if self.kv_cache is None:
            raise RuntimeError("allocate the KV cache before capturing graphs")
        runner = GraphRunner(self.model, self.config)
        runner.capture()
        self.graph_runner = runner

    # ------------------------------------------------------------------ cache

    def kv_cache_bytes_per_block(self) -> int:
        text = self.config.model_config.text
        head_dim = text.head_dim or text.hidden_size // text.num_attention_heads
        elements = (
            2
            * text.num_hidden_layers
            * self.block_size
            * text.num_key_value_heads
            * head_dim
        )
        return elements * torch.tensor([], dtype=self.config.dtype).element_size()

    def _graph_reserve(self) -> int:
        """Memory held back for the CUDA graph pool.

        Graphs are captured after the KV cache is allocated, so their activations and
        flash-attention workspaces have to come out of the budget in advance or
        capture will run out of memory. Decode activations are small -- one token per
        sequence, and every bucket shares one pool -- so a flat reserve is enough.
        """
        if self.config.enforce_eager:
            return 0
        return 384 * 2**20

    def determine_num_blocks(self) -> int:
        """Size the KV cache from free memory and measured peak activations.

        ``gpu_memory_utilization`` is a fraction of memory *still free* after the
        weights are loaded, not of the card's total capacity. On a shared GPU a
        fraction-of-total budget is unusable: the setting's meaning would depend on
        what other tenants happen to be holding.
        """
        if self.config.num_kvcache_blocks > 0:
            return self.config.num_kvcache_blocks

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self._profile_run()
        torch.cuda.synchronize()

        stats = torch.cuda.memory_stats()
        peak_activation = stats["allocated_bytes.all.peak"] - stats["allocated_bytes.all.current"]
        # Hand the profiling activations back so free memory is not under-reported.
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()

        budget = free * self.config.gpu_memory_utilization - peak_activation - self._graph_reserve()
        num_blocks = int(budget) // self.kv_cache_bytes_per_block()
        if num_blocks <= 0:
            raise RuntimeError(
                f"no memory left for the KV cache: {free / 2**30:.1f}GiB free of "
                f"{total / 2**30:.1f}GiB, {peak_activation / 2**30:.1f}GiB needed for "
                f"activations at max_num_batched_tokens={self.config.max_num_batched_tokens}. "
                "Lower max_num_batched_tokens or free GPU memory."
            )
        return num_blocks

    def allocate_kv_cache(self, num_blocks: int) -> None:
        text = self.config.model_config.text
        head_dim = text.head_dim or text.hidden_size // text.num_attention_heads
        self.kv_cache = torch.zeros(
            2,
            text.num_hidden_layers,
            num_blocks,
            self.block_size,
            text.num_key_value_heads,
            head_dim,
            dtype=self.config.dtype,
            device=self.device,
        )
        for index, layer in enumerate(self.attention_layers):
            layer.k_cache = self.kv_cache[0, index]
            layer.v_cache = self.kv_cache[1, index]

    @torch.inference_mode()
    def _profile_audio_run(self) -> None:
        """Encode the widest audio batch the scheduler can build.

        The audio tower's activations are far from negligible: the convolutional
        downsampler widens 128 mel bins to ``downsample_hidden_size`` channels before
        the strides shrink them, so one ``conv_chunksize`` slice can hold well over a
        gigabyte. Leaving that out of the budget does not cause an OOM -- the KV cache
        just takes the memory first and the encoder then runs in whatever is left,
        making the allocator churn. That showed up as a 50x slowdown in the
        downsampler at high ``gpu_memory_utilization``, with no error to explain it.
        """
        audio = self.config.model_config.audio
        # The scheduler's batch cap. A single request may legally exceed it -- it is
        # admitted alone rather than rejected -- so this is the common worst case,
        # not an absolute one.
        frames = max(self.config.max_audio_batch_frames, audio.chunk_frames)
        mel = torch.zeros(
            audio.num_mel_bins, frames, dtype=self.config.dtype, device=self.device
        )
        batch = pack_audio_batch([mel], audio.n_window_infer, device=self.device)
        self.model.encode_audio(batch)

    @torch.inference_mode()
    def _profile_run(self) -> None:
        """Forward the widest batch the scheduler can build, without a KV cache."""
        self._profile_audio_run()
        max_len = self.config.max_model_len
        num_sequences = max(1, self.config.max_num_batched_tokens // max_len)
        num_tokens = num_sequences * max_len

        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)
        positions = torch.arange(max_len, device=self.device).repeat(num_sequences)
        cu_seqlens = torch.arange(
            0, num_tokens + 1, max_len, dtype=torch.int32, device=self.device
        )
        context = AttentionContext(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            slot_mapping=None,
            block_tables=None,
        )
        with attention_context(context):
            hidden = self.model(input_ids, positions)
            self.model.compute_logits(hidden[cu_seqlens[1:].long() - 1])

    # ------------------------------------------------------------------- step

    def _prepare(self, batch: ModelBatch):
        requests = batch.requests
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        audio_spans: list[tuple[int, torch.Tensor]] = []
        block_tables: list[list[int]] = []

        for request in requests:
            start = request.num_computed_tokens
            end = len(request)
            query_len = end - start

            input_ids.extend(request.token_ids[start:end])
            positions.extend(range(start, end))
            for position in range(start, end):
                block_id = request.block_table[position // self.block_size]
                slot_mapping.append(block_id * self.block_size + position % self.block_size)

            span = self._audio_span(request, start, end, cu_seqlens_q[-1])
            if span is not None:
                audio_spans.append(span)

            cu_seqlens_q.append(cu_seqlens_q[-1] + query_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, query_len)
            max_seqlen_k = max(max_seqlen_k, end)
            block_tables.append(request.block_table)

        width = max(len(table) for table in block_tables)
        padded_tables = [table + [-1] * (width - len(table)) for table in block_tables]

        as_int32 = {"dtype": torch.int32, "device": self.device, "pin_memory": False}
        context = AttentionContext(
            cu_seqlens_q=torch.tensor(cu_seqlens_q, **as_int32),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, **as_int32),
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=torch.tensor(slot_mapping, **as_int32),
            block_tables=torch.tensor(padded_tables, **as_int32),
        )
        return (
            torch.tensor(input_ids, dtype=torch.int64, device=self.device),
            torch.tensor(positions, dtype=torch.int64, device=self.device),
            context,
            audio_spans,
        )

    @staticmethod
    def _audio_span(
        request: AsrRequest, start: int, end: int, batch_offset: int
    ) -> tuple[int, torch.Tensor] | None:
        """Locate this request's audio placeholders inside the flattened batch."""
        if request.audio_embeds is None:
            return None
        audio_start = request.layout.audio_offset
        audio_end = audio_start + request.layout.audio_length
        low = max(audio_start, start)
        high = min(audio_end, end)
        if low >= high:
            return None
        return (
            batch_offset + (low - start),
            request.audio_embeds[low - audio_start : high - audio_start],
        )

    def can_use_graph(self, batch: ModelBatch) -> bool:
        """Graphs cover decode-only batches that fit a captured bucket."""
        return (
            self.graph_runner is not None
            and not batch.prefill
            and bool(batch.decode)
            and self.graph_runner.bucket_for(len(batch.decode)) is not None
        )

    @staticmethod
    def _phase_name(batch: ModelBatch) -> str:
        """What to attribute this batch's device time to.

        A mixed batch is its own category rather than being split between the two:
        the tokens share one kernel launch, so there is no honest way to divide the
        measurement, and pretending otherwise would flatter whichever phase we
        assigned the shared cost to.
        """
        if batch.prefill and batch.decode:
            return "llm_mixed"
        return "llm_prefill" if batch.prefill else "llm_decode"

    @torch.inference_mode()
    def run(self, batch: ModelBatch) -> list[int]:
        with self.timer.phase(self._phase_name(batch)):
            if self.can_use_graph(batch):
                last_hidden = self.graph_runner.run(batch.decode)
                logits = self.model.compute_logits(last_hidden)
            else:
                logits = self._forward_eager(batch)
        return self._sample(logits, batch.requests)

    def _forward_hidden(self, batch: ModelBatch):
        input_ids, positions, context, audio_spans = self._prepare(batch)

        with attention_context(context):
            inputs_embeds = self.model.embed_tokens(input_ids)
            if audio_spans:
                self.model.scatter_audio_embeddings(inputs_embeds, audio_spans)
            return self.model(None, positions, inputs_embeds=inputs_embeds), context

    def _forward_eager(self, batch: ModelBatch) -> torch.Tensor:
        hidden_states, context = self._forward_hidden(batch)
        last_token_indices = context.cu_seqlens_q[1:].long() - 1
        return self.model.compute_logits(hidden_states[last_token_indices])

    @torch.inference_mode()
    def verify(self, batch: ModelBatch, draft_lens: list[int]) -> list[list[int]]:
        """Greedy predictions at the tail of each query span, for speculative decoding.

        A request whose ``token_ids`` end in ``n`` draft tokens gets ``n + 1``
        predictions back: one per draft token, plus the token that follows the whole
        draft, which the same pass produces for free. Prediction ``j`` is what the model
        would have emitted in place of draft token ``j``, so the accepted prefix is the
        leading run where the two agree.

        Only the tail is projected through the vocabulary. Doing it for every position
        would be correct and unusable: a 1299s recording queries 16885 positions, and
        logits for all of them in bf16 come to 5.1 GiB.

        Greedy only. Non-zero temperature would need the draft distribution as well to
        stay unbiased, and ASR runs at temperature 0.
        """
        hidden_states, context = self._forward_hidden(batch)

        ends = context.cu_seqlens_q[1:].long().tolist()
        starts = context.cu_seqlens_q[:-1].long().tolist()
        index: list[int] = []
        widths: list[int] = []
        for start, end, num_draft in zip(starts, ends, draft_lens):
            width = min(num_draft + 1, end - start)
            index.extend(range(end - width, end))
            widths.append(width)

        gathered = hidden_states[torch.tensor(index, dtype=torch.int64, device=self.device)]
        predictions = self.model.compute_logits(gathered).argmax(dim=-1).tolist()

        out: list[list[int]] = []
        cursor = 0
        for width in widths:
            out.append(predictions[cursor : cursor + width])
            cursor += width
        return out

    def _sample(self, logits: torch.Tensor, requests) -> list[int]:
        temperatures = torch.tensor(
            [request.sampling.temperature for request in requests],
            dtype=torch.float32,
            device=self.device,
        )
        return self.sampler(logits, temperatures).tolist()
