"""Paged KV cache allocation with audio-aware prefix caching.

Prefix caching keys blocks by their token ids, which breaks down for audio: every
position in the audio span carries the same ``<|audio_pad|>`` id, so two different
recordings hash identically while holding completely different KV state. Because
prefix hashes chain, a block containing audio poisons every block after it too.

Rather than disabling prefix caching outright, each request reports how many
leading blocks are purely textual and only those participate. That keeps reuse
available for the case where it actually pays: a long shared system prompt used to
bias the model toward domain vocabulary.
"""

from __future__ import annotations

import numpy as np
import xxhash

from qwen_asr_vllm.engine.request import AsrRequest

NO_HASH = -1


class Block:
    __slots__ = ("block_id", "ref_count", "hash", "token_ids")

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = NO_HASH
        self.token_ids: list[int] = []

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = NO_HASH
        self.token_ids = []

    def update(self, block_hash: int, token_ids: list[int]) -> None:
        self.hash = block_hash
        self.token_ids = token_ids


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_cache: bool = True):
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        # Insertion-ordered dict rather than a deque: allocation pops the oldest
        # free block in O(1), and a prefix-cache hit on a currently-free block
        # also removes it in O(1).
        self.free_block_ids: dict[int, None] = dict.fromkeys(range(num_blocks))
        self.num_cache_hit_blocks = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = NO_HASH) -> int:
        digest = xxhash.xxh64()
        if prefix != NO_HASH:
            # intdigest() is unsigned 64-bit, so the conversion must be too.
            digest.update(prefix.to_bytes(8, "little"))
        digest.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return digest.intdigest()

    def _claim(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"block {block_id} is still referenced"
        block.reset()
        del self.free_block_ids[block_id]
        return block

    def _claim_any(self) -> Block:
        return self._claim(next(iter(self.free_block_ids)))

    def _release(self, block_id: int) -> None:
        self.free_block_ids[block_id] = None

    def can_allocate(self, request: AsrRequest) -> bool:
        return self.num_free_blocks >= request.num_blocks

    def can_ever_allocate(self, request: AsrRequest) -> bool:
        """Would this request fit if the cache were completely empty?

        Distinguishes transient pressure, which waiting resolves, from a request
        larger than the cache, which waiting never resolves.
        """
        return len(self.blocks) >= request.num_blocks

    def allocate(self, request: AsrRequest) -> None:
        """Bind physical blocks to a request, reusing cached textual prefix blocks."""
        assert not request.block_table
        cacheable_blocks = request.num_cacheable_blocks if self.enable_prefix_cache else 0

        prefix_hash = NO_HASH
        cache_miss = False
        for index in range(request.num_blocks):
            token_ids = request.block(index)
            in_cacheable_prefix = index < cacheable_blocks and len(token_ids) == self.block_size
            block_hash = self.compute_hash(token_ids, prefix_hash) if in_cacheable_prefix else NO_HASH

            block_id = self.hash_to_block_id.get(block_hash, -1) if block_hash != NO_HASH else -1
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                block = self._claim_any()
            else:
                request.num_cached_tokens += self.block_size
                self.num_cache_hit_blocks += 1
                block = self.blocks[block_id]
                if block.ref_count:
                    block.ref_count += 1
                else:
                    block = self._claim(block_id)

            if block_hash != NO_HASH:
                block.update(block_hash, token_ids)
                self.hash_to_block_id[block_hash] = block.block_id
                prefix_hash = block_hash

            request.block_table.append(block.block_id)

        request.num_computed_tokens = request.num_cached_tokens

    def deallocate(self, request: AsrRequest) -> None:
        for block_id in reversed(request.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._release(block_id)
        request.block_table.clear()
        request.num_cached_tokens = 0

    def can_append(self, request: AsrRequest) -> bool:
        """A new block is only needed when the next token opens one."""
        needs_block = len(request) % self.block_size == 1
        return self.num_free_blocks >= int(needs_block)

    def may_append(self, request: AsrRequest) -> None:
        """Extend the block table if the freshly appended token started a new block.

        Decode positions always sit after the audio span, so blocks filled during
        decoding are never cacheable and never need a hash.
        """
        if len(request) % self.block_size == 1:
            request.block_table.append(self._claim_any().block_id)
