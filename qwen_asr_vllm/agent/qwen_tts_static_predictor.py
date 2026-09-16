from __future__ import annotations

import threading
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

from transformers.cache_utils import Cache, CacheLayerMixin

from qwen_asr_vllm.agent.qwen_tts_fast_predictor import (
    FastCodePredictorError,
    _sample_next_token,
)


class _PrefixStaticLayer(CacheLayerMixin):
    # Keep the same causal-mask elision decision as DynamicCache. The backing
    # tensors are static, but the attention call must still use SDPA's native
    # causal path instead of materializing a compile-time mask.
    is_compileable = False
    is_sliding = False

    def __init__(self, capacity: int):
        super().__init__()
        self.max_cache_len = int(capacity)
        self._view_length = 0

    def lazy_initialization(self, key_states):
        import torch

        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.zeros(
            (
                self.max_batch_size,
                self.num_heads,
                self.max_cache_len,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros_like(self.keys)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        import torch

        if not self.is_initialized:
            self.lazy_initialization(key_states)
        cache_position = (cache_kwargs or {}).get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(
                key_states.shape[-2], device=key_states.device
            )
        self.keys.index_copy_(2, cache_position, key_states)
        self.values.index_copy_(2, cache_position, value_states)
        return (
            self.keys[..., : self._view_length, :],
            self.values[..., : self._view_length, :],
        )

    def get_mask_sizes(self, cache_position):
        return self._view_length, 0

    def get_seq_length(self):
        return self._view_length

    def get_max_cache_shape(self):
        return self.max_cache_len

    def reset(self):
        super().reset()
        self._view_length = 0


class PrefixStaticCache(Cache):
    """Static-address KV storage that exposes only the active prefix.

    ``transformers.StaticCache`` returns its complete preallocated backing
    tensor from ``update``. Its causal mask hides future slots, but the extra
    masked positions still participate in the attention reduction and can
    change FP16 logits relative to Qwen-TTS's DynamicCache. This cache keeps
    the backing allocation fixed while returning a prefix view whose shape
    matches DynamicCache for each prepared step.
    """

    def __init__(self, *, config: Any, max_cache_len: int):
        config = config.get_text_config(decoder=True)
        layer_count = int(config.num_hidden_layers)
        super().__init__(
            layers=[
                _PrefixStaticLayer(int(max_cache_len))
                for _ in range(layer_count)
            ]
        )
        self._max_cache_len = int(max_cache_len)
        self._past_length = 0

    def prepare_step(self, *, past_length: int, query_length: int) -> None:
        past_length = int(past_length)
        query_length = int(query_length)
        view_length = past_length + query_length
        if past_length < 0 or query_length <= 0 or view_length > self._max_cache_len:
            raise FastCodePredictorError(
                "invalid prefix static cache step: "
                f"past_length={past_length}, query_length={query_length}, "
                f"max_cache_len={self._max_cache_len}"
            )
        self._past_length = past_length
        self._query_length = query_length
        for layer in self.layers:
            layer._view_length = view_length

    def get_mask_sizes(self, cache_position, layer_idx):
        return self._past_length + int(cache_position.shape[0]), 0

    def get_seq_length(self, layer_idx=0):
        return self._past_length

    def get_max_cache_shape(self, layer_idx=0):
        return self._max_cache_len

    def reset(self):
        super().reset()
        self._past_length = 0
        self._query_length = 0


class StaticCodePredictorEngine:
    def __init__(self, code_predictor: Any, *, cache_factory=None):
        self._code_predictor = code_predictor
        self._cache_factory = cache_factory or _default_cache_factory
        self._cache_pool: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
        self._pool_lock = threading.Lock()
        self.cache_creations = 0
        self.cache_hits = 0

    def generate(
        self,
        *,
        inputs_embeds,
        max_new_tokens: int,
        do_sample: bool | None = True,
        top_p: float | None = 1.0,
        top_k: int | None = 50,
        temperature: float | None = 0.9,
        output_hidden_states: bool | None = None,
        return_dict_in_generate: bool | None = True,
        **kwargs,
    ):
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise FastCodePredictorError(
                f"unsupported static code predictor kwargs: {unsupported}"
            )
        if inputs_embeds is None:
            raise FastCodePredictorError("inputs_embeds is required")
        if max_new_tokens <= 0:
            raise FastCodePredictorError("max_new_tokens must be positive")

        import torch

        prompt_length = int(inputs_embeds.shape[1])
        max_cache_len = prompt_length + int(max_new_tokens) - 1
        cache_key = (
            int(inputs_embeds.shape[0]),
            str(inputs_embeds.device),
            str(inputs_embeds.dtype),
            max_cache_len,
        )
        cache = self._acquire_cache(cache_key, max_cache_len=max_cache_len)
        sequences = []
        generation_steps = None
        next_token = None

        try:
            with torch.inference_mode():
                cache.reset()
                for step_index in range(int(max_new_tokens)):
                    prepare_step = getattr(cache, "prepare_step", None)
                    if callable(prepare_step):
                        prepare_step(
                            past_length=prompt_length + max(0, step_index - 1),
                            query_length=prompt_length if step_index == 0 else 1,
                        )
                    if step_index == 0:
                        outputs = self._code_predictor(
                            inputs_embeds=inputs_embeds,
                            past_key_values=cache,
                            cache_position=torch.arange(
                                prompt_length,
                                device=inputs_embeds.device,
                            ),
                            use_cache=True,
                            output_hidden_states=bool(output_hidden_states),
                        )
                    else:
                        outputs = self._code_predictor(
                            input_ids=next_token,
                            past_key_values=cache,
                            cache_position=torch.tensor(
                                [prompt_length + step_index - 1],
                                device=inputs_embeds.device,
                            ),
                            use_cache=True,
                            output_hidden_states=bool(output_hidden_states),
                            generation_steps=generation_steps,
                        )

                    logits = outputs.logits[:, -1, :]
                    next_token = _sample_next_token(
                        logits,
                        do_sample=bool(do_sample),
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                    )
                    sequences.append(next_token)
                    generation_steps = outputs.generation_steps
        finally:
            self._release_cache(cache_key, cache)

        generated = torch.cat(sequences, dim=-1)
        if return_dict_in_generate:
            return SimpleNamespace(sequences=generated)
        return generated

    def _acquire_cache(self, key: tuple[Any, ...], *, max_cache_len: int):
        with self._pool_lock:
            if self._cache_pool[key]:
                self.cache_hits += 1
                return self._cache_pool[key].pop()
            self.cache_creations += 1
        return self._cache_factory(
            code_predictor=self._code_predictor,
            max_cache_len=max_cache_len,
        )

    def _release_cache(self, key: tuple[Any, ...], cache: Any) -> None:
        with self._pool_lock:
            self._cache_pool[key].append(cache)


def install_static_code_predictor(talker: Any, *, cache_factory=None) -> bool:
    code_predictor = getattr(talker, "code_predictor", None)
    if code_predictor is None:
        raise FastCodePredictorError("talker has no code_predictor")
    if getattr(code_predictor.generate, "_qav_static_code_predictor", False):
        return False

    original_generate = code_predictor.generate
    engine = StaticCodePredictorEngine(
        code_predictor,
        cache_factory=cache_factory,
    )

    def static_generate(**kwargs):
        return engine.generate(**kwargs)

    static_generate._qav_fast_code_predictor = True  # type: ignore[attr-defined]
    static_generate._qav_static_code_predictor = True  # type: ignore[attr-defined]
    static_generate._qav_original_generate = original_generate  # type: ignore[attr-defined]
    static_generate._qav_engine = engine  # type: ignore[attr-defined]
    code_predictor.generate = static_generate
    return True


def _default_cache_factory(*, code_predictor: Any, max_cache_len: int):
    return PrefixStaticCache(
        config=code_predictor.config,
        max_cache_len=max_cache_len,
    )
