from types import SimpleNamespace

import torch

from qwen_asr_vllm.agent.qwen_tts_outer_engine import (
    explicit_talker_generate,
    install_explicit_talker_step_engine,
)


class FakeTalker:
    def __init__(self):
        self.calls = []
        self.config = SimpleNamespace(codec_eos_token_id=9)

    def generate(self, **kwargs):
        raise AssertionError("original talker.generate should not be called")

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = (
            kwargs.get("inputs_embeds")
            if kwargs.get("inputs_embeds") is not None
            else kwargs["input_ids"]
        ).shape[0]
        step = len(self.calls) - 1
        logits = torch.zeros(batch_size, 1, 16)
        logits[:, :, min(step + 1, 9)] = 10.0
        codec_ids = None
        if kwargs.get("input_ids") is not None:
            token = kwargs["input_ids"]
            codec_ids = torch.cat(
                [token, torch.full((batch_size, 2), step + 3, dtype=torch.long)],
                dim=-1,
            )
        return SimpleNamespace(
            logits=logits,
            past_key_values=f"past-{step}",
            past_hidden=torch.full((batch_size, 1, 4), float(step)),
            generation_step=step,
            trailing_text_hidden=kwargs["trailing_text_hidden"],
            tts_pad_embed=kwargs["tts_pad_embed"],
            hidden_states=(torch.full((batch_size, 1, 4), float(step)), codec_ids),
        )


def test_explicit_talker_generate_runs_prefill_and_decode_steps():
    talker = FakeTalker()

    result = explicit_talker_generate(
        talker,
        inputs_embeds=torch.zeros(2, 3, 4),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        trailing_text_hidden=torch.zeros(2, 4, 4),
        tts_pad_embed=torch.zeros(1, 1, 4),
        max_new_tokens=3,
        min_new_tokens=2,
        do_sample=False,
        eos_token_id=9,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    assert len(talker.calls) == 4
    assert "inputs_embeds" in talker.calls[0]
    assert talker.calls[1]["input_ids"].tolist() == [[1], [1]]
    assert talker.calls[2]["input_ids"].tolist() == [[2], [2]]
    assert len(result.hidden_states) == 4
    assert result.hidden_states[0][-1] is None
    assert result.hidden_states[1][-1].shape == (2, 3)


def test_install_explicit_talker_step_engine_replaces_generate_once():
    talker = FakeTalker()

    assert install_explicit_talker_step_engine(talker) is True
    assert install_explicit_talker_step_engine(talker) is False
    result = talker.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        trailing_text_hidden=torch.zeros(1, 2, 4),
        tts_pad_embed=torch.zeros(1, 1, 4),
        max_new_tokens=1,
        do_sample=False,
    )

    assert len(result.hidden_states) == 2


def test_explicit_talker_step_engine_can_use_compiled_forward_callable():
    talker = FakeTalker()
    compiled = []

    def compiler(fn, **kwargs):
        compiled.append((fn, kwargs))

        def wrapped(**call_kwargs):
            return fn(**call_kwargs)

        return wrapped

    install_explicit_talker_step_engine(
        talker,
        compile_step=True,
        compile_mode="reduce-overhead",
        compiler=compiler,
    )
    talker.generate(
        inputs_embeds=torch.zeros(1, 2, 4),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        trailing_text_hidden=torch.zeros(1, 2, 4),
        tts_pad_embed=torch.zeros(1, 1, 4),
        max_new_tokens=1,
        do_sample=False,
    )

    assert compiled == [(talker.forward, {"mode": "reduce-overhead"})]
