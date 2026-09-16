import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bench.qwen_tts_streaming_engine_probe import run_probe, run_streaming_probe


class FakeWrapper:
    def __init__(self):
        self._speaker = "alice"
        self._language = "english"
        self.model = FakeModel()

    def generate_custom_voice(self, **kwargs):
        codes = np.arange(8, dtype=np.int64).reshape(8, 1)
        audio = np.zeros(codes.shape[0] * 4, dtype=np.float32)
        return [audio], 16000

    def _ensure_list(self, value):
        return value if isinstance(value, list) else [value]

    def _validate_languages(self, languages):
        return None

    def _validate_speakers(self, speakers):
        return None

    def _build_assistant_text(self, text):
        return text

    def _tokenize_texts(self, texts):
        return [np.array([[1, 2, 3]], dtype=np.int64) for _ in texts]

    def _merge_generate_kwargs(self, **kwargs):
        return {k: v for k, v in kwargs.items() if v is not None}


class FakeModel:
    tts_model_type = "custom_voice"
    tts_model_size = "1b7"
    speech_tokenizer = None

    def __init__(self):
        self.speech_tokenizer = FakeSpeechTokenizer()
        self.talker = FakeTalker()
        self.config = SimpleNamespace(
            talker_config=SimpleNamespace(codec_eos_token_id=999)
        )

    def get_custom_voice_codes(self, **kwargs):
        return [np.arange(8, dtype=np.int64).reshape(8, 1)]

    def generate(self, **kwargs):
        for _ in range(5):
            self.talker.forward()
            time.sleep(0.01)
        return [np.arange(5, dtype=np.int64).reshape(5, 1)], None


class FakeTalker:
    def __init__(self):
        self.calls = 0
        self.generation_finished = False

    def forward(self):
        self.calls += 1
        frame = np.array([[self.calls - 1]], dtype=np.int64)
        if self.calls == 5:
            self.generation_finished = True
        return SimpleNamespace(hidden_states=(None, frame))


class FakeSpeechTokenizer:
    def get_decode_upsample_rate(self):
        return 4

    def decode(self, encoded):
        codes = np.asarray(encoded[0]["audio_codes"])
        audio = np.repeat(codes[:, 0].astype(np.float32) / 8, 4)
        return [audio], 16000


def test_run_probe_writes_stage_a_chunked_decode_report(tmp_path: Path):
    report = run_probe(
        FakeWrapper(),
        text="hello",
        chunk_size=3,
        left_context_size=1,
        out_dir=tmp_path,
    )

    assert report["stage"] == "codec_chunk_decode"
    assert report["text_chars"] == 5
    assert report["chunks"] == 3
    assert report["chunked_first_audio_ms"] is not None
    assert Path(report["chunked_wav"]).is_file()
    assert Path(report["full_wav"]).is_file()
    assert Path(report["report_json"]).is_file()
    assert report["mean_abs_diff_overlap"] == 0
    persisted = json.loads(Path(report["report_json"]).read_text())
    assert persisted["chunks"] == 3


def test_run_streaming_probe_writes_stage_b_report_before_generation_finishes(tmp_path: Path):
    wrapper = FakeWrapper()

    report = run_streaming_probe(
        wrapper,
        text="hello",
        chunk_size=2,
        first_chunk_size=1,
        left_context_size=1,
        out_dir=tmp_path,
    )

    assert report["stage"] == "codec_step_stream_decode"
    assert report["first_chunk_size"] == 1
    assert report["chunk_codec_frames"] == [1, 2, 2]
    assert report["stream_first_audio_ms"] < report["stream_total_ms"]
    assert Path(report["streaming_wav"]).is_file()
    assert Path(report["report_json"]).is_file()
