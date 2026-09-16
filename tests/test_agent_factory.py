from qwen_asr_vllm.agent.factory import build_voice_factory
from qwen_asr_vllm.agent.async_coordinator import AsyncVoiceAgentCoordinator
from qwen_asr_vllm.agent.service_runners import (
    MultiplexedThreadedLlmBackend,
    MultiplexedThreadedTtsBackend,
    ThreadedAsrSession,
)


class FakeEngine:
    def __init__(self):
        self.opened = []

    def open_stream(self, **kwargs):
        self.opened.append(kwargs)
        return FakeAsrSession()


class FakeAsrSession:
    def feed(self, pcm):
        return []

    def close(self, reuse_last=True):
        return []


class FakeLlm:
    def chat_stream(self, text):
        yield "ok"


class ResidentFakeLlm(FakeLlm):
    resident_runner = True


class FakeTts:
    def synthesize(self, text):
        return text.encode()


def test_build_voice_factory_creates_independent_configured_coordinators():
    engine = FakeEngine()
    factory = build_voice_factory(
        llm=FakeLlm(),
        tts=FakeTts(),
        llm_trigger="committed",
        asr_kwargs={"language": "en", "chunk_policy": "speculate"},
    )

    first = factory(engine)
    second = factory(engine)

    assert first is not second
    assert first.llm_trigger == "committed"
    assert second.llm_trigger == "committed"
    assert engine.opened == [
        {"language": "en", "chunk_policy": "speculate"},
        {"language": "en", "chunk_policy": "speculate"},
    ]


def test_build_voice_factory_can_create_async_coordinators():
    engine = FakeEngine()
    factory = build_voice_factory(
        llm=FakeLlm(),
        tts=FakeTts(),
        async_mode=True,
        llm_trigger="committed",
        asr_kwargs={"language": "en"},
    )

    coordinator = factory(engine)

    try:
        assert isinstance(coordinator, AsyncVoiceAgentCoordinator)
        assert coordinator.llm_trigger == "committed"
        assert isinstance(coordinator.session, ThreadedAsrSession)
        assert isinstance(coordinator.llm, MultiplexedThreadedLlmBackend)
        assert isinstance(coordinator.tts, MultiplexedThreadedTtsBackend)
        assert engine.opened == [{"language": "en"}]
    finally:
        coordinator.session.close()
        factory.close()


def test_build_voice_factory_passes_async_latency_tuning_options():
    engine = FakeEngine()
    factory = build_voice_factory(
        llm=ResidentFakeLlm(),
        tts=FakeTts(),
        async_mode=True,
        llm_trigger="committed",
        tts_flush_after_ms=250,
        tts_flush_min_chars=9,
        tts_defer_short_segments_chars=48,
        tts_defer_short_segments_ms=80,
        tts_stream_first_segment_only=True,
        min_committed_words=3,
        min_committed_audio_seconds=0.7,
        max_committed_audio_seconds=1.5,
        defer_tts_audio_until_asr_final=True,
    )

    coordinator = factory(engine)

    try:
        assert coordinator.tts_flush_after_seconds == 0.25
        assert coordinator.tts_flush_min_chars == 9
        assert coordinator.tts_defer_short_segments_chars == 48
        assert coordinator.tts_defer_short_segments_seconds == 0.08
        assert coordinator.tts_stream_first_segment_only is True
        assert coordinator.min_committed_words == 3
        assert coordinator.min_committed_audio_seconds == 0.7
        assert coordinator.max_committed_audio_seconds == 1.5
        assert coordinator.defer_tts_audio_until_asr_final is True
    finally:
        coordinator.session.close()
        factory.close()


def test_build_voice_factory_reuses_resident_async_runners():
    engine = FakeEngine()
    factory = build_voice_factory(
        llm=FakeLlm(),
        tts=FakeTts(),
        async_mode=True,
        llm_trigger="committed",
        asr_kwargs={"language": "en"},
    )

    first = factory(engine)
    second = factory(engine)

    try:
        assert isinstance(first, AsyncVoiceAgentCoordinator)
        assert isinstance(second, AsyncVoiceAgentCoordinator)
        assert isinstance(first.session, ThreadedAsrSession)
        assert isinstance(second.session, ThreadedAsrSession)
        assert first.llm is second.llm
        assert first.tts is second.tts
        assert engine.opened == [{"language": "en"}, {"language": "en"}]
    finally:
        first.session.close()
        second.session.close()
        factory.close()


def test_build_voice_factory_does_not_wrap_resident_llm_runner():
    engine = FakeEngine()
    llm = ResidentFakeLlm()
    factory = build_voice_factory(
        llm=llm,
        tts=FakeTts(),
        async_mode=True,
    )

    coordinator = factory(engine)

    try:
        assert coordinator.llm is llm
        assert isinstance(coordinator.tts, MultiplexedThreadedTtsBackend)
    finally:
        coordinator.session.close()
        factory.close()
