import asyncio

from bench.voice_agent_timing import TimingResult


def test_timing_collector_keeps_asr_final_after_early_done():
    from bench.voice_agent_timing import collect_until_input_complete

    async def scenario():
        events = asyncio.Queue()
        input_complete = asyncio.Event()
        observed = []

        async def next_event():
            return await events.get()

        async def run():
            await collect_until_input_complete(
                next_event,
                lambda event: observed.append(event["type"]),
                input_complete,
            )

        task = asyncio.create_task(run())
        await events.put({"type": "done"})
        await asyncio.sleep(0)
        await events.put({"type": "asr_final"})
        await asyncio.sleep(0)
        input_complete.set()
        await task
        return observed

    assert asyncio.run(scenario()) == ["done", "asr_final"]


def test_timing_collector_stops_waiting_when_input_completes_after_done():
    from bench.voice_agent_timing import collect_until_input_complete

    async def scenario():
        events = asyncio.Queue()
        input_complete = asyncio.Event()
        waiting_after_done = asyncio.Event()
        calls = 0

        async def next_event():
            nonlocal calls
            calls += 1
            if calls == 2:
                waiting_after_done.set()
            return await events.get()

        await events.put({"type": "done"})
        task = asyncio.create_task(
            collect_until_input_complete(next_event, lambda _event: None, input_complete)
        )
        await waiting_after_done.wait()
        input_complete.set()
        await task

    asyncio.run(scenario())


def test_timing_output_comparison_requires_text_and_pcm_match():
    from bench.voice_agent_timing import compare_timing_outputs

    base = TimingResult(
        "base",
        100.0,
        500.0,
        [],
        details={
            "asr_hypothesis": "same transcript",
            "llm_output": "same reply",
            "tts_rendered_inputs": ["same reply"],
            "tts_audio_bytes": 3200,
            "tts_audio_duration_ms": 100.0,
        },
    )
    matching = TimingResult(
        "candidate",
        50.0,
        250.0,
        [],
        details=dict(base.details),
    )
    shorter_audio = TimingResult(
        "candidate",
        50.0,
        200.0,
        [],
        details={**base.details, "tts_audio_bytes": 3000},
    )

    accepted = compare_timing_outputs(base, matching)
    rejected = compare_timing_outputs(base, shorter_audio)

    assert accepted == {
        "eligible": True,
        "mismatches": [],
        "speedup": {"first_audio": 2.0, "total": 2.0},
    }
    assert rejected["eligible"] is False
    assert rejected["mismatches"] == ["tts_audio_bytes"]
    assert rejected["speedup"] is None
