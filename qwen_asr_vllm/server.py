"""HTTP server, OpenAI-compatible where it can be.

``POST /v1/audio/transcriptions`` mirrors OpenAI's endpoint, so existing clients and
SDKs work unchanged; the fields that have no OpenAI equivalent (``context`` for
hotword biasing) are additive. Two extras this model needs and that endpoint has no
place for: the detected language is returned in the JSON, and ``timeout`` is per
request rather than only a client-side socket deadline, because a queued request's
wait is what actually needs bounding under load.

Requests are handed to ``AsyncAsrEngine``, so a slow transcription never blocks the
event loop and the engine still sees a single mutating thread.

Note the absence of ``from __future__ import annotations``: the route handlers are
defined inside ``create_app``, and stringized annotations cannot be resolved from a
function scope, so FastAPI would fail to build their parameter models.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from qwen_asr_vllm.agent import latency_data
from qwen_asr_vllm.audio.decode import UnsupportedAudio, decode_audio
from qwen_asr_vllm.agent.latency_ui import AGENT_LATENCY_HTML
from qwen_asr_vllm.engine.async_engine import AsyncAsrEngine, RequestCancelled
from qwen_asr_vllm.engine.request import SamplingParams

logger = logging.getLogger(__name__)

# OpenAI's endpoint takes response_format; only the two that make sense without
# word-level timestamps are supported, and the model does not produce those.
SUPPORTED_FORMATS = ("json", "text", "verbose_json")


def _split_agent_event(event):
    body = event.as_dict()
    audio = body.pop("audio", None)
    return body, audio


def create_app(engine: AsyncAsrEngine, voice_factory=None):
    """Build the FastAPI app around an already-started engine."""
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
    from fastapi import WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

    @asynccontextmanager
    async def lifespan(_app):
        yield
        engine.close()

    app = FastAPI(title="qwen-asr-vllm", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine

    @app.exception_handler(UnsupportedAudio)
    async def _unsupported_audio(request: Request, exc: UnsupportedAudio):
        return JSONResponse(status_code=400, content={"error": {"message": str(exc)}})

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
        temperature: Annotated[float, Form()] = 0.0,
        context: Annotated[str, Form()] = "",
        timeout: Annotated[float | None, Form()] = None,
        max_new_tokens: Annotated[int, Form()] = 0,
    ):
        if response_format not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"response_format must be one of {SUPPORTED_FORMATS}",
            )

        waveform = decode_audio(await file.read())
        handle = engine.submit(
            waveform,
            context=context,
            language=language,
            # Scaled to the upload's length: the flat default truncates long audio, and
            # an HTTP caller has no way to notice that happened.
            sampling=SamplingParams.for_audio(
                len(waveform) / 16000, max_new_tokens, temperature=temperature
            ),
            timeout=timeout,
        )

        try:
            # The engine loop lives on its own thread, so awaiting here parks this
            # coroutine without holding up the event loop or any other request.
            output = await asyncio.wrap_future(handle.future)
        except RequestCancelled as exc:
            # 504 rather than 499: the deadline was ours to enforce, not the client's.
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if response_format == "text":
            return PlainTextResponse(output.text + "\n")
        body = {"text": output.text, "language": output.language}
        if response_format == "verbose_json":
            body.update(
                task="transcribe",
                duration=output.audio_seconds,
                usage={
                    "type": "tokens",
                    "input_tokens": output.num_prompt_tokens,
                    "output_tokens": output.num_output_tokens,
                },
                finish_reason=output.finish_reason,
                timings={
                    "queue_seconds": output.timings.queue_seconds,
                    "encode_seconds": output.timings.encode_seconds,
                    "prefill_seconds": output.timings.prefill_seconds,
                    "decode_seconds": output.timings.decode_seconds,
                    "total_seconds": output.timings.total_seconds,
                },
            )
        return body

    @app.get("/health")
    async def health():
        report = engine.health()
        status_code = 200 if report["status"] == "ok" else 503
        return JSONResponse(status_code=status_code, content=report)

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            engine.metrics.render_prometheus(
                scheduler_stats=engine.engine.scheduler.stats,
                extra={
                    "kv_cache_blocks": engine.engine.num_kvcache_blocks,
                    "kv_cache_blocks_free": engine.engine.block_manager.num_free_blocks,
                    "requests_in_flight": engine.num_in_flight,
                },
            ),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": engine.config.model, "object": "model"}]}

    if voice_factory is not None:

        async def close_voice_factory():
            close = getattr(voice_factory, "close", None)
            if callable(close):
                close()

        app.router.on_shutdown.append(close_voice_factory)

        @app.get("/agent-latency")
        async def agent_latency():
            return HTMLResponse(AGENT_LATENCY_HTML)

        @app.get("/agent-latency/datasets")
        async def agent_latency_datasets(split: str = "test-clean", limit: int = 8):
            bounded_limit = max(1, min(int(limit), 32))
            try:
                samples = latency_data.load_librispeech_samples(split=split, limit=bounded_limit)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return {
                "split": split,
                "samples": [
                    {
                        "index": index,
                        "sample_id": sample.sample_id,
                        "duration_seconds": round(float(sample.duration), 3),
                        "golden_asr_text": sample.text,
                    }
                    for index, sample in enumerate(samples)
                ],
            }

        @app.get("/agent-latency/datasets/{index}/audio")
        async def agent_latency_dataset_audio(
            index: int,
            split: str = "test-clean",
            limit: int = 8,
        ):
            bounded_limit = max(1, min(int(limit), 32))
            try:
                audio = latency_data.get_librispeech_sample_wav(
                    index,
                    split=split,
                    limit=bounded_limit,
                )
            except IndexError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return Response(audio, media_type="audio/wav")

        @app.websocket("/v1/voice/sessions")
        async def voice_session(ws: WebSocket):
            import inspect
            import numpy as np

            await ws.accept()
            voice = None
            sender_task = None

            async def send_one(event):
                body, audio = _split_agent_event(event)
                await ws.send_json(body)
                if audio:
                    await ws.send_bytes(audio)

            async def send_events(events):
                for event in events or ():
                    await send_one(event)

            async def maybe_await(value):
                if inspect.isawaitable(value):
                    return await value
                return value

            def is_async_voice(session) -> bool:
                return hasattr(session, "next_event")

            async def run_sender(session):
                while True:
                    event = await session.next_event()
                    await send_one(event)
                    if event.type == "done":
                        return

            async def start_voice():
                nonlocal voice, sender_task
                if sender_task is not None and not sender_task.done():
                    sender_task.cancel()
                voice = voice_factory(engine)
                sender_task = (
                    asyncio.create_task(run_sender(voice)) if is_async_voice(voice) else None
                )
                await ws.send_json({"type": "session_started"})

            async def close_voice():
                nonlocal voice, sender_task
                if voice is None:
                    return
                result = await maybe_await(voice.close())
                if is_async_voice(voice):
                    if sender_task is not None:
                        await sender_task
                else:
                    await send_events(result)
                voice = None
                sender_task = None

            try:
                while True:
                    message = await ws.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    if message.get("text") is not None:
                        import json

                        control = json.loads(message["text"])
                        kind = control.get("type")
                        if kind == "start":
                            await start_voice()
                        elif kind == "close":
                            await close_voice()
                        elif kind == "reset":
                            await close_voice()
                            await start_voice()
                        elif kind == "interrupt":
                            if voice is None:
                                await ws.send_json({"type": "error", "text": "no active voice session"})
                            else:
                                interrupt = getattr(voice, "interrupt", None)
                                if callable(interrupt):
                                    result = await maybe_await(interrupt())
                                    if not is_async_voice(voice):
                                        await send_events(result)
                                else:
                                    await ws.send_json({"type": "error", "text": "voice session does not support interrupt"})
                        elif kind == "tts_played":
                            if voice is None:
                                await ws.send_json({"type": "error", "text": "no active voice session"})
                            else:
                                playback_ack = getattr(voice, "playback_ack", None)
                                if callable(playback_ack):
                                    result = await maybe_await(playback_ack(control))
                                    if not is_async_voice(voice):
                                        await send_events(result)
                                else:
                                    await ws.send_json({"type": "tts_playback_ack", "payload": control})
                        else:
                            await ws.send_json(
                                {"type": "error", "text": f"unknown control message: {kind}"}
                            )
                    elif message.get("bytes") is not None:
                        if voice is None:
                            await start_voice()
                        pcm = np.frombuffer(message["bytes"], dtype=np.float32).copy()
                        result = await maybe_await(voice.feed(pcm))
                        if not is_async_voice(voice):
                            await send_events(result)
            except WebSocketDisconnect:
                if voice is not None:
                    try:
                        await maybe_await(voice.close())
                    except Exception:
                        logger.debug("voice session close after disconnect failed", exc_info=True)
            finally:
                if sender_task is not None and not sender_task.done():
                    sender_task.cancel()
                if voice is not None:
                    try:
                        await maybe_await(voice.close())
                    except Exception:
                        logger.debug("voice session final close failed", exc_info=True)

    return app


def serve(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    frontend_workers: int = 4,
    log_level: str = "info",
    **engine_kwargs,
) -> None:
    import uvicorn

    engine = AsyncAsrEngine(model=model, frontend_workers=frontend_workers, **engine_kwargs)
    try:
        uvicorn.run(create_app(engine), host=host, port=port, log_level=log_level)
    finally:
        engine.close()
