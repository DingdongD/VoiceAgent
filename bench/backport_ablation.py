"""Ablation behind the /home/nano-vllm backport defaults.

Two knobs move independently: how much audio goes into one audio-tower call
(max_audio_encode_frames) and how many sequences the text decoder runs at once.
Separating them is what shows that decoder concurrency is a free 7x while batching
the *upstream* audio tower buys 5% at a large cost in WER — hence the shipped
default of one audio-tower call per request.
"""
import sys
import time

sys.path.insert(0, "/home/qwen-asr-vllm")

from bench.data import load_librispeech
from bench.metrics import compute_wer

# nanovllm itself is an editable install, but its ASR wrapper reaches into the
# qwen_asr research package for the processor and output parser.
sys.path.insert(0, "/home/qwen_asr/Qwen3-ASR")

MODEL = "/mnt/llm_data/voice_ckpt/Qwen3-ASR-0.6B"

samples = load_librispeech(num_samples=64)
audios = [(s.audio, s.sample_rate) for s in samples]
refs = [s.text for s in samples]
audio_sec = sum(s.duration for s in samples)


def run(label, *, max_audio_encode_frames, submit_all):
    import gc
    import torch
    from nanovllm.asr.qwen3 import NanoQwen3ASR

    backend = NanoQwen3ASR(MODEL, max_audio_encode_frames=max_audio_encode_frames)
    started = time.perf_counter()
    if submit_all:
        hyps = [r.text for r in backend.transcribe(audios)]
    else:
        hyps = [backend.transcribe([a])[0].text for a in audios]
    elapsed = time.perf_counter() - started
    print(
        f"{label:<34} wall={elapsed:6.2f}s rtf={elapsed / audio_sec:.4f} "
        f"wer={compute_wer(refs, hyps):.4f}"
    )
    backend.close()
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    return hyps


serial = run("decode=1  encode=1", max_audio_encode_frames=1, submit_all=False)
decode_only = run("decode=64 encode=1", max_audio_encode_frames=1, submit_all=True)
both = run("decode=64 encode=12000", max_audio_encode_frames=12000, submit_all=True)

print(f"decode-only vs serial identical: {sum(a == b for a, b in zip(decode_only, serial))}/64")
print(f"both        vs serial identical: {sum(a == b for a, b in zip(both, serial))}/64")
