"""Local ASR dataset loading.

``datasets`` 5.x decodes audio through ``torchcodec``, which has no build matching
this environment's torch. The audio column is therefore loaded undecoded and
handed to ``soundfile``, which keeps the dependency surface to what is already
installed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000

LIBRISPEECH_ROOT = Path("/mnt/llm_data/asr_datasets/librispeech_asr")
LIBRISPEECH_SPLITS = {
    "test-clean": ("clean", "test"),
    "test-other": ("other", "test"),
    "dev-clean": ("clean", "validation"),
    "dev-other": ("other", "validation"),
}


@dataclass
class AudioSample:
    audio: np.ndarray
    sample_rate: int
    text: str
    sample_id: str

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


def _decode(entry: dict) -> tuple[np.ndarray, int]:
    if entry.get("bytes"):
        return sf.read(io.BytesIO(entry["bytes"]), dtype="float32")
    return sf.read(entry["path"], dtype="float32")


def _arrow_files(config_name: str, arrow_split: str) -> list[str]:
    for root in (LIBRISPEECH_ROOT / config_name, LIBRISPEECH_ROOT / "openslr___librispeech_asr" / config_name):
        if not root.exists():
            continue
        files = sorted(str(p) for p in root.glob(f"0.0.0/*/librispeech_asr-{arrow_split}*.arrow"))
        if files:
            return files
    raise FileNotFoundError(
        f"no librispeech arrow shards for {config_name}/{arrow_split} under {LIBRISPEECH_ROOT}"
    )


def load_librispeech(
    split: str = "test-clean",
    num_samples: int | None = None,
    seed: int | None = None,
    sort_by_duration: bool = False,
) -> list[AudioSample]:
    """Load a LibriSpeech split from the local arrow shards."""
    from datasets import Audio, load_dataset

    if split not in LIBRISPEECH_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(LIBRISPEECH_SPLITS)}")
    config_name, arrow_split = LIBRISPEECH_SPLITS[split]

    dataset = load_dataset("arrow", data_files=_arrow_files(config_name, arrow_split), split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    indices = list(range(len(dataset)))
    if num_samples is not None and num_samples < len(indices):
        if seed is None:
            indices = indices[:num_samples]
        else:
            rng = np.random.default_rng(seed)
            indices = sorted(rng.choice(len(dataset), size=num_samples, replace=False).tolist())

    samples = []
    for index in indices:
        row = dataset[index]
        audio, sample_rate = _decode(row["audio"])
        samples.append(
            AudioSample(
                audio=audio,
                sample_rate=sample_rate,
                text=row["text"],
                sample_id=row.get("id") or str(index),
            )
        )

    if sort_by_duration:
        samples.sort(key=lambda sample: sample.duration)
    return samples


TEDLIUM_LONG_FORM = Path("/mnt/llm_data/asr_datasets/tedlium/tedlium_long_form")
_RESAMPLE_CACHE = Path("/tmp/qwen_asr_vllm_tedlium_16k")


def load_tedlium_long_form(
    min_seconds: float = 0.0, max_seconds: float | None = None
) -> list[AudioSample]:
    """Whole TED talks, 3 to 22 minutes each, as they were recorded.

    Preferred over stitching short clips together for anything that depends on the
    audio making sense end to end. It is also spontaneous presentation speech rather
    than read audiobook prose, which matters for cost attribution: decode time is
    proportional to emitted tokens, and a speaker who pauses emits fewer tokens per
    second of audio than a narrator reading continuously.

    The shards store 192kHz mono WAV, so every load would otherwise pay a 12:1
    resample of up to 22 minutes of audio. Resampled copies are cached under
    ``/tmp`` -- the cost belongs to fixture preparation, not to a measured stage.
    """
    import soundfile

    shards = sorted(TEDLIUM_LONG_FORM.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {TEDLIUM_LONG_FORM}")

    _RESAMPLE_CACHE.mkdir(parents=True, exist_ok=True)
    samples: list[AudioSample] = []
    for shard in shards:
        import pyarrow.parquet as pq

        table = pq.read_table(shard)
        for index in range(table.num_rows):
            sample_id = f"{shard.stem}-{index}"
            cached = _RESAMPLE_CACHE / f"{sample_id}.npy"
            text = table.column("gt")[index].as_py().strip()
            if cached.exists():
                audio = np.load(cached)
            else:
                raw = table.column("audio")[index].as_py()["bytes"]
                audio, sample_rate = soundfile.read(io.BytesIO(raw), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=-1)
                if sample_rate != TARGET_SAMPLE_RATE:
                    import librosa

                    audio = librosa.resample(
                        audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
                    )
                audio = np.ascontiguousarray(audio, dtype=np.float32)
                np.save(cached, audio)

            duration = len(audio) / TARGET_SAMPLE_RATE
            if duration < min_seconds or (max_seconds and duration > max_seconds):
                continue
            samples.append(
                AudioSample(
                    audio=audio,
                    sample_rate=TARGET_SAMPLE_RATE,
                    text=text,
                    sample_id=sample_id,
                )
            )

    samples.sort(key=lambda sample: sample.duration)
    return samples


def load_contiguous_speech(
    seconds: float, split: str = "test-clean"
) -> tuple[np.ndarray, str]:
    """Genuinely continuous speech of at least ``seconds``, with its reference text.

    LibriSpeech ids are ``speaker-chapter-utterance`` and utterances within a chapter
    are consecutive segments of one audiobook reading, so concatenating a chapter in id
    order reconstructs the original recording. Pooling random clips instead produces
    audio that jumps between speakers and mid-sentence, which matters for anything
    sensitive to whether the audio makes sense: a streaming transcript revises itself
    far more when the model is fed discontinuous speech, and that revision would be an
    artefact of the fixture rather than a property of the model.

    Chapters are chained only when one is too short; the longest is about 8 minutes.
    """
    from collections import defaultdict

    chapters: dict[tuple[str, str], list[AudioSample]] = defaultdict(list)
    for sample in load_librispeech(split=split):
        speaker, chapter, _ = sample.sample_id.split("-", 2)
        chapters[(speaker, chapter)].append(sample)

    ordered = sorted(
        chapters.values(),
        key=lambda group: sum(item.duration for item in group),
        reverse=True,
    )
    audio: list[np.ndarray] = []
    text: list[str] = []
    collected = 0.0
    # Stop on utterance boundaries so the returned text matches the audio. Clipping
    # mid-utterance left reference words with no acoustics and inflated deletion WER.
    done = False
    for group in ordered:
        for sample in sorted(group, key=lambda item: item.sample_id):
            if collected >= seconds:
                done = True
                break
            if collected > 0 and collected + sample.duration > seconds:
                done = True
                break
            audio.append(sample.audio)
            text.append(sample.text)
            collected += sample.duration
        if done or collected >= seconds:
            break

    if collected < seconds * 0.9:
        raise ValueError(f"{split} holds only {collected:.0f}s, {seconds:.0f}s requested")
    waveform = np.concatenate(audio)
    return waveform.astype(np.float32), " ".join(text)
