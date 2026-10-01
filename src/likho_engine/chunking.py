"""Cutting a recording into the chunks Whisper decodes one at a time.

Whisper writes at most 224 tokens per decode, which is only about 13 s of fast Hindi in
Devanagari; everything after that is silently dropped. faster-whisper's own splitter also
cuts speech at a length limit wherever that falls: in the middle of a word when nobody
pauses, leaving a short remainder that is decoded on its own with no context.

This module plans the chunks instead:

* Silero VAD finds the speech. Only silences of at least skip_silence_seconds are left
  out; shorter pauses are decoded together with the speech around them, so a quiet
  caller the detector is unsure about is never dropped.
* Speech longer than chunk_seconds is cut into evenly sized pieces that aim a little
  below the limit. Each cut goes in a pause (a run of near-silent frames) as close to
  the even split as possible, and only when there is no pause at all at the quietest
  single frame. Even sizes mean no 1 s leftovers.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps

from likho_engine.config import SAMPLE_RATE

log = logging.getLogger(__name__)

FRAME_SECONDS = 0.02  # loudness is measured per 20 ms frame...
SMOOTH_FRAMES = 3  # ...and averaged over 60 ms, so one quiet frame inside a word is not taken for a pause
QUIET_FRACTION = 0.1  # a frame under a tenth of the region's typical loudness is silent
LONG_PAUSE_FRAMES = 6  # 120 ms of silence is a gap between words or sentences...
SHORT_PAUSE_FRAMES = 3  # ...60 ms may still be the closure of a "k" or "p" inside a word, so it is second choice
TARGET_FRACTION = 0.85  # pieces aim at this share of chunk_seconds, leaving room to move a cut to a pause...
SEARCH_FRACTION = 0.25  # ...this far either side of the even split (as a share of chunk_seconds)
MIN_PIECE_FRACTION = 0.3  # a piece is never shorter than this share of chunk_seconds: no context-less scraps
SPEECH_PAD_SECONDS = 0.4  # kept around each stretch of speech so its first and last word are whole
MIN_SPEECH_SECONDS = 0.25  # shorter blips on their own are clicks, not words

Samples = tuple[int, int]  # a [start, end) range of samples
FRAME = int(FRAME_SECONDS * SAMPLE_RATE)


@dataclass(frozen=True)
class Chunk:
    """A [start, end) range of the recording, in seconds."""

    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


def speech_regions(audio: np.ndarray, threshold: float, skip_silence_seconds: float) -> list[Samples]:
    """Sample ranges of speech, separated only by silences of at least skip_silence_seconds."""
    options = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=int(MIN_SPEECH_SECONDS * 1000),
        max_speech_duration_s=float("inf"),
        min_silence_duration_ms=int(skip_silence_seconds * 1000),
        speech_pad_ms=int(SPEECH_PAD_SECONDS * 1000),
    )
    regions = get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
    return [(int(region["start"]), int(region["end"])) for region in regions]


def loudness(audio: np.ndarray) -> np.ndarray:
    """Smoothed RMS level of each FRAME_SECONDS frame."""
    count = len(audio) // FRAME
    if count == 0:
        return np.zeros(0)
    frames = audio[: count * FRAME].reshape(count, FRAME).astype(np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return np.convolve(rms, np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES, mode="same")


def quiet_level(loud: np.ndarray, region: Samples) -> float:
    """The loudness under which a frame of this region counts as silent."""
    first, last = region[0] // FRAME, max(region[0] // FRAME + 1, region[1] // FRAME)
    frames = loud[first:last]
    return QUIET_FRACTION * float(np.median(frames)) if len(frames) else 0.0


def _frames(lo: int, hi: int) -> tuple[int, int]:
    """The whole frames inside samples [lo, hi) as a [first, last) frame range."""
    return math.ceil(lo / FRAME), hi // FRAME


def find_pause(loud: np.ndarray, lo: int, hi: int, target: int, level: float, min_frames: int) -> int | None:
    """The middle sample of the pause (>= min_frames silent frames) nearest to target within [lo, hi)."""
    first, last = _frames(lo, hi)
    pauses: list[int] = []
    run_start: int | None = None
    for index in range(first, last + 1):
        silent = index < last and loud[index] < level
        if silent and run_start is None:
            run_start = index
        elif not silent and run_start is not None:
            if index - run_start >= min_frames:
                pauses.append((run_start + index) * FRAME // 2)
            run_start = None
    return min(pauses, key=lambda sample: abs(sample - target)) if pauses else None


def quietest_sample(loud: np.ndarray, lo: int, hi: int) -> int:
    """The start of the quietest frame in samples [lo, hi); the middle when no whole frame fits."""
    first, last = _frames(lo, hi)
    if last <= first:
        return (lo + hi) // 2
    return int(first + np.argmin(loud[first:last])) * FRAME


def cut_point(loud: np.ndarray, lo: int, hi: int, target: int, level: float, wide: Samples | None = None) -> int:
    """Where to cut: a long pause in [lo, hi) near target, else a long pause anywhere in the wider
    range, else a short pause in [lo, hi), else the quietest frame in [lo, hi)."""
    windows = [(lo, hi, LONG_PAUSE_FRAMES)]
    if wide is not None and (wide[0] < lo or wide[1] > hi):
        windows.append((wide[0], wide[1], LONG_PAUSE_FRAMES))
    windows.append((lo, hi, SHORT_PAUSE_FRAMES))
    for low, high, min_frames in windows:
        pause = find_pause(loud, low, high, target, level, min_frames)
        if pause is not None:
            return pause
    return quietest_sample(loud, lo, hi)


def split_evenly(region: Samples, chunk_samples: int, loud: np.ndarray) -> list[Samples]:
    """Cut a region into evenly sized pieces of at most chunk_samples, each cut in a pause near the even split."""
    start, end = region
    level = quiet_level(loud, region)
    slack = int(SEARCH_FRACTION * chunk_samples)
    min_piece = int(MIN_PIECE_FRACTION * chunk_samples)
    pieces: list[Samples] = []
    while end - start > chunk_samples:
        remaining = end - start
        share = remaining / math.ceil(remaining / (TARGET_FRACTION * chunk_samples))  # even share for this piece
        target = int(start + share)
        wide = (start + min_piece, min(start + chunk_samples, end - min_piece))  # wherever the piece may end
        lo, hi = max(wide[0], target - slack), min(wide[1], target + slack)
        cut = cut_point(loud, lo, hi, target, level, wide)
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces


def halve(chunk: Chunk, loud: np.ndarray) -> list[Chunk]:
    """Split a chunk in two, in a pause as near its middle as possible."""
    start, end = int(chunk.start * SAMPLE_RATE), int(chunk.end * SAMPLE_RATE)
    middle, slack = start + (end - start) // 2, int(SEARCH_FRACTION * (end - start))
    cut = cut_point(loud, middle - slack, middle + slack, middle, quiet_level(loud, (start, end)))
    return [Chunk(start / SAMPLE_RATE, cut / SAMPLE_RATE), Chunk(cut / SAMPLE_RATE, end / SAMPLE_RATE)]


def plan_chunks(
    audio: np.ndarray,
    chunk_seconds: float,
    speech_threshold: float,
    skip_silence_seconds: float,
    loud: np.ndarray | None = None,
) -> list[Chunk]:
    """The chunks to decode, in order. Every moment of speech is in exactly one chunk."""
    if len(audio) == 0:
        return []
    regions = speech_regions(audio, speech_threshold, skip_silence_seconds)
    if not regions:
        log.warning("The voice detector found no speech; decoding the whole recording anyway.")
        regions = [(0, len(audio))]
    if loud is None:
        loud = loudness(audio)
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    return [
        Chunk(start / SAMPLE_RATE, end / SAMPLE_RATE)
        for region in regions
        for start, end in split_evenly(region, chunk_samples, loud)
    ]
