"""Everything that can be tuned about the speech engine, in one place."""

from dataclasses import dataclass

# Whisper works on 16 kHz mono audio; every input is converted to this.
SAMPLE_RATE = 16_000

# Multilingual models only. English-only models ("medium.en", ...) cannot handle Hindi/Urdu.
#   "turbo"     large-v3-turbo: near large-v3 accuracy, much faster, ~1.6 GB (default)
#   "large-v3"  ~3 GB; several times slower on a CPU
#   "medium" / "small" / "base"  smaller and faster, noticeably less accurate
MODEL_CHOICES = ("turbo", "large-v3", "medium", "small", "base")

# How many of the model's language guesses are kept with a transcript.
LANGUAGE_CANDIDATES = 5


@dataclass(frozen=True)
class EngineSettings:
    """Options that control model loading, chunking and decoding."""

    # --- model -------------------------------------------------------------
    model_size: str = "turbo"
    device: str = "auto"  # "auto" picks cuda when available, otherwise cpu
    compute_type: str = "auto"  # "auto" picks float16 on cuda and int8 on cpu
    cpu_threads: int = 0  # 0 = let CTranslate2 decide

    # --- decoding ----------------------------------------------------------
    beam_size: int = 5
    batch_size: int = 8

    # --- chunking (see chunking.py) ----------------------------------------
    # Speech is decoded in chunks of at most this many seconds (the planner aims at about
    # 85% of it), each cut in a pause so no word is cut in half. Whisper writes at most
    # 224 tokens per chunk, about 13 s of fast Hindi in Devanagari, and drops the rest;
    # 12 s stays under that, and a chunk that still overflows is decoded again in halves.
    chunk_seconds: int = 12
    # Voice-detector probability above which a moment counts as speech. Low on purpose:
    # a quiet caller must not be dropped; a false alarm only costs decoding time.
    speech_threshold: float = 0.3
    # Only silences at least this long are skipped. Shorter pauses are decoded together
    # with the speech around them, so nothing quiet is lost.
    skip_silence_seconds: float = 3.0

    # Only transcribe the first N seconds (0 = whole file). Handy for quick checks.
    limit_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.chunk_seconds <= 0 or self.chunk_seconds > 30:
            raise ValueError("chunk_seconds must be between 1 and 30")
        if not 0.0 < self.speech_threshold < 1.0:
            raise ValueError("speech_threshold must be between 0 and 1")
        if self.skip_silence_seconds < 0.5:
            raise ValueError("skip_silence_seconds must be at least 0.5")
        if self.beam_size < 1 or self.batch_size < 1:
            raise ValueError("beam_size and batch_size must be at least 1")
