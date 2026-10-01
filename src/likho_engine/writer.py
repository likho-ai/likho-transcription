"""Writing transcripts to disk as txt, srt and json."""

import json
from collections.abc import Iterable
from pathlib import Path

from likho_engine.transcript import Segment, Transcript

FORMAT_CHOICES = ("txt", "srt", "json")
TIMESTAMP_CHOICES = ("clock", "seconds")
LAYER_CHOICES = ("roman", "script")
# The file name says which layer a text file holds: call.hinglish.txt or call.script.txt.
LAYER_SUFFIX = {"roman": "hinglish", "script": "script"}


def clock(seconds: float) -> str:
    """12.9 -> '00:12', 3725.0 -> '1:02:05'."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def text_of(segment: Segment, layer: str = "roman") -> str:
    return segment.text_script if layer == "script" else segment.text_roman


def format_line(segment: Segment, timestamps: str = "clock", layer: str = "roman") -> str:
    """One console/txt line: [00:12 -> 00:15] text  (or [12.34s -> 15.67s] text)."""
    text = text_of(segment, layer)
    if timestamps == "seconds":
        return f"[{segment.start:.2f}s -> {segment.end:.2f}s] {text}"
    return f"[{clock(segment.start)} -> {clock(segment.end)}] {text}"


def to_txt(transcript: Transcript, timestamps: str = "clock", layer: str = "roman") -> str:
    return "".join(format_line(s, timestamps, layer) + "\n" for s in transcript.segments)


def _srt_time(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def to_srt(transcript: Transcript, layer: str = "roman") -> str:
    blocks = []
    for number, s in enumerate(transcript.segments, start=1):
        blocks.append(f"{number}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{text_of(s, layer)}\n")
    return "\n".join(blocks)


def to_json(transcript: Transcript) -> str:
    """Everything: both layers of every line, the language detection and the timings."""
    data = {
        "source": transcript.source,
        "language": {
            "detected": transcript.detection.language,
            "probability": transcript.detection.probability,
            "candidates": [{"language": code, "probability": prob} for code, prob in transcript.detection.candidates],
            "decoded_as": transcript.decoded_as,
            "policy": transcript.policy,
        },
        "audio_seconds": transcript.audio_seconds,
        "elapsed_seconds": transcript.elapsed_seconds,
        "chunks": transcript.chunks,
        "silence_skipped_seconds": transcript.silence_skipped_seconds,
        "segments": [
            {
                "index": s.index,
                "start": s.start,
                "end": s.end,
                "text_script": s.text_script,
                "text_roman": s.text_roman,
            }
            for s in transcript.segments
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def transcript_paths(source: Path, out_dir: Path, formats: Iterable[str], layer: str = "roman") -> list[Path]:
    """Where the transcript of a recording goes: <out_dir>/<recording name>.<layer>.<format>.

    json always holds both layers, so it is named <recording name>.likho.json whatever the layer.
    """
    stem = source.stem
    return [
        out_dir / (f"{stem}.likho.json" if fmt == "json" else f"{stem}.{LAYER_SUFFIX[layer]}.{fmt}") for fmt in formats
    ]


def write_transcript(
    transcript: Transcript,
    out_dir: Path,
    formats: Iterable[str],
    timestamps: str = "clock",
    layer: str = "roman",
) -> list[Path]:
    """Save the transcript in each format and return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = list(formats)
    written = []
    for fmt, path in zip(formats, transcript_paths(Path(transcript.source), out_dir, formats, layer), strict=True):
        if fmt == "txt":
            content = to_txt(transcript, timestamps, layer)
        elif fmt == "srt":
            content = to_srt(transcript, layer)
        else:
            content = to_json(transcript)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
