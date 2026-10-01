"""Command line: transcribe recordings on this machine, without any service.

likho-transcribe                          # every new recording in ./recordings
likho-transcribe call.wav other.mp3       # specific files or folders
likho-transcribe -f txt -f srt -f json --limit-seconds 60 call.wav
likho-transcribe --layer script call.wav  # the Devanagari layer instead of Hinglish
likho-transcribe --watch                  # keep running, transcribe new files as they arrive
"""

import argparse
import io
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from likho_engine.audio import find_recordings
from likho_engine.config import MODEL_CHOICES, EngineSettings
from likho_engine.engine import Detection, Engine
from likho_engine.files import DEFAULT_GLOSSARY, DEFAULT_SPELLINGS, load_glossary, load_spellings, optional_file
from likho_engine.transcript import Decision, Language, LocalLanguage, Segment, transcribe
from likho_engine.watch import watch
from likho_engine.writer import (
    FORMAT_CHOICES,
    LAYER_CHOICES,
    TIMESTAMP_CHOICES,
    format_line,
    transcript_paths,
    write_transcript,
)

log = logging.getLogger("likho_engine")

LANGUAGE_CHOICES = ("auto", "hi", "en")


@dataclass(frozen=True)
class Output:
    """Where and how the transcripts are written."""

    directory: Path
    formats: tuple[str, ...]
    timestamps: str
    layer: str
    language: str  # "auto", or a language to force


def build_parser() -> argparse.ArgumentParser:
    defaults = EngineSettings()
    parser = argparse.ArgumentParser(
        prog="likho-transcribe",
        description="Transcribe Hindi/Urdu/English recordings into Devanagari and Hinglish with faster-whisper.",
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="recordings or folders (default: ./recordings)")

    out = parser.add_argument_group("output")
    out.add_argument("-o", "--output-dir", type=Path, default=Path("transcripts"), help="default: %(default)s")
    out.add_argument("-f", "--format", action="append", choices=FORMAT_CHOICES, dest="formats")
    out.add_argument("--layer", choices=LAYER_CHOICES, default="roman", help="roman = Hinglish, script = as spoken")
    out.add_argument("--timestamps", choices=TIMESTAMP_CHOICES, default="clock", help="00:15 or 15.57s")
    out.add_argument("--force", action="store_true", help="redo recordings that already have a transcript")
    out.add_argument("-q", "--quiet", action="store_true", help="do not print transcript lines")
    out.add_argument("--no-progress", action="store_true", help="hide the progress bar")

    model = parser.add_argument_group("model")
    model.add_argument("-m", "--model", default=defaults.model_size, help=f"{', '.join(MODEL_CHOICES)} or a local path")
    model.add_argument("-l", "--language", choices=LANGUAGE_CHOICES, default="auto", help="force the language")
    model.add_argument("--device", choices=("auto", "cpu", "cuda"), default=defaults.device)
    model.add_argument("--compute-type", default=defaults.compute_type, help="auto, int8, float16, float32 ...")
    model.add_argument("--cpu-threads", type=int, default=defaults.cpu_threads)
    model.add_argument("--beam-size", type=int, default=defaults.beam_size)
    model.add_argument("--batch-size", type=int, default=defaults.batch_size)

    chunking = parser.add_argument_group("chunking")
    chunking.add_argument("--chunk-seconds", type=int, default=defaults.chunk_seconds)
    chunking.add_argument("--speech-threshold", type=float, default=defaults.speech_threshold)
    chunking.add_argument("--skip-silence-seconds", type=float, default=defaults.skip_silence_seconds)

    vocab = parser.add_argument_group("vocabulary")
    vocab.add_argument("--glossary", type=Path, help="names to listen for, one per line (default: ./glossary.txt)")
    vocab.add_argument(
        "--spellings",
        "--custom-words",
        type=Path,
        dest="spellings",
        help='JSON {"देवनागरी": "hinglish"} (default: ./custom_words.json)',
    )

    run = parser.add_argument_group("run")
    run.add_argument("--limit-seconds", type=float, default=0.0, help="only the first N seconds of each file")
    run.add_argument("--watch", action="store_true", help="keep running and transcribe new recordings")
    run.add_argument("--poll-seconds", type=float, default=10.0)
    return parser


def engine_settings(args: argparse.Namespace) -> EngineSettings:
    return EngineSettings(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        chunk_seconds=args.chunk_seconds,
        speech_threshold=args.speech_threshold,
        skip_silence_seconds=args.skip_silence_seconds,
        limit_seconds=args.limit_seconds,
    )


def configure_output() -> None:
    # Hinglish is ASCII, but Devanagari is not; never crash on it, even when the output
    # is redirected to a file on Windows.
    if isinstance(sys.stdout, io.TextIOWrapper):
        # line_buffering: lines show up immediately even when output is redirected to a file
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("faster_whisper", "huggingface_hub", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def has_transcript(path: Path, output: Output) -> bool:
    return all(p.is_file() for p in transcript_paths(path, output.directory, output.formats, output.layer))


class ConsoleProgress:
    """Console output for one recording: a progress bar with ETA (on stderr) and each line as it arrives."""

    def __init__(self, show_lines: bool, show_bar: bool, output: Output) -> None:
        self.show_lines = show_lines
        self.show_bar = show_bar
        self.output = output
        self.bar: tqdm | None = None

    def start(self, audio_seconds: float, detection: Detection, decision: Decision) -> None:
        log.info(
            "Detected '%s' (%.2f) -> decoding as '%s'", detection.language, detection.probability, decision.decode_as
        )
        self.bar = tqdm(
            total=round(audio_seconds),
            unit="s",
            disable=not self.show_bar,
            leave=False,
            dynamic_ncols=True,
            bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
        )

    def segment(self, segment: Segment) -> None:
        if self.bar is not None:
            self.bar.update(max(0.0, min(segment.end, self.bar.total) - self.bar.n))
        if self.show_lines:
            tqdm.write(format_line(segment, self.output.timestamps, self.output.layer))

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


def transcribe_recordings(
    engine: Engine, language: Language, recordings: list[Path], output: Output, *, quiet: bool, progress_bar: bool
) -> int:
    """Transcribe each recording, printing progress and saving the result. Returns the number of failures."""
    failures = 0
    for path in recordings:
        log.info("\n=== %s ===", path.name)
        progress = ConsoleProgress(show_lines=not quiet, show_bar=progress_bar, output=output)
        try:
            transcript = transcribe(
                engine,
                path,
                language,
                force_language=output.language,
                on_start=progress.start,
                on_segment=progress.segment,
            )
        except Exception:  # keep going with the other recordings
            failures += 1
            log.exception("Failed to transcribe %s", path.name)
            continue
        finally:
            progress.close()
        written = write_transcript(transcript, output.directory, output.formats, output.timestamps, output.layer)
        log.info(
            "(%.0fs of audio in %.0fs, %.1fx realtime, %.0fs of silence skipped) saved: %s",
            transcript.audio_seconds,
            transcript.elapsed_seconds,
            transcript.realtime_factor,
            transcript.silence_skipped_seconds,
            ", ".join(p.name for p in written),
        )
    return failures


def watched_folder(inputs: list[Path], default_dir: Path) -> Path:
    """--watch works on exactly one folder: the one given, or the recordings folder."""
    if len(inputs) > 1 or (inputs and not inputs[0].is_dir()):
        raise ValueError("--watch takes a single folder (or none, for the recordings folder)")
    return inputs[0].resolve() if inputs else default_dir


def main(argv: list[str] | None = None, *, engine: Engine | None = None) -> int:
    """Run the command. `engine` lets tests pass one built with a stand-in model."""
    configure_output()
    args = build_parser().parse_args(argv)
    recordings_dir = Path("recordings").resolve()
    output = Output(
        directory=args.output_dir,
        formats=tuple(args.formats or ("txt",)),
        timestamps=args.timestamps,
        layer=args.layer,
        language=args.language,
    )
    try:
        settings = engine_settings(args)
        recordings = find_recordings(args.inputs, recordings_dir)
        folder = watched_folder(args.inputs, recordings_dir) if args.watch else None
        glossary_file = optional_file(args.glossary, DEFAULT_GLOSSARY)
        spellings_file = optional_file(args.spellings, DEFAULT_SPELLINGS)
        glossary, spellings = load_glossary(glossary_file), load_spellings(spellings_file)
    except (ValueError, OSError) as exc:
        log.error("error: %s", exc)
        return 2

    pending = recordings if args.force else [p for p in recordings if not has_transcript(p, output)]
    skipped = len(recordings) - len(pending)
    if skipped:
        log.info("Skipping %d recording(s) that already have a transcript (use --force to redo them).", skipped)
    if not pending and folder is None:
        if not recordings:
            log.error("No recordings found. Put audio files in %s or pass them as arguments.", recordings_dir)
            return 1
        log.info("Nothing to do.")
        return 0

    if glossary:
        log.info("Loaded %d glossary names from %s", len(glossary), glossary_file)
    if spellings:
        log.info("Loaded %d spellings from %s", len(spellings), spellings_file)
    language = LocalLanguage(glossary, spellings)
    if engine is None:
        engine = Engine(settings)
    bar = not args.no_progress and sys.stderr.isatty()

    failures = (
        transcribe_recordings(engine, language, pending, output, quiet=args.quiet, progress_bar=bar) if pending else 0
    )

    if folder is not None:
        active = engine

        def handle_new(paths: list[Path]) -> None:
            transcribe_recordings(active, language, paths, output, quiet=args.quiet, progress_bar=bar)

        try:
            watch(
                folder,
                is_done=lambda path: has_transcript(path, output),
                handle=handle_new,
                poll_seconds=args.poll_seconds,
            )
        except KeyboardInterrupt:
            log.info("\nStopped watching.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
