"""The engine and the two-layer transcript, with a stand-in model and pipeline."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from likho_engine import Decision, Detection, EngineSettings, LocalLanguage, StopRequested, transcribe
from likho_engine.config import SAMPLE_RATE
from likho_engine.engine import resolve_device
from likho_engine.transcript import default_decision
from tests.fakes import FakePipeline, TruncatingPipeline, fake_engine, write_silent_wav

SILENCE = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)


def test_resolve_device_auto_picks_a_concrete_pair() -> None:
    device, compute_type = resolve_device("auto", "auto")
    assert (device, compute_type) in {("cpu", "int8"), ("cuda", "float16")}
    assert resolve_device("cpu", "float32") == ("cpu", "float32")


@pytest.mark.parametrize(
    "bad", [{"chunk_seconds": 0}, {"chunk_seconds": 31}, {"speech_threshold": 1.0}, {"batch_size": 0}]
)
def test_settings_reject_values_that_would_lose_speech(bad: dict) -> None:
    with pytest.raises(ValueError):
        EngineSettings(**bad)


class TestDetection:
    def test_keeps_the_runners_up(self) -> None:
        engine, _ = fake_engine("hi", 0.91, ranked=[("hi", 0.91), ("ur", 0.07), ("en", 0.02)])
        detection = engine.detect(SILENCE)
        assert detection == Detection("hi", 0.91, (("hi", 0.91), ("ur", 0.07), ("en", 0.02)))

    def test_keeps_at_most_five_candidates(self) -> None:
        ranked = [(code, 0.1) for code in ("hi", "ur", "en", "pa", "bn", "mr", "ta")]
        engine, _ = fake_engine("hi", 0.1, ranked=ranked)
        assert [code for code, _ in engine.detect(SILENCE).candidates] == ["hi", "ur", "en", "pa", "bn"]

    def test_works_when_the_model_reports_only_its_best_guess(self) -> None:
        engine, _ = fake_engine("en", 0.6)
        assert engine.detect(SILENCE).candidates == (("en", 0.6),)


@pytest.mark.parametrize(
    ("language", "probability", "expected"),
    [
        ("hi", 0.90, Decision("hi", True)),
        ("en", 0.95, Decision("en", False)),  # clearly English
        ("en", 0.66, Decision("hi", True)),  # Hindi with English words still counts as Hindi
        ("ur", 0.80, Decision("hi", True)),  # Urdu is decoded as Hindi so the text is Devanagari
    ],
)
def test_default_decision(language: str, probability: float, expected: Decision) -> None:
    assert default_decision(Detection(language, probability)) == expected


class TestDecode:
    def test_passes_the_plan_and_the_settings_to_the_pipeline(self) -> None:
        engine, pipeline = fake_engine(settings=EngineSettings(beam_size=3))
        plan = engine.plan(SILENCE)
        lines = list(engine.decode(SILENCE, plan, "hi", hotwords="त्रिफला"))

        assert [(line.start, line.end) for line in lines] == [(0.5, 4.0), (4.0, 9.5)]
        assert lines[0].text == "नमस्ते, आपका ऑर्डर कल तक पहुँच जाएगा"  # trimmed, script unchanged
        call = pipeline.calls[0]
        assert (call["language"], call["beam_size"], call["batch_size"], call["hotwords"]) == ("hi", 3, 8, "त्रिफला")
        # the voice detector finds nothing in silence, so the whole recording is decoded as one chunk
        assert call["clip_timestamps"] == [{"start": 0.0, "end": 10.0}]
        assert (plan.audio_seconds, plan.speech_seconds, plan.silence_skipped_seconds) == (10.0, 10.0, 0.0)

    def test_empty_lines_are_dropped(self) -> None:
        engine, _ = fake_engine(pipeline=FakePipeline([(0.0, 1.0, "   "), (1.0, 2.0, " ठीक है ")]))
        assert [line.text for line in engine.decode(SILENCE, engine.plan(SILENCE), "hi")] == ["ठीक है"]

    def test_stops_between_lines_when_asked(self) -> None:
        engine, _ = fake_engine()
        seen = []
        with pytest.raises(StopRequested):
            for line in engine.decode(SILENCE, engine.plan(SILENCE), "hi", should_stop=lambda: len(seen) >= 1):
                seen.append(line)
        assert len(seen) == 1

    def test_no_real_tokenizer_means_no_overflow_check(self) -> None:
        engine, _ = fake_engine()
        assert engine.budget(None) is None
        assert not engine.ran_out_of_tokens(SimpleNamespace(tokens=[0] * 500))

    def test_ran_out_of_tokens_triggers_at_the_measured_limit(self) -> None:
        engine, _ = fake_engine()
        engine._budgets[None] = 224
        assert engine.ran_out_of_tokens(SimpleNamespace(tokens=[0] * 224))
        assert engine.ran_out_of_tokens(SimpleNamespace(tokens=[0] * 223))  # seen with a shorter max_length
        assert not engine.ran_out_of_tokens(SimpleNamespace(tokens=[0] * 222))

    def test_a_chunk_that_fills_the_token_budget_is_decoded_again_in_halves(self) -> None:
        pipeline = TruncatingPipeline()
        engine, _ = fake_engine(pipeline=pipeline)
        engine._budgets[None] = 3

        lines = list(engine.decode(SILENCE, engine.plan(SILENCE), "hi"))

        assert len(pipeline.calls) == 2
        halves = pipeline.calls[1]["clip_timestamps"]
        assert len(halves) == 2
        assert halves[0]["start"] == 0.0 and halves[1]["end"] == 10.0
        assert halves[0]["end"] == halves[1]["start"]  # no second of the chunk is lost
        assert [line.text for line in lines] == ["नमस्ते", "धन्यवाद"]


class TestTranscribe:
    def test_writes_both_layers_and_reports_progress(self, tmp_path: Path) -> None:
        engine, _ = fake_engine("hi", 0.9, ranked=[("hi", 0.9), ("ur", 0.1)])
        started, seen = [], []

        transcript = transcribe(
            engine,
            write_silent_wav(tmp_path / "call.wav"),
            LocalLanguage(),
            on_start=lambda *args: started.append(args),
            on_segment=seen.append,
        )

        assert [(s.index, s.text_script, s.text_roman) for s in transcript.segments] == [
            (0, "नमस्ते, आपका ऑर्डर कल तक पहुँच जाएगा", "namaste, aapka order kal tak pahunch jayega"),
            (1, "दवा दिन में दो बार लीजिए", "dawa din mein do baar lijiye"),
        ]
        assert seen == transcript.segments
        assert started == [(10.0, transcript.detection, Decision("hi", True))]
        assert (transcript.decoded_as, transcript.policy, transcript.transliterated) == ("hi", "auto", True)
        assert (transcript.audio_seconds, transcript.chunks, transcript.silence_skipped_seconds) == (10.0, 1, 0.0)
        assert transcript.detection.candidates == (("hi", 0.9), ("ur", 0.1))
        assert transcript.elapsed_seconds > 0 and transcript.realtime_factor > 0

    def test_english_calls_are_not_transliterated(self, tmp_path: Path) -> None:
        engine, pipeline = fake_engine("en", 0.97, pipeline=FakePipeline([(0.0, 3.0, " Thank you for calling ")]))
        transcript = transcribe(engine, write_silent_wav(tmp_path / "call.wav"), LocalLanguage())
        assert pipeline.calls[0]["language"] == "en"
        assert transcript.transliterated is False
        assert transcript.segments[0].text_script == transcript.segments[0].text_roman == "Thank you for calling"

    @pytest.mark.parametrize(("forced", "decoded", "transliterated"), [("hi", "hi", True), ("en", "en", False)])
    def test_a_forced_language_overrides_the_detection(
        self, tmp_path: Path, forced: str, decoded: str, transliterated: bool
    ) -> None:
        engine, pipeline = fake_engine("en", 0.99)
        transcript = transcribe(engine, write_silent_wav(tmp_path / "call.wav"), LocalLanguage(), force_language=forced)
        assert pipeline.calls[0]["language"] == decoded
        assert (transcript.decoded_as, transcript.policy, transcript.transliterated) == (
            decoded,
            forced,
            transliterated,
        )

    def test_auto_is_not_a_forced_language(self, tmp_path: Path) -> None:
        engine, _ = fake_engine("hi", 0.9)
        transcript = transcribe(engine, write_silent_wav(tmp_path / "call.wav"), LocalLanguage(), force_language="auto")
        assert transcript.policy == "auto"

    def test_glossary_becomes_hotwords_and_own_spellings_apply(self, tmp_path: Path) -> None:
        engine, pipeline = fake_engine(pipeline=FakePipeline([(0.0, 3.0, " त्रिफला लीजिए ")]))
        language = LocalLanguage(glossary=["त्रिफला", "अश्वगंधा"], spellings={"त्रिफला": "Triphala"})

        transcript = transcribe(engine, write_silent_wav(tmp_path / "call.wav"), language)

        assert pipeline.calls[0]["hotwords"] == "त्रिफला, अश्वगंधा"
        assert transcript.segments[0].text_roman == "Triphala lijiye"
        assert transcript.segments[0].text_script == "त्रिफला लीजिए"  # layer 1 is never rewritten

    def test_a_looping_phrase_is_trimmed_in_both_layers(self, tmp_path: Path) -> None:
        engine, _ = fake_engine(pipeline=FakePipeline([(0.0, 5.0, " ठीक है जी, ठीक है जी, ठीक है जी, ठीक है जी, धन्यवाद ")]))
        segment = transcribe(engine, write_silent_wav(tmp_path / "call.wav"), LocalLanguage()).segments[0]
        assert segment.text_script == "ठीक है जी, धन्यवाद"
        assert segment.text_roman == "theek hai ji, dhanyavaad"

    def test_audio_can_be_passed_in_instead_of_a_file(self) -> None:
        engine, _ = fake_engine()
        transcript = transcribe(engine, "rec_123", LocalLanguage(), audio=SILENCE)
        assert transcript.source == "rec_123"
        assert len(transcript.segments) == 2

    def test_stopping_raises_and_keeps_what_was_reported(self, tmp_path: Path) -> None:
        engine, _ = fake_engine()
        seen = []
        with pytest.raises(StopRequested):
            transcribe(
                engine,
                write_silent_wav(tmp_path / "call.wav"),
                LocalLanguage(),
                on_segment=seen.append,
                should_stop=lambda: len(seen) >= 1,
            )
        assert len(seen) == 1
