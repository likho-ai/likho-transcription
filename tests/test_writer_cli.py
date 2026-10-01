"""Output files and the command line, with a stand-in model."""

import json
from pathlib import Path

import pytest

from likho_engine import Detection, Segment, Transcript
from likho_engine.cli import main
from likho_engine.files import load_glossary, load_spellings, optional_file
from likho_engine.writer import clock, format_line, to_json, to_srt, to_txt, transcript_paths, write_transcript
from tests.fakes import fake_engine, write_silent_wav

TRANSCRIPT = Transcript(
    source="calls/call-0001.mp3",
    detection=Detection("hi", 0.91, (("hi", 0.91), ("ur", 0.07))),
    decoded_as="hi",
    policy="auto",
    transliterated=True,
    audio_seconds=75.0,
    elapsed_seconds=50.0,
    chunks=7,
    silence_skipped_seconds=4.0,
    segments=[
        Segment(0, 2.1, 12.4, "नमस्ते जी", "namaste ji"),
        Segment(1, 3725.0, 3731.25, "धन्यवाद", "dhanyavaad"),
    ],
)


class TestWriter:
    @pytest.mark.parametrize(("seconds", "text"), [(12.9, "00:12"), (75, "01:15"), (3725.0, "1:02:05")])
    def test_clock(self, seconds: float, text: str) -> None:
        assert clock(seconds) == text

    def test_line_formats(self) -> None:
        first = TRANSCRIPT.segments[0]
        assert format_line(first) == "[00:02 -> 00:12] namaste ji"
        assert format_line(first, "seconds") == "[2.10s -> 12.40s] namaste ji"
        assert format_line(first, layer="script") == "[00:02 -> 00:12] नमस्ते जी"

    def test_txt_and_srt_in_either_layer(self) -> None:
        assert to_txt(TRANSCRIPT) == "[00:02 -> 00:12] namaste ji\n[1:02:05 -> 1:02:11] dhanyavaad\n"
        assert to_srt(TRANSCRIPT, "script") == (
            "1\n00:00:02,100 --> 00:00:12,400\nनमस्ते जी\n\n2\n01:02:05,000 --> 01:02:11,250\nधन्यवाद\n"
        )

    def test_json_holds_both_layers_and_the_detection(self) -> None:
        data = json.loads(to_json(TRANSCRIPT))
        assert data["language"] == {
            "detected": "hi",
            "probability": 0.91,
            "candidates": [{"language": "hi", "probability": 0.91}, {"language": "ur", "probability": 0.07}],
            "decoded_as": "hi",
            "policy": "auto",
        }
        assert data["segments"][0] == {
            "index": 0,
            "start": 2.1,
            "end": 12.4,
            "text_script": "नमस्ते जी",
            "text_roman": "namaste ji",
        }
        assert (data["chunks"], data["silence_skipped_seconds"]) == (7, 4.0)

    def test_file_names_say_which_layer_they_hold(self, tmp_path: Path) -> None:
        source = Path("calls/call-0001.mp3")
        assert [p.name for p in transcript_paths(source, tmp_path, ["txt", "srt", "json"])] == [
            "call-0001.hinglish.txt",
            "call-0001.hinglish.srt",
            "call-0001.likho.json",
        ]
        assert transcript_paths(source, tmp_path, ["txt"], "script")[0].name == "call-0001.script.txt"

    def test_write_creates_the_folder_and_utf8_files(self, tmp_path: Path) -> None:
        written = write_transcript(TRANSCRIPT, tmp_path / "out", ["txt", "json"], layer="script")
        assert [p.name for p in written] == ["call-0001.script.txt", "call-0001.likho.json"]
        assert written[0].read_text(encoding="utf-8").startswith("[00:02 -> 00:12] नमस्ते जी")


class TestFiles:
    def test_glossary_ignores_comments_and_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "glossary.txt"
        path.write_text("# names\nत्रिफला\n\n  अश्वगंधा  # a herb\n", encoding="utf-8")
        assert load_glossary(path) == ["त्रिफला", "अश्वगंधा"]
        assert load_glossary(None) == []

    def test_spellings_must_be_an_object_of_strings(self, tmp_path: Path) -> None:
        path = tmp_path / "words.json"
        path.write_text('{" त्रिफला ": " Triphala "}', encoding="utf-8")
        assert load_spellings(path) == {"त्रिफला": "Triphala"}
        path.write_text('["a"]', encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object"):
            load_spellings(path)

    def test_optional_file_prefers_the_explicit_path(self, tmp_path: Path) -> None:
        explicit, default = tmp_path / "mine.txt", tmp_path / "glossary.txt"
        assert optional_file(explicit, default) == explicit.resolve()
        assert optional_file(None, default) is None
        default.touch()
        assert optional_file(None, default) == default.resolve()


class TestCli:
    @pytest.fixture
    def workdir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "recordings").mkdir()
        return tmp_path

    def test_transcribes_the_recordings_folder_and_skips_done_files(
        self, workdir: Path, capsys: pytest.CaptureFixture
    ) -> None:
        write_silent_wav(workdir / "recordings" / "call-0001.wav")
        engine, pipeline = fake_engine()

        assert main(["--no-progress"], engine=engine) == 0
        out = workdir / "transcripts" / "call-0001.hinglish.txt"
        assert out.read_text(encoding="utf-8").splitlines() == [
            "[00:00 -> 00:04] namaste, aapka order kal tak pahunch jayega",
            "[00:04 -> 00:09] dawa din mein do baar lijiye",
        ]
        assert "[00:00 -> 00:04] namaste" in capsys.readouterr().out

        assert main(["--no-progress"], engine=engine) == 0  # already done: nothing is decoded again
        assert len(pipeline.calls) == 1
        assert main(["--no-progress", "--force"], engine=engine) == 0
        assert len(pipeline.calls) == 2

    def test_layer_formats_and_vocabulary_files(self, workdir: Path) -> None:
        write_silent_wav(workdir / "recordings" / "call-0001.wav")
        (workdir / "glossary.txt").write_text("त्रिफला\n", encoding="utf-8")
        (workdir / "custom_words.json").write_text('{"दवा": "medicine"}', encoding="utf-8")
        engine, pipeline = fake_engine()

        assert main(["--no-progress", "-q", "--layer", "script", "-f", "txt", "-f", "json"], engine=engine) == 0

        assert pipeline.calls[0]["hotwords"] == "त्रिफला"
        script = (workdir / "transcripts" / "call-0001.script.txt").read_text(encoding="utf-8")
        assert script.splitlines()[1] == "[00:04 -> 00:09] दवा दिन में दो बार लीजिए"
        data = json.loads((workdir / "transcripts" / "call-0001.likho.json").read_text(encoding="utf-8"))
        assert data["segments"][1]["text_roman"] == "medicine din mein do baar lijiye"

    def test_no_recordings_is_an_error_with_a_hint(self, workdir: Path, caplog: pytest.LogCaptureFixture) -> None:
        assert main(["--no-progress"], engine=fake_engine()[0]) == 1
        assert "No recordings found" in caplog.text

    def test_a_missing_input_or_bad_setting_exits_with_2(self, workdir: Path, caplog: pytest.LogCaptureFixture) -> None:
        assert main(["missing.wav"], engine=fake_engine()[0]) == 2
        assert "no such file or folder" in caplog.text
        write_silent_wav(workdir / "recordings" / "call.wav")
        assert main(["--chunk-seconds", "99"], engine=fake_engine()[0]) == 2
        assert "chunk_seconds must be between 1 and 30" in caplog.text

    def test_one_bad_recording_does_not_stop_the_others(self, workdir: Path) -> None:
        (workdir / "recordings" / "broken.wav").write_bytes(b"not audio")
        write_silent_wav(workdir / "recordings" / "good.wav")
        assert main(["--no-progress", "-q"], engine=fake_engine()[0]) == 1
        assert (workdir / "transcripts" / "good.hinglish.txt").is_file()
        assert not (workdir / "transcripts" / "broken.hinglish.txt").exists()
