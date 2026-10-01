"""Tests for the folder watcher."""

from pathlib import Path

from likho_engine.watch import FolderWatcher, watch


def test_file_is_reported_once_its_size_is_stable(tmp_path: Path) -> None:
    watcher = FolderWatcher(tmp_path, is_done=lambda p: False)
    call = tmp_path / "call.wav"
    call.write_bytes(b"x" * 10)

    assert watcher.poll() == []  # first sighting: size not yet confirmed stable
    call.write_bytes(b"x" * 20)  # still being copied
    assert watcher.poll() == []
    assert watcher.poll() == [call]  # same size twice in a row
    assert watcher.poll() == []  # reported only once


def test_non_recordings_and_done_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "done.wav").write_bytes(b"x")
    (tmp_path / "new.wav").write_bytes(b"x")
    watcher = FolderWatcher(tmp_path, is_done=lambda p: p.name == "done.wav")

    watcher.poll()
    assert watcher.poll() == [tmp_path / "new.wav"]


def test_missing_folder_is_not_fatal(tmp_path: Path) -> None:
    watcher = FolderWatcher(tmp_path / "gone", is_done=lambda p: False)
    assert watcher.poll() == []


def test_watch_loop_hands_over_batches_until_stopped(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"x")
    handled: list[list[Path]] = []
    polls = 0

    def stop() -> bool:
        nonlocal polls
        polls += 1
        return polls > 3

    watch(tmp_path, is_done=lambda p: False, handle=handled.append, poll_seconds=0, stop=stop)

    assert handled == [[tmp_path / "a.wav"]]
