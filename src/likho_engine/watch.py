"""Watch a folder and hand over new recordings once they have finished copying."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

from likho_engine.audio import is_recording

log = logging.getLogger(__name__)


class FolderWatcher:
    """Polls a folder for new recordings.

    A file is reported only after its size has stayed the same between two polls, so a
    recording that is still being copied is not transcribed half-way. Each file is
    reported once; files for which is_done() is true (e.g. a transcript exists) are ignored.
    """

    def __init__(self, folder: Path, is_done: Callable[[Path], bool]) -> None:
        self.folder = folder
        self.is_done = is_done
        self._sizes: dict[Path, int] = {}
        self._reported: set[Path] = set()

    def poll(self) -> list[Path]:
        """Return the recordings that are new, complete and not done yet."""
        ready: list[Path] = []
        try:
            entries = sorted(self.folder.iterdir())
        except OSError as exc:
            log.warning("Cannot read %s: %s", self.folder, exc)
            return ready
        for path in entries:
            if path in self._reported or not is_recording(path) or self.is_done(path):
                continue
            size = path.stat().st_size
            if self._sizes.get(path) == size:
                ready.append(path)
                self._reported.add(path)
            else:
                self._sizes[path] = size
        return ready


def watch(
    folder: Path,
    is_done: Callable[[Path], bool],
    handle: Callable[[list[Path]], None],
    poll_seconds: float = 10.0,
    stop: Callable[[], bool] | None = None,
) -> None:
    """Poll folder until stop() is true (or Ctrl+C), passing each batch of new recordings to handle()."""
    watcher = FolderWatcher(folder, is_done)
    log.info("Watching %s for new recordings every %.0fs (Ctrl+C to stop)...", folder, poll_seconds)
    while stop is None or not stop():
        ready = watcher.poll()
        if ready:
            handle(ready)
        time.sleep(poll_seconds)
