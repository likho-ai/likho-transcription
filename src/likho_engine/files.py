"""Optional files for the command line: a glossary of names and a table of Hinglish spellings."""

import json
from pathlib import Path

# Picked up automatically from the folder the command runs in, when they exist.
DEFAULT_GLOSSARY = Path("glossary.txt")
DEFAULT_SPELLINGS = Path("custom_words.json")


def optional_file(explicit: Path | None, default: Path) -> Path | None:
    """An explicitly chosen file, else the default one if it exists, else None."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    default = default.resolve()
    return default if default.is_file() else None


def load_glossary(path: Path | None) -> list[str]:
    """Read names from a text file, one per line. Blank lines and '#' comments are ignored."""
    if path is None:
        return []
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def load_spellings(path: Path | None) -> dict[str, str]:
    """Read a JSON object of {"देवनागरी": "hinglish"} spellings."""
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f'{path}: expected a JSON object of {{"देवनागरी": "hinglish"}} pairs')
    return {key.strip(): value.strip() for key, value in data.items()}
