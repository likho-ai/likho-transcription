"""Text cleanup applied to every transcript line.

A speech model sometimes gets stuck and repeats a phrase over and over ("theek hai ji, theek
hai ji, theek hai ji, ..."). collapse_repeats() trims such runs while leaving natural
repetition ("haan haan", "theek hai theek hai") alone.
"""

MAX_PHRASE_WORDS = 8
_PUNCTUATION = ",.?!;:\"'।॥"  # ASCII marks and the Devanagari danda


def _normalise(token: str) -> str:
    return token.strip(_PUNCTUATION).lower()


def _is_periodic(phrase: list[str]) -> bool:
    """True when the phrase is itself a shorter unit repeated ("ji ji", "a b a b")."""
    n = len(phrase)
    return any(n % d == 0 and phrase == phrase[:d] * (n // d) for d in range(1, n))


def collapse_repeats(text: str) -> str:
    """Collapse a word repeated 4+ times in a row to 2, and a phrase repeated 3+ times to 1."""
    tokens = text.split()
    norm = [_normalise(t) for t in tokens]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        for n in range(min(MAX_PHRASE_WORDS, len(tokens) - i), 0, -1):  # longest phrase first
            phrase = norm[i : i + n]
            if _is_periodic(phrase):
                continue  # let the shorter unit handle it
            repeats = 1
            while norm[i + repeats * n : i + (repeats + 1) * n] == phrase:
                repeats += 1
            keep, limit = (2, 4) if n == 1 else (1, 3)
            if repeats >= limit:
                out.extend(tokens[i : i + keep * n])
                i += repeats * n
                break
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)
