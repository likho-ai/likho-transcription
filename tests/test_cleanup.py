"""Tests for transcript cleanup."""

import pytest

from likho_engine.cleanup import collapse_repeats


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # a looping phrase is cut to one occurrence
        (
            "hello, theek hai ji, theek hai ji, theek hai ji, theek hai ji, to",
            "hello, theek hai ji, to",
        ),
        # a phrase said twice is natural speech and stays
        ("theek hai theek hai bhai", "theek hai theek hai bhai"),
        # single words: three in a row stay, more are cut to two
        ("ji ji ji", "ji ji ji"),
        ("ji ji ji ji ji ji ji", "ji ji"),
        # nothing to do
        ("aap kaise hain", "aap kaise hain"),
        # Devanagari, with the danda counted as punctuation
        ("ठीक है। ठीक है। ठीक है। ठीक है। अच्छा", "ठीक है। अच्छा"),
        ("जी जी जी जी जी", "जी जी"),
        ("", ""),
    ],
)
def test_collapse_repeats(text: str, expected: str) -> None:
    assert collapse_repeats(text) == expected
