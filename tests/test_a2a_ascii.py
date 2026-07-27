"""The A2A brief must survive the trip to the buyer's file.

A delivered deliverable.txt was downloaded as the buyer and read back. It decoded cleanly as UTF-8
and contained the literal characters "â€”" where an em dash had been written: the UTF-8 bytes were
decoded as cp1252 somewhere between this service and the file, then re-encoded. The buyer opens the
artifact they paid for and sees corrupted punctuation.

That encoding step is not ours to fix. Not depending on it is.
"""
from __future__ import annotations

import pytest

from services.a2a import to_ascii_punctuation


@pytest.mark.parametrize(("given", "expected"), [
    ("funding is ordinary — not unusual", "funding is ordinary -- not unusual"),
    ("the model's own claim", "the model's own claim"),
    ("he said “calibrated”", 'he said "calibrated"'),
    ("range 1–2", "range 1-2"),
    ("and so on…", "and so on..."),
    ("2 × 3", "2 x 3"),
    ("a → b", "a -> b"),
    ("n ≥ 20", "n >= 20"),
])
def test_typographic_punctuation_becomes_ascii(given, expected):
    assert to_ascii_punctuation(given) == expected


def test_it_reaches_into_the_whole_brief():
    brief = {
        "summary": "funding is ordinary — see below",
        "findings": [{"finding": "band is wide — 3.5%", "endpoint": "forecast.range"}],
        "caveats": ["only 128 paths — read as rare"],
    }
    out = to_ascii_punctuation(brief)
    assert "—" not in str(out)
    assert out["findings"][0]["endpoint"] == "forecast.range"


def test_the_result_is_encodable_as_plain_ascii():
    """The property that actually matters: nothing left that a byte-level round trip can mangle."""
    out = to_ascii_punctuation("bands — 'raw' “quantile” output… 2 × 128 paths, n ≥ 20")
    out.encode("ascii")          # raises if anything typographic survived


def test_numbers_and_ordinary_text_are_untouched():
    text = "80% band 63772.78 to 65289.29 (width_pct 2.3264) [forecast.range]"
    assert to_ascii_punctuation(text) == text
