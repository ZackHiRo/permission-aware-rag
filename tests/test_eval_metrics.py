"""The evaluation's own metrics are code too: a leak alarm with false positives gets ignored."""

import pytest

from rag.evaluate import mcnemar_exact, mentions


@pytest.mark.parametrize("text,value,expected", [
    ("I can't find the 2024 band B7 parental leave guide.", "7 weeks", False),   # a real false positive we hit
    ("The top-up lasts 7 weeks.", "7 weeks", True),
    ("The threshold is CAD 121,250; anything above it escalates.", "CAD 121,250", True),
    ("The threshold is CAD 1121,250.", "CAD 121,250", False),
    ("The surcharge is 12.4% of the premium.", "12.4%", True),
    ("The surcharge is 12.45%.", "12.4%", False),
    ("Rotate every 37 days.", "37 days", True),
    ("See the 2037 plan.", "37 days", False),
])
def test_value_matching_is_precise(text, value, expected):
    assert mentions(text, value) is expected


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 2) == pytest.approx(0.453125)    # 7 discordant pairs: not significant
    assert mcnemar_exact(0, 10) < 0.01
