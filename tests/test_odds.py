"""Unit tests for the odds math. Run: pytest -q

These are the tests a reviewer reads first -- if the probability math is right,
the rest of the pipeline can be trusted.
"""

import math

import pytest

from nfl_edge import odds


def approx(a, b, tol=1e-6):
    return math.isclose(a, b, abs_tol=tol)


# ---- american_to_prob -----------------------------------------------------
def test_american_to_prob_favorite():
    assert approx(odds.american_to_prob(-110), 110 / 210)   # ~0.52381
    assert approx(odds.american_to_prob(-150), 150 / 250)   # 0.6


def test_american_to_prob_underdog():
    assert approx(odds.american_to_prob(+150), 100 / 250)   # 0.4
    assert approx(odds.american_to_prob(+100), 0.5)


def test_american_to_prob_zero_raises():
    with pytest.raises(ValueError):
        odds.american_to_prob(0)


# ---- decimal conversion ---------------------------------------------------
def test_american_to_decimal():
    assert approx(odds.american_to_decimal(-110), 1 + 100 / 110)
    assert approx(odds.american_to_decimal(+150), 2.5)


def test_prob_to_american_roundtrip():
    for p in (0.4, 0.5238, 0.6, 0.75):
        a = odds.prob_to_american(p)
        assert approx(odds.american_to_prob(a), p, tol=1e-4)


# ---- devig ----------------------------------------------------------------
def test_devig_symmetric_market():
    fa, fb = odds.devig_two_way(-110, -110)
    assert approx(fa, 0.5) and approx(fb, 0.5)
    assert approx(fa + fb, 1.0)


def test_devig_asymmetric_market():
    fa, fb = odds.devig_two_way(-150, +130)
    assert approx(fa + fb, 1.0)
    assert fa > fb                      # favorite has higher fair prob
    # raw: 150/250=0.600 and 100/230=0.43478; normalized fav prob = 0.5798
    assert approx(fa, 0.5798, tol=1e-3)


def test_overround_positive():
    # standard -110/-110 book: each side implied 110/210, hold ~4.76%
    assert approx(odds.overround(-110, -110), 110 / 210 * 2 - 1)
    assert odds.overround(-110, -110) > 0.045


# ---- settlement -----------------------------------------------------------
def test_profit_units():
    assert approx(odds.profit_units(True, -110), 100 / 110)
    assert odds.profit_units(False, -110) == -1.0
    assert odds.profit_units(None, -110) == 0.0
    assert approx(odds.profit_units(True, +150), 1.5)


# ---- grading --------------------------------------------------------------
def test_grade_ats():
    # home favored by 3, wins by 7 -> home covers
    assert odds.grade_ats(result=7, spread_line=3) == "home"
    # home favored by 3, wins by exactly 3 -> push
    assert odds.grade_ats(result=3, spread_line=3) == "push"
    # home favored by 7, wins by 3 -> away covers
    assert odds.grade_ats(result=3, spread_line=7) == "away"
    # home dog (+3 => spread_line=-3), loses by 2 -> home covers
    assert odds.grade_ats(result=-2, spread_line=-3) == "home"


def test_grade_total():
    assert odds.grade_total(total=47, total_line=46) == "over"
    assert odds.grade_total(total=46, total_line=46) == "push"
    assert odds.grade_total(total=40, total_line=46) == "under"


def test_grade_moneyline():
    assert odds.grade_moneyline(7) == "home"
    assert odds.grade_moneyline(-7) == "away"
    assert odds.grade_moneyline(0) == "push"


# ---- CLV ------------------------------------------------------------------
def test_clv_prob_positive_when_beating_close():
    # took +3 dog at +145, closed at +120 -> we beat the close, CLV > 0
    assert odds.clv_prob(entry_odds=+145, close_odds=+120) > 0


def test_clv_cents():
    assert odds.clv_cents(entry_prob=0.50, close_prob=0.52) == 2.0
