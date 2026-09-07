"""Odds math: the numeric core of the whole project.

Everything downstream (backtest, edge engine, Kalshi comparison) depends on these
functions being correct, so they are pure, dependency-free, and unit-tested in
tests/test_odds.py.

Conventions used across the repo (verified against nflreadpy load_schedules):
    * spread_line > 0  => the HOME team is favored by that many points.
    * result           => home margin = home_score - away_score.
    * total            => actual combined points = home_score + away_score.
    * A bet of 1 unit is STAKED. A win returns (decimal_odds - 1) profit,
      a loss returns -1, a push returns 0. ROI = sum(profit) / sum(staked).
"""

from __future__ import annotations

import math
from typing import Optional


# --------------------------------------------------------------------------
# American odds <-> probability / decimal
# --------------------------------------------------------------------------
def american_to_prob(odds: float) -> float:
    """Implied win probability of a single American-odds price (includes vig).

    -110 -> 0.5238,  +150 -> 0.4000,  +100 -> 0.5000
    """
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def american_to_decimal(odds: float) -> float:
    """American odds -> decimal odds. -110 -> 1.9091, +150 -> 2.5000."""
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds < 0:
        return 1.0 + 100.0 / -odds
    return 1.0 + odds / 100.0


def prob_to_american(prob: float) -> float:
    """Fair (no-vig) probability -> American odds. 0.5 -> -100 (== +100)."""
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")
    if prob >= 0.5:
        return -round(100.0 * prob / (1.0 - prob), 2)
    return round(100.0 * (1.0 - prob) / prob, 2)


# --------------------------------------------------------------------------
# Vig removal -- the single most important calculation in the project.
# --------------------------------------------------------------------------
def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Strip the vig from a two-sided market, returning fair probabilities.

    Uses the multiplicative (normalization) method: divide each side's raw
    implied probability by the overround so the two fair probs sum to 1.

    -110 / -110  -> (0.5,   0.5)
    -150 / +130  -> (0.5769, 0.4231)  (favorite has the higher fair prob)
    """
    raw_a = american_to_prob(odds_a)
    raw_b = american_to_prob(odds_b)
    overround = raw_a + raw_b
    if overround <= 0:
        raise ValueError("degenerate market")
    return raw_a / overround, raw_b / overround


def overround(odds_a: float, odds_b: float) -> float:
    """The bookmaker's total vig on a two-sided market. -110/-110 -> ~0.0476."""
    return american_to_prob(odds_a) + american_to_prob(odds_b) - 1.0


# --------------------------------------------------------------------------
# Bet settlement
# --------------------------------------------------------------------------
def profit_units(won: Optional[bool], odds: float = -110.0) -> float:
    """Profit in units for a 1-unit stake.

    won is True (win), False (loss), or None (push -> 0).
    """
    if won is None:
        return 0.0
    if won:
        return american_to_decimal(odds) - 1.0
    return -1.0


# --------------------------------------------------------------------------
# Grading -- which side of each market won.
# Returns 'home'/'away' (spread & moneyline) or 'over'/'under' (total),
# or 'push'. result = home margin, total = combined points.
# --------------------------------------------------------------------------
def grade_ats(result: float, spread_line: float) -> str:
    """Against-the-spread winner. spread_line > 0 => home favored by that many.

    Home covers when the home margin beats the spread.
    """
    margin = result - spread_line
    if margin > 0:
        return "home"
    if margin < 0:
        return "away"
    return "push"


def grade_total(total: float, total_line: float) -> str:
    if total > total_line:
        return "over"
    if total < total_line:
        return "under"
    return "push"


def grade_moneyline(result: float) -> str:
    """Straight-up winner. NFL games cannot end tied in the data's result field
    except for the rare regular-season tie (result == 0)."""
    if result > 0:
        return "home"
    if result < 0:
        return "away"
    return "push"  # regular-season tie -> moneyline push


# --------------------------------------------------------------------------
# Closing Line Value (LIVE ops only -- see README).
# The backtest has one line per game (the close), so historical CLV cannot be
# computed. During the week you log the line you actually took, then compare it
# to the closing number with these helpers.
# --------------------------------------------------------------------------
def clv_prob(entry_odds: float, close_odds: float) -> float:
    """CLV as a probability delta: (fair prob at close) - (fair prob at entry)
    for the SAME side, priced as a one-sided vig-free estimate via the pair.

    Positive => you got a better number than the market closed at (good).
    Note: this treats each price one-sided; for a rigorous number devig against
    the paired side at both timestamps.
    """
    return american_to_prob(close_odds) - american_to_prob(entry_odds)


def clv_cents(entry_prob: float, close_prob: float) -> float:
    """CLV in probability 'cents' (percentage points). +2.0 == 2 points of edge
    captured vs. the close."""
    return round((close_prob - entry_prob) * 100.0, 2)
