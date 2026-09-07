"""Unit tests for the edge engine's money math (fees, Kelly, scan)."""

import math

from nfl_edge.edge import fees, scan
from nfl_edge.edge.kalshi import KalshiMarket


def approx(a, b, tol=1e-9):
    return math.isclose(a, b, abs_tol=tol)


# ---- fees -----------------------------------------------------------------
def test_fee_peaks_at_midprice():
    # fee is largest near 0.50 and near-zero at the extremes
    assert fees.fee_per_contract(0.50) >= fees.fee_per_contract(0.90)
    assert fees.fee_per_contract(0.50) >= fees.fee_per_contract(0.10)


def test_fee_rounds_up_to_cent():
    # 0.07 * 0.75 * 0.25 = 0.0131 -> rounds up to 0.02
    assert approx(fees.fee_per_contract(0.75), 0.02)


def test_fee_zero_out_of_range():
    assert fees.fee_per_contract(None) == 0.0
    assert fees.fee_per_contract(1.0) == 0.0
    assert fees.fee_per_contract(0.0) == 0.0


def test_total_cost():
    assert approx(fees.total_cost(0.75), 0.77)


# ---- kelly ----------------------------------------------------------------
def test_kelly_positive_edge():
    # fair 0.82, cost 0.77 -> (0.82-0.77)/(1-0.77) = 0.2174
    assert approx(scan.kelly_fraction(0.82, 0.77), 0.05 / 0.23, tol=1e-4)


def test_kelly_zero_when_no_edge():
    assert scan.kelly_fraction(0.70, 0.77) == 0.0        # negative edge -> 0
    assert scan.kelly_fraction(0.50, 1.0) == 0.0         # cost >= 1 -> 0


# ---- scan ----------------------------------------------------------------
def _mkt(event, code, ask):
    return KalshiMarket(ticker=f"{event}-{code}", event_ticker=event, team_code=code,
                        team_name=code, yes_ask=ask, yes_bid=ask - 0.01,
                        last_price=ask, close_time="2026-09-23T00:20:00Z",
                        status="open", volume=10.0)


def test_scan_finds_and_thresholds_edges():
    mk = [_mkt("E1", "KC", 0.75), _mkt("E1", "IND", 0.26)]
    book = {frozenset({"KC", "IND"}): {"probs": {"KC": 0.82, "IND": 0.18},
                                       "n_books": 6, "commence_time": "",
                                       "home": "KC", "away": "IND"}}
    rows = scan.scan(mk, book, min_edge=0.03)
    assert len(rows) == 1                 # only KC clears +3% (IND is -EV)
    assert rows[0].team == "KC"
    assert approx(rows[0].edge, 0.05, tol=1e-9)
    # raising the threshold above the edge removes it
    assert scan.scan(mk, book, min_edge=0.06) == []


def test_scan_skips_unmatched_games():
    mk = [_mkt("E1", "KC", 0.75), _mkt("E1", "IND", 0.26)]
    assert scan.scan(mk, {}, min_edge=0.0) == []   # no book price -> no rows
