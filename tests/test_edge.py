"""Unit tests for the edge engine's money math (fees, Kelly, scan)."""

import math

from nfl_edge.edge import dist, fees, scan
from nfl_edge.edge.kalshi import KalshiMarket, KalshiLadderMarket


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


# ---- distribution pricer --------------------------------------------------
def test_prob_over_at_mean_is_half():
    assert approx(dist.prob_over(0.0, 0.0, 13.0), 0.5)
    assert approx(dist.prob_total_over(47.0, 47.0), 0.5)


def test_prob_over_monotonic_in_strike():
    p_low = dist.prob_margin_over(7.0, 2.5)
    p_high = dist.prob_margin_over(7.0, 10.5)
    assert p_low > p_high            # harder to clear a bigger number
    assert 0.0 < p_high < p_low < 1.0


def test_prob_over_zero_sd_is_step():
    assert dist.prob_over(5.0, 3.0, 0.0) == 1.0
    assert dist.prob_over(2.0, 3.0, 0.0) == 0.0


# ---- spread & total scans -------------------------------------------------
def _ladder(event, kind, strike, ask, team=""):
    return KalshiLadderMarket(ticker=f"{event}-{team}{strike}", event_ticker=event,
                              kind=kind, team_code=team, floor_strike=strike,
                              yes_ask=ask, yes_bid=ask - 0.01, last_price=ask,
                              close_time="2026-09-14T17:00:00Z", status="open",
                              volume=5.0)


_CONS = {
    frozenset({"KC", "DEN"}): {
        "probs": {"KC": 0.80, "DEN": 0.20},
        "margin": {"KC": 7.0, "DEN": -7.0},
        "total": 47.0, "n_books": 5,
        "home": "KC", "away": "DEN", "commence_time": "",
    }
}


def test_scan_spreads_prices_and_matches():
    mk = [_ladder("KXNFLSPREAD-26SEP14DENKC", "spread", 3.5, 0.43, team="KC")]
    rows = scan.scan_spreads(mk, _CONS, min_edge=0.03)
    assert len(rows) == 1
    r = rows[0]
    assert r.market == "SPREAD" and r.team == "KC" and r.line == 3.5
    # P(margin>3.5 | mean 7, sd 12.7) ~ 0.61; cost ~0.45 -> clear positive edge
    assert 0.55 < r.fair_prob < 0.66
    assert r.edge > 0.10


def test_scan_totals_prices_and_thresholds():
    # Over 44.5 with mean total 47 -> ~0.57 fair, cost ~0.57 -> tiny edge, excluded
    tight = scan.scan_totals([_ladder("KXNFLTOTAL-26SEP14DENKC", "total", 44.5, 0.55)],
                             _CONS, min_edge=0.03)
    assert tight == []
    # a cheap Over is +EV
    cheap = scan.scan_totals([_ladder("KXNFLTOTAL-26SEP14DENKC", "total", 44.5, 0.45)],
                             _CONS, min_edge=0.03)
    assert len(cheap) == 1 and cheap[0].market == "TOTAL" and cheap[0].line == 44.5
