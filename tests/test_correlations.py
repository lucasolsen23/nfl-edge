"""Tests for correlated-parlay math."""

from nfl_edge.props.correlations import (
    Leg, base_correlation, correlation_matrix, parlay_probability,
    independent_probability, evaluate_parlay,
)


def _leg(pid, team, opp, stat, side, prob):
    return Leg(pid, team, opp, stat, side, prob, 2026, 1)


def test_base_correlation_relationships():
    qb = _leg("qb", "CIN", "TB", "passing_yards", "OVER", 0.6)
    wr = _leg("wr", "CIN", "TB", "receiving_yards", "OVER", 0.6)
    wr_yds = _leg("wr", "CIN", "TB", "receiving_yards", "OVER", 0.6)
    wr_rec = _leg("wr", "CIN", "TB", "receptions", "OVER", 0.6)
    other = _leg("rb", "DAL", "NYG", "rushing_yards", "OVER", 0.6)
    assert base_correlation(qb, wr) > 0            # QB <-> his WR
    assert base_correlation(wr_yds, wr_rec) > 0.5  # same player yds<->rec (~0.75)
    assert base_correlation(qb, other) == 0.0      # different game


def test_direction_flips_correlation_sign():
    a = _leg("wr", "CIN", "TB", "receiving_yards", "OVER", 0.6)
    b_over = _leg("wr", "CIN", "TB", "receptions", "OVER", 0.6)
    b_under = _leg("wr", "CIN", "TB", "receptions", "UNDER", 0.6)
    assert correlation_matrix([a, b_over])[0, 1] > 0
    assert correlation_matrix([a, b_under])[0, 1] < 0    # opposite dirs -> negative


def test_positive_correlation_raises_joint_over_independent():
    a = _leg("wr", "CIN", "TB", "receiving_yards", "OVER", 0.6)
    b = _leg("wr", "CIN", "TB", "receptions", "OVER", 0.6)   # 0.75 corr, same dir
    joint = parlay_probability([a, b], n=60_000, seed=1)
    indep = independent_probability([a, b])
    assert joint > indep + 0.02                    # correlation clearly lifts it


def test_anticorrelated_same_dir_lowers_joint():
    # betting OVER on two positively-correlated stats is good; betting one OVER
    # and one UNDER is a trap -> joint below independent
    a = _leg("wr", "CIN", "TB", "receiving_yards", "OVER", 0.6)
    b = _leg("wr", "CIN", "TB", "receptions", "UNDER", 0.6)
    joint = parlay_probability([a, b], n=60_000, seed=1)
    assert joint < independent_probability([a, b])


def test_evaluate_parlay_ev():
    a = _leg("wr", "CIN", "TB", "receiving_yards", "OVER", 0.62)
    b = _leg("qb", "CIN", "TB", "passing_yards", "OVER", 0.62)
    ev = evaluate_parlay([a, b], payout=3.0)
    assert ev.breakeven == 1 / 3.0
    assert ev.lift > 0                             # QB-WR stack adds joint prob
