"""Tests for defensive-player props (tackles normal, sacks Poisson)."""

import math

import polars as pl

from nfl_edge.props import defense as D


def approx(a, b, tol=1e-3):
    return math.isclose(a, b, abs_tol=tol)


def test_sacks_poisson_prob():
    # P(>=1 sack | lambda=0.5) = 1 - e^-0.5 = 0.3935
    assert approx(D.prob_over("sacks", 0.5, 0.7, 0.5), 1 - math.exp(-0.5))
    # P(>=2 | lambda=1.0) = 1 - e^-1(1+1) = 0.2642
    assert approx(D.prob_over("sacks", 1.0, 1.0, 1.5), 1 - math.exp(-1.0) * 2)
    # monotonic: higher line -> lower prob
    assert D.prob_over("sacks", 0.8, 0.9, 0.5) > D.prob_over("sacks", 0.8, 0.9, 1.5)


def test_tackles_normal_prob():
    # at the mean, exactly 0.5; higher line -> lower
    assert approx(D.prob_over("tackles", 7.0, 2.5, 7.0), 0.5)
    assert D.prob_over("tackles", 7.0, 2.5, 5.5) > D.prob_over("tackles", 7.0, 2.5, 9.5)


def test_poisson_cdf():
    assert approx(D._poisson_cdf(0, 0.5), math.exp(-0.5))


def _fake_def_proj() -> pl.DataFrame:
    return pl.DataFrame({
        "player_id": ["d1", "d2"],
        "player_display_name": ["Roquan Smith", "Micah Parsons"],
        "position_group": ["LB", "DL"],
        "team2026": ["BAL", "DAL"],
        "opponent_team": ["CIN", "NYG"],
        "is_home": [True, False],
        "gameday": ["2026-09-13", "2026-09-14"],
        "season": [2026, 2026], "week": [1, 1],
        "stat": ["tackles", "sacks"],
        "opp_factor": [1.05, 1.2], "note": ["", "favorable matchup"],
        "proj_mean": [9.0, 0.8], "proj_sd": [2.5, 0.9],
    })


def test_score_def_lines():
    lines = [
        {"player": "Roquan Smith", "stat": "tackles", "line": 7.5},   # proj 9 -> OVER
        {"player": "Micah Parsons", "stat": "sacks", "line": 0.5},    # lam .8 -> OVER
    ]
    edges = D.score_def_lines(lines, _fake_def_proj())
    assert len(edges) == 2
    by = {e.player: e for e in edges}
    assert by["Roquan Smith"].pick == "OVER"
    assert by["Micah Parsons"].pick == "OVER"
    assert approx(by["Micah Parsons"].prob, 1 - math.exp(-0.8), tol=1e-3)
