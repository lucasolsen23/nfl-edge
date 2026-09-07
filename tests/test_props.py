"""Tests for the player-props projection + scanner."""

import polars as pl

from nfl_edge.props import projections as P
from nfl_edge.props import scan_props as S
from nfl_edge.props import live as L


def test_prob_over_normal_symmetric():
    # no stat -> normal: at the mean it's exactly 0.5
    assert abs(P.prob_over(50, 20, 50) - 0.5) < 1e-9
    # higher line -> lower P(over)
    assert P.prob_over(50, 20, 40) > P.prob_over(50, 20, 60)


def test_prob_over_empirical_is_skewed():
    # empirical calibration should be loaded and right-skewed:
    # P(over the mean) < 0.5 for receiving yards (median below mean)
    assert P.using_empirical()
    assert P.prob_over(50, 20, 50, stat="receiving_yards") < 0.5


def test_prob_over_monotonic_in_line():
    lo = P.prob_over(60, 25, 40, stat="rushing_yards")
    hi = P.prob_over(60, 25, 90, stat="rushing_yards")
    assert lo > hi and 0.0 < hi < lo < 1.0


def _fake_projections() -> pl.DataFrame:
    return pl.DataFrame({
        "player_id": ["p1", "p1", "p2"],
        "player_display_name": ["Al Star", "Al Star", "Bo Back"],
        "stat": ["receiving_yards", "receiving_yards", "rushing_yards"],
        "proj_mean": [70.0, 80.0, 55.0],       # p1 has two weeks; latest = 80
        "proj_sd": [20.0, 20.0, 18.0],
        "opponent_team": ["MIA", "BUF", "DAL"],
        "season": [2025, 2025, 2025],
        "week": [3, 4, 4],
    })


def test_score_lines_picks_side_and_uses_latest():
    lines = [
        {"player": "Al Star", "stat": "receiving_yards", "line": 55.5},  # proj 80 -> OVER
        {"player": "Bo Back", "stat": "rushing_yards", "line": 90.5},    # proj 55 -> UNDER
        {"player": "Ghost", "stat": "receiving_yards", "line": 10.5},    # no match -> skip
    ]
    edges = S.score_lines(lines, _fake_projections())
    assert len(edges) == 2
    by_player = {e.player: e for e in edges}
    assert by_player["Al Star"].pick == "OVER"
    assert by_player["Al Star"].proj_mean == 80.0      # latest week used, not 70
    assert by_player["Bo Back"].pick == "UNDER"
    # ranked by confidence descending
    assert edges[0].prob >= edges[1].prob


def test_upcoming_matchups_two_sided_and_normalized():
    sched = pl.DataFrame({
        "season": [2026], "week": [1], "game_type": ["REG"],
        "home_team": ["KC"], "away_team": ["LA"],   # 'LA' should normalize to LAR
        "gameday": ["2026-09-13"], "result": [None],
    })
    m = L.upcoming_matchups(sched, 2026, 1)
    d = {r["team"]: r for r in m.to_dicts()}
    assert set(d) == {"KC", "LAR"}                 # LA -> LAR
    assert d["KC"]["opponent"] == "LAR" and d["KC"]["is_home"] is True
    assert d["LAR"]["opponent"] == "KC" and d["LAR"]["is_home"] is False
