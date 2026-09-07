"""Tests for the player-props projection + scanner."""

import polars as pl

from nfl_edge.props import projections as P
from nfl_edge.props import scan_props as S
from nfl_edge.props import live as L
from nfl_edge.props import context as C


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


def test_pressure_and_pass_factor():
    def_rel = {"SF": 1.4, "MIA": 0.7}      # SF strong rush, MIA weak
    off_rel = {"ARI": 1.2, "PHI": 0.8}     # ARI leaky O-line, PHI good
    tough = C.matchup_pressure("ARI", "SF", def_rel, off_rel)   # leaky vs strong
    easy = C.matchup_pressure("PHI", "MIA", def_rel, off_rel)   # good vs weak
    assert tough > 1.0 > easy
    # more pressure -> fewer passing yards, clamped
    assert C.pass_yards_pressure_factor(tough) < 1.0
    assert C.pass_yards_pressure_factor(easy) > 1.0
    assert 0.90 <= C.pass_yards_pressure_factor(5.0) <= 1.10   # clamp holds


def test_matchup_note():
    assert "favorable" in C.matchup_note(1.15, None)
    assert "tough" in C.matchup_note(0.85, None)
    assert "sack risk" in C.matchup_note(1.0, 1.3)
    assert C.matchup_note(1.0, 1.0) == ""


def test_injuries_out_graceful_for_future_season():
    # 2026 injuries aren't published yet -> empty set, no crash
    assert C.injuries_out(2026, 1) == set()
