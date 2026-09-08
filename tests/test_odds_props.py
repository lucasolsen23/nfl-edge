"""Tests for the auto book-props source (pure parsing + market-anchored scoring)."""

import polars as pl

from nfl_edge.props import odds_props as O


def _book(point_over=("Over", -110, 50.5), point_under=("Under", -110, 50.5),
          player="Al Star", market="player_reception_yds"):
    return {"markets": [{"key": market, "outcomes": [
        {"name": point_over[0], "description": player, "price": point_over[1], "point": point_over[2]},
        {"name": point_under[0], "description": player, "price": point_under[1], "point": point_under[2]},
    ]}]}


def test_consensus_parse_line_and_fair():
    lines = O.consensus_from_bookmakers([_book(), _book()])
    assert len(lines) == 1
    ln = lines[0]
    assert ln["stat"] == "receiving_yards" and ln["line"] == 50.5
    assert abs(ln["book_fair_over"] - 0.5) < 0.02      # -110/-110 -> ~0.5 fair
    assert ln["n_books"] == 2


def test_consensus_uses_modal_line():
    # two books at 50.5, one at 48.5 -> consensus 50.5
    books = [_book(), _book(),
             _book(point_over=("Over", -110, 48.5), point_under=("Under", -110, 48.5))]
    ln = O.consensus_from_bookmakers(books)[0]
    assert ln["line"] == 50.5 and ln["n_books"] == 2


def _proj_df():
    return pl.DataFrame({
        "player_id": ["p1", "p2"],
        "player_display_name": ["Al Star", "Bo Bench"],
        "stat": ["receiving_yards", "receiving_yards"],
        "proj_mean": [80.0, 200.0],      # p2 is absurd vs a 70.5 line -> filtered
        "proj_sd": [20.0, 25.0],
        "opponent_team": ["MIA", "BUF"], "season": [2026, 2026], "week": [1, 1],
        "note": ["", ""], "opp_factor": [1.0, 1.0], "team2026": ["CIN", "DAL"],
    })


def test_score_blends_toward_book_and_filters_implausible():
    lines = [
        {"player": "Al Star", "stat": "receiving_yards", "line": 70.5, "book_fair_over": 0.5},
        {"player": "Bo Bench", "stat": "receiving_yards", "line": 70.5, "book_fair_over": 0.5},
    ]
    edges = O.score_slate(lines, _proj_df(), pl.DataFrame())
    # Bo Bench (200 proj vs 70.5) disagrees wildly -> dropped as model error
    assert [e.player for e in edges] == ["Al Star"]
    e = edges[0]
    # blended edge is modest (< the raw model-vs-book gap), and OVER (proj>line)
    assert e.pick == "OVER" and 0.0 < e.mkt_edge < 0.15
