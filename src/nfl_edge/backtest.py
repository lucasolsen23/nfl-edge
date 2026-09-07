"""Backtest harness -- step 3.

Replays completed seasons game-by-game in chronological order and grades a
strategy in UNITS against the closing line. Deliberately built and validated
against DUMB baselines first: if a baseline that should be ~break-even shows a
real edge here, that's a leakage bug, not alpha.

Design notes
------------
* State (team point differential) is updated only AFTER a game is graded, so
  every pick is made with information that existed before kickoff -- the
  walk-forward discipline that keeps the backtest honest.
* We grade against the closing spread/total. We do NOT compute CLV here: the
  data has one line per game (the close), so there is no entry-vs-close delta to
  measure. CLV is a live-ops metric (see README).
* Staking is flat: 1 unit per bet. ROI = total profit / total staked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import polars as pl

from . import odds
from .ingest import read_cached_schedules

# A pick is which side to bet, or None to pass.
#   ats   strategies return 'home' | 'away' | None
#   total strategies return 'over' | 'under' | None
Pick = Optional[str]


# --------------------------------------------------------------------------
# Walk-forward state: cumulative point differential per team, reset each season.
# --------------------------------------------------------------------------
class SeasonState:
    def __init__(self) -> None:
        self.pf: dict[str, int] = defaultdict(int)   # points for
        self.pa: dict[str, int] = defaultdict(int)   # points against
        self.gp: dict[str, int] = defaultdict(int)   # games played

    def point_diff(self, team: str) -> float:
        if self.gp[team] == 0:
            return 0.0
        return (self.pf[team] - self.pa[team]) / self.gp[team]

    def update(self, home: str, away: str, hs: int, as_: int) -> None:
        self.pf[home] += hs; self.pa[home] += as_; self.gp[home] += 1
        self.pf[away] += as_; self.pa[away] += hs; self.gp[away] += 1


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------
@dataclass
class Strategy:
    name: str
    market: str                       # 'ats' or 'total'
    pick_fn: Callable[[dict, SeasonState], Pick]

    def pick(self, game: dict, state: SeasonState) -> Pick:
        return self.pick_fn(game, state)


def _always(side: str) -> Callable[[dict, SeasonState], Pick]:
    return lambda game, state: side


def _home_dog(game: dict, state: SeasonState) -> Pick:
    # bet the home team only when it's an underdog (spread_line < 0)
    return "home" if (game["spread_line"] is not None and game["spread_line"] < 0) else None


def _better_diff_ats(game: dict, state: SeasonState) -> Pick:
    """Bet the team with the better season-to-date point differential, ATS.

    Skip until both teams have played at least 3 games (small-sample noise).
    Uses only prior-game info via SeasonState -> no leakage.
    """
    h, a = game["home_team"], game["away_team"]
    if state.gp[h] < 3 or state.gp[a] < 3:
        return None
    dh, da = state.point_diff(h), state.point_diff(a)
    if dh == da:
        return None
    return "home" if dh > da else "away"


def model_ats_strategy(
    pred_map: dict[str, float], threshold: float = 1.0, name: str | None = None
) -> Strategy:
    """Bet ATS when the model's predicted home margin disagrees with the spread
    by more than `threshold` points.

    pred_map: game_id -> predicted home margin (out-of-sample).
    spread_line > 0 => home favored by that many, so the model's implied edge on
    the home side is (pred_margin - spread_line).
    """
    def pick(game: dict, state: SeasonState) -> Pick:
        pm = pred_map.get(game["game_id"])
        if pm is None:
            return None
        edge = pm - game["spread_line"]
        if edge > threshold:
            return "home"
        if edge < -threshold:
            return "away"
        return None

    return Strategy(name or f"model_ats_thr{threshold:g}", "ats", pick)


BASELINES: list[Strategy] = [
    Strategy("always_home_ats", "ats", _always("home")),
    Strategy("always_away_ats", "ats", _always("away")),
    Strategy("home_underdog_ats", "ats", _home_dog),
    Strategy("better_ppg_diff_ats", "ats", _better_diff_ats),
    Strategy("always_over", "total", _always("over")),
    Strategy("always_under", "total", _always("under")),
]


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------
@dataclass
class Result:
    strategy: str
    market: str
    bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    units: float = 0.0
    staked: float = 0.0
    _clv_samples: list = field(default_factory=list)

    @property
    def win_pct(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0

    @property
    def roi(self) -> float:
        return self.units / self.staked if self.staked else 0.0

    def row(self) -> dict:
        return {
            "strategy": self.strategy,
            "market": self.market,
            "bets": self.bets,
            "wins": self.wins,
            "losses": self.losses,
            "pushes": self.pushes,
            "win_pct": round(self.win_pct, 4),
            "units": round(self.units, 2),
            "roi": round(self.roi, 4),
        }


def _ats_odds(game: dict, side: str) -> float:
    o = game.get("home_spread_odds") if side == "home" else game.get("away_spread_odds")
    return float(o) if o is not None else -110.0


def _total_odds(game: dict, side: str) -> float:
    o = game.get("over_odds") if side == "over" else game.get("under_odds")
    return float(o) if o is not None else -110.0


def run(
    seasons: range | list[int],
    strategies: list[Strategy] = BASELINES,
    game_types: tuple[str, ...] = ("REG",),
) -> list[Result]:
    df = read_cached_schedules()
    df = df.filter(
        pl.col("season").is_in(list(seasons))
        & pl.col("game_type").is_in(list(game_types))
        & pl.col("result").is_not_null()
    ).sort(["season", "week", "gameday", "gametime"])

    results = {s.name: Result(s.name, s.market) for s in strategies}
    states: dict[int, SeasonState] = {}

    for game in df.iter_rows(named=True):
        season = game["season"]
        state = states.setdefault(season, SeasonState())

        for strat in strategies:
            if game["spread_line"] is None or game["total_line"] is None:
                continue
            pick = strat.pick(game, state)
            if pick is None:
                continue
            if strat.market == "ats":
                winner = odds.grade_ats(game["result"], game["spread_line"])
                price = _ats_odds(game, pick)
            else:
                winner = odds.grade_total(game["total"], game["total_line"])
                price = _total_odds(game, pick)

            r = results[strat.name]
            r.bets += 1
            r.staked += 1.0
            if winner == "push":
                r.pushes += 1
                r.units += odds.profit_units(None, price)
            elif winner == pick:
                r.wins += 1
                r.units += odds.profit_units(True, price)
            else:
                r.losses += 1
                r.units += odds.profit_units(False, price)

        # walk-forward: update AFTER grading
        state.update(game["home_team"], game["away_team"],
                     game["home_score"], game["away_score"])

    return list(results.values())


def report(results: list[Result]) -> pl.DataFrame:
    return pl.DataFrame([r.row() for r in results])
