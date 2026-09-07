"""Live projections for the upcoming NFL week (2026 season).

Ties the historical projection model to this week's real games:
  1. each player's CURRENT form (last N games played, through the latest season)
  2. their CURRENT team, from the 2026 roster (handles offseason moves)
  3. their UPCOMING opponent, from the 2026 schedule
  4. that opponent's defensive factor vs their position (season-to-date; ~1.0
     early in the year before 2026 defensive data exists)

Output matches the columns scan_props expects, so prop lines can be scored
directly against this week's projections.

Early-season caveat: before 2026 games are played, "current form" is last
season's, and opponent factors are ~1.0 -- projections sharpen as the year goes.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import projections as P
from . import context as C
from ..edge.teams import norm_abbr

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "data" / "raw"

SEASON = 2026
PRIOR_SEASON = 2025          # defense/pass-rush prior until current season accrues


def _pass_rush(season: int) -> tuple[dict, dict]:
    """Load just the dropback rows we need from cached pbp and build pass-rush
    indices for `season` (falls back to the prior season if empty)."""
    path = RAW / "pbp.parquet"
    if not path.exists():
        return {}, {}
    for yr in (season, PRIOR_SEASON):
        d = (pl.scan_parquet(path)
             .filter((pl.col("season") == yr) & (pl.col("qb_dropback") == 1))
             .select(["season", "defteam", "posteam", "sack", "qb_dropback"])
             .collect())
        if d.height:
            return C.pass_rush_indices(d, yr)
    return {}, {}


def _load():
    ps = pl.read_parquet(RAW / "player_stats.parquet")
    sched = pl.read_parquet(RAW / "schedules.parquet")
    ros = pl.read_parquet(RAW / "rosters_2026.parquet")
    return ps, sched, ros


def upcoming_week(sched: pl.DataFrame, season: int = SEASON) -> int | None:
    up = sched.filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
        & pl.col("result").is_null()
    )
    return int(up["week"].min()) if up.height else None


def upcoming_matchups(sched: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """team -> (opponent, is_home, gameday) for the given week."""
    g = sched.filter((pl.col("season") == season) & (pl.col("week") == week))
    home = g.select(team="home_team", opponent="away_team",
                    gameday="gameday").with_columns(is_home=pl.lit(True))
    away = g.select(team="away_team", opponent="home_team",
                    gameday="gameday").with_columns(is_home=pl.lit(False))
    both = pl.concat([home, away])
    return both.with_columns(
        pl.col("team").map_elements(norm_abbr, return_dtype=pl.String),
        pl.col("opponent").map_elements(norm_abbr, return_dtype=pl.String),
    )


def current_form(ps: pl.DataFrame, stat: str, positions: tuple[str, ...],
                 window: int = P.ROLL_WINDOW) -> pl.DataFrame:
    """Each player's last `window` games (inclusive): mean, SD, position group."""
    rel = ps.filter(
        (pl.col("season_type") == "REG")
        & pl.col("position_group").is_in(list(positions))
    ).sort(["player_id", "season", "week"])
    rel = rel.with_columns(
        pl.col("week").cum_count(reverse=True).over("player_id").alias("_rn")
    ).filter(pl.col("_rn") <= window)
    agg = rel.group_by("player_id").agg(
        form_mean=pl.col(stat).mean(),
        form_sd=pl.col(stat).std(),
        position_group=pl.col("position_group").last(),
        games=pl.len(),
    ).filter(pl.col("games") >= P.ROLL_MIN)
    pos_sd = agg.group_by("position_group").agg(pl.col("form_sd").median().alias("pos_sd"))
    return agg.join(pos_sd, on="position_group", how="left").with_columns(
        pl.max_horizontal(
            pl.col("form_sd").fill_null(pl.col("pos_sd")),
            pl.col("pos_sd") * 0.5, pl.lit(1.0)
        ).alias("form_sd")
    )


def defense_factors(ps: pl.DataFrame, stat: str, positions: tuple[str, ...],
                    season: int) -> dict:
    """(defense, position_group) -> season-to-date factor for `season`. Empty
    (all ~1.0) until that season has games."""
    rel = ps.filter(
        (pl.col("season") == season) & (pl.col("season_type") == "REG")
        & pl.col("position_group").is_in(list(positions))
    )
    if rel.height == 0:
        return {}
    dg = rel.group_by(["opponent_team", "position_group"]).agg(
        allowed=pl.col(stat).sum(), games=pl.col("week").n_unique()
    ).with_columns((pl.col("allowed") / pl.col("games")).alias("per_game"))
    league = dg.group_by("position_group").agg(pl.col("per_game").mean().alias("lg"))
    dg = dg.join(league, on="position_group").with_columns(
        (pl.col("per_game") / pl.col("lg")).clip(*P.OPP_CLAMP).alias("factor")
    )
    return {(norm_abbr(r["opponent_team"]), r["position_group"]): r["factor"]
            for r in dg.to_dicts()}


def live_projections(target_week: int | None = None) -> pl.DataFrame:
    ps, sched, ros = _load()
    week = target_week or upcoming_week(sched, SEASON)
    if week is None:
        return pl.DataFrame()
    mat = upcoming_matchups(sched, SEASON, week)

    # matchup context: pass rush / protection, injuries
    def_rel, off_rel = _pass_rush(SEASON)
    out_ids = C.injuries_out(SEASON, week)

    # 2026 active roster -> current team per player
    roster = ros.filter(pl.col("status") == "ACT").select(
        player_id="gsis_id", team2026="team"
    ).with_columns(pl.col("team2026").map_elements(norm_abbr, return_dtype=pl.String))

    frames = []
    for stat, positions in P.STATS.items():
        form = current_form(ps, stat, positions)
        # opponent factor: current season if it has data, else last-season prior
        fac = {**defense_factors(ps, stat, positions, PRIOR_SEASON),
               **defense_factors(ps, stat, positions, SEASON)}
        is_pass = stat == "passing_yards"
        f = (form.join(roster, on="player_id", how="inner")
                 .join(ps.group_by("player_id").agg(
                       pl.col("player_display_name").last()), on="player_id", how="left")
                 .join(mat, left_on="team2026", right_on="team", how="inner"))
        if out_ids:                                    # drop players ruled out
            f = f.filter(~pl.col("player_id").is_in(list(out_ids)))

        def _row_ctx(s: dict) -> dict:
            opp_f = fac.get((s["opponent"], s["position_group"]), 1.0)
            pressure = (C.matchup_pressure(s["team2026"], s["opponent"], def_rel, off_rel)
                        if is_pass else None)
            pass_f = C.pass_yards_pressure_factor(pressure) if is_pass else 1.0
            return {"opp_factor": opp_f, "pressure": pressure or 0.0,
                    "pass_factor": pass_f,
                    "note": C.matchup_note(opp_f, pressure)}

        f = f.with_columns(
            pl.struct(["opponent", "position_group", "team2026"]).map_elements(
                _row_ctx,
                return_dtype=pl.Struct({"opp_factor": pl.Float64, "pressure": pl.Float64,
                                        "pass_factor": pl.Float64, "note": pl.String}),
            ).alias("_ctx")
        ).unnest("_ctx").with_columns(
            (pl.col("form_mean") * pl.col("opp_factor") * pl.col("pass_factor")
             ).alias("proj_mean"),
            pl.col("form_sd").alias("proj_sd"),
            pl.lit(stat).alias("stat"),
            pl.lit(SEASON).alias("season"),
            pl.lit(week).alias("week"),
            pl.col("opponent").alias("opponent_team"),
        )
        frames.append(f.select([
            "player_id", "player_display_name", "position_group", "team2026",
            "opponent_team", "is_home", "gameday", "season", "week", "stat",
            "opp_factor", "pressure", "note", "proj_mean", "proj_sd",
        ]))
    return pl.concat(frames, how="vertical")


if __name__ == "__main__":
    lp = live_projections()
    print(f"live projections for 2026 week {lp['week'][0] if lp.height else '?'}: "
          f"{lp.height} player-stat rows")
