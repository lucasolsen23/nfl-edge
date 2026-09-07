"""Defensive-player props: tackles and sacks.

Different from offensive props in two ways:
  * distribution -- tackles are count-ish but roughly normal (starters 5-8/gm),
    while sacks are rare counts (a DL sacks in ~20% of games), so sacks use a
    POISSON, not a normal.
  * opponent logic is inverted -- a defender faces the OPPOSING offense, so:
      tackles scale with the opponent's pace (plays per game)
      sacks   scale with the opponent's pass protection (O-line sacks allowed)

Reuses the live scaffolding (roster -> 2026 team, upcoming opponent, injuries)
and the pass-rush data already built in context.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from . import live as LV
from . import context as C
from .scan_props import PropEdge, format_props
from ..edge.teams import norm_abbr

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"

# stat -> position groups that get the prop
DEF_STATS = {"tackles": ("LB", "DB"), "sacks": ("DL", "LB")}
ROLL_WINDOW, ROLL_MIN = 10, 3
PACE_CLAMP = (0.85, 1.15)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for X ~ Poisson(lam)."""
    if k < 0:
        return 0.0
    s, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        if i > 0:
            term *= lam / i
        s += term
    return s


def prob_over(stat: str, mean: float, sd: float, line: float) -> float:
    """P(stat > line). Sacks -> Poisson(mean); tackles -> normal(mean, sd)."""
    if stat == "sacks":
        lam = max(mean, 1e-6)
        k = int(math.floor(line)) + 1            # need >= k (e.g. line 0.5 -> k=1)
        return 1.0 - _poisson_cdf(k - 1, lam)
    if sd is None or sd <= 0:
        return 1.0 if mean > line else 0.0
    return 1.0 - _phi((line - mean) / sd)


def _with_def_stats(ps: pl.DataFrame) -> pl.DataFrame:
    return ps.with_columns(
        (pl.col("def_tackles_solo").fill_null(0)
         + pl.col("def_tackle_assists").fill_null(0)).alias("tackles"),
        pl.col("def_sacks").fill_null(0.0).alias("sacks"),
    )


def current_def_form(ps: pl.DataFrame, stat: str, positions: tuple[str, ...],
                     window: int = ROLL_WINDOW) -> pl.DataFrame:
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
    ).filter(pl.col("games") >= ROLL_MIN)
    pos_sd = agg.group_by("position_group").agg(pl.col("form_sd").median().alias("pos_sd"))
    return agg.join(pos_sd, on="position_group", how="left").with_columns(
        pl.max_horizontal(pl.col("form_sd").fill_null(pl.col("pos_sd")),
                          pl.col("pos_sd") * 0.5, pl.lit(0.5)).alias("form_sd")
    )


def offense_pace_factor(season: int) -> dict:
    """team -> offensive plays/game relative to league (facing a fast, high-play
    offense means more tackle opportunities)."""
    path = RAW / "pbp.parquet"
    if not path.exists():
        return {}
    for yr in (season, season - 1):
        d = (pl.scan_parquet(path)
             .filter((pl.col("season") == yr) & pl.col("play_type").is_in(["pass", "run"]))
             .select(["posteam", "week"]).collect())
        if d.height:
            pg = d.group_by(["posteam", "week"]).agg(pl.len().alias("plays")) \
                  .group_by("posteam").agg(pl.col("plays").mean().alias("ppg"))
            lg = pg["ppg"].mean()
            return {norm_abbr(r["posteam"]): min(PACE_CLAMP[1], max(PACE_CLAMP[0], r["ppg"] / lg))
                    for r in pg.to_dicts() if r["posteam"] and lg}
    return {}


def live_def_projections(target_week: int | None = None) -> pl.DataFrame:
    ps, sched, ros = LV._load()
    ps = _with_def_stats(ps)
    week = target_week or LV.upcoming_week(sched, LV.SEASON)
    if week is None:
        return pl.DataFrame()
    mat = LV.upcoming_matchups(sched, LV.SEASON, week)
    _, off_rel = LV._pass_rush(LV.SEASON)          # opponent O-line sacks allowed
    pace = offense_pace_factor(LV.SEASON)
    out_ids = C.injuries_out(LV.SEASON, week)

    roster = ros.filter(pl.col("status") == "ACT").select(
        player_id="gsis_id", team2026="team"
    ).with_columns(pl.col("team2026").map_elements(norm_abbr, return_dtype=pl.String))
    names = ps.group_by("player_id").agg(pl.col("player_display_name").last())

    frames = []
    for stat, positions in DEF_STATS.items():
        form = current_def_form(ps, stat, positions)
        f = (form.join(roster, on="player_id", how="inner")
                 .join(names, on="player_id", how="left")
                 .join(mat, left_on="team2026", right_on="team", how="inner"))
        if out_ids:
            f = f.filter(~pl.col("player_id").is_in(list(out_ids)))
        is_sacks = stat == "sacks"

        def _ctx(s: dict) -> dict:
            fac = (off_rel.get(s["opponent"], 1.0) if is_sacks
                   else pace.get(s["opponent"], 1.0))
            note = ("favorable matchup" if fac >= 1.08
                    else "tough matchup" if fac <= 0.92 else "")
            return {"opp_factor": fac, "note": note}

        f = f.with_columns(
            pl.struct(["opponent"]).map_elements(
                _ctx, return_dtype=pl.Struct({"opp_factor": pl.Float64, "note": pl.String})
            ).alias("_c")
        ).unnest("_c").with_columns(
            (pl.col("form_mean") * pl.col("opp_factor")).alias("proj_mean"),
            pl.col("form_sd").alias("proj_sd"),
            pl.lit(stat).alias("stat"),
            pl.lit(LV.SEASON).alias("season"),
            pl.lit(week).alias("week"),
            pl.col("opponent").alias("opponent_team"),
        )
        frames.append(f.select([
            "player_id", "player_display_name", "position_group", "team2026",
            "opponent_team", "is_home", "gameday", "season", "week", "stat",
            "opp_factor", "note", "proj_mean", "proj_sd",
        ]))
    return pl.concat(frames, how="vertical")


def score_def_lines(lines: list[dict], proj: pl.DataFrame) -> list[PropEdge]:
    latest = proj.sort(["season", "week"]).group_by(["player_id", "stat"]).last()
    edges: list[PropEdge] = []
    for ln in lines:
        stat = ln["stat"]
        row = latest.filter(
            (pl.col("stat") == stat)
            & (pl.col("player_display_name").str.to_lowercase() == ln["player"].lower())
        )
        if row.is_empty():
            continue
        r = row.to_dicts()[0]
        p_over = prob_over(stat, r["proj_mean"], r["proj_sd"], ln["line"])
        pick, prob = ("OVER", p_over) if p_over >= 0.5 else ("UNDER", 1 - p_over)
        edges.append(PropEdge(
            player=r["player_display_name"], stat=stat, line=ln["line"], pick=pick,
            prob=round(prob, 4), proj_mean=round(r["proj_mean"], 2),
            proj_sd=round(r["proj_sd"], 2), opponent=r["opponent_team"],
            season=r["season"], week=r["week"], note=r.get("note", "") or "",
            opp_factor=r.get("opp_factor", 1.0), player_id=r["player_id"],
            team=r["team2026"],
        ))
    edges.sort(key=lambda e: e.prob, reverse=True)
    return edges
