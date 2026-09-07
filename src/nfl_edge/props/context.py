"""Matchup context for props: defense strength, pass-rush, and injuries.

Adds the "who feasts / who struggles vs this defense" layer:
  * positional defense strength -- how much a defense allows to WR/TE/RB/QB
    (last completed season as a prior, so Week 1 isn't neutral)
  * pass-rush vs pass-protection -- team sack rate generated (D-line) vs sack
    rate allowed (O-line); a high-pressure matchup downgrades QB passing and
    flags sack risk
  * injuries -- players ruled Out/Doubtful are dropped from projections

Team-unit altitude on purpose: individual O-lineman-vs-edge matchups need
snap-level grading (PFF-tier, paid), so we stay at the unit level, which carries
most of the signal from free data.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import nflreadpy as nfl

from ..edge.teams import norm_abbr

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "data" / "raw"

OUT_STATUSES = ("Out", "Doubtful")          # treat these as not playing
PRESSURE_CLAMP = (0.90, 1.10)               # passing-yards factor bounds


# --------------------------------------------------------------------------
# Pass rush (D-line) vs pass protection (O-line), from play-by-play
# --------------------------------------------------------------------------
def pass_rush_indices(pbp: pl.DataFrame, season: int) -> tuple[dict, dict]:
    """Return (defense_pressure_rel, offense_sack_allowed_rel), each team's sack
    rate on dropbacks relative to the league that season (1.0 = average)."""
    d = pbp.filter((pl.col("season") == season) & (pl.col("qb_dropback") == 1))
    if d.height == 0:
        return {}, {}
    deff = d.group_by("defteam").agg(sr=pl.col("sack").mean())
    lg = deff["sr"].mean()
    def_rel = {norm_abbr(r["defteam"]): (r["sr"] / lg if lg else 1.0)
               for r in deff.to_dicts() if r["defteam"]}
    off = d.group_by("posteam").agg(sr=pl.col("sack").mean())
    lgo = off["sr"].mean()
    off_rel = {norm_abbr(r["posteam"]): (r["sr"] / lgo if lgo else 1.0)
               for r in off.to_dicts() if r["posteam"]}
    return def_rel, off_rel


def matchup_pressure(off_team: str, def_team: str,
                     def_rel: dict, off_rel: dict) -> float:
    """Combined pressure for an offense vs a defense (1.0 neutral, >1 = more
    sacks/pressure than average). Geometric mean of the two sides."""
    dp = def_rel.get(norm_abbr(def_team), 1.0)      # defense pass rush
    op = off_rel.get(norm_abbr(off_team), 1.0)      # offense sacks allowed
    return (max(dp, 0.1) * max(op, 0.1)) ** 0.5


def pass_yards_pressure_factor(pressure: float, k: float = 0.12) -> float:
    """More pressure -> fewer passing yards. Modest, clamped."""
    return min(PRESSURE_CLAMP[1], max(PRESSURE_CLAMP[0], 1.0 - k * (pressure - 1.0)))


# --------------------------------------------------------------------------
# Injuries
# --------------------------------------------------------------------------
def injuries_out(season: int, week: int) -> set[str]:
    """gsis_ids ruled Out/Doubtful for (season, week). Empty if that season's
    injury reports aren't published yet (e.g. before the 2026 season)."""
    try:
        inj = nfl.load_injuries(seasons=[season])
    except Exception:            # noqa: BLE001 -- season not published yet
        return set()
    rel = inj.filter(
        (pl.col("week") == week) & pl.col("report_status").is_in(list(OUT_STATUSES))
    )
    return set(rel["gsis_id"].drop_nulls().to_list())


# --------------------------------------------------------------------------
# Matchup note for display
# --------------------------------------------------------------------------
def matchup_note(opp_factor: float, pressure: float | None) -> str:
    parts = []
    if opp_factor >= 1.08:
        parts.append("favorable matchup")
    elif opp_factor <= 0.92:
        parts.append("tough matchup")
    if pressure is not None and pressure >= 1.15:
        parts.append("high sack risk")
    return ", ".join(parts)
