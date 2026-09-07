"""Walk-forward player-stat projections.

For each player-week and stat we project:
    proj_mean = player_recent_form  x  opponent_defense_factor
    proj_sd   = player's game-to-game SD (shrunk toward a position baseline)

Everything is point-in-time: recent form uses only the player's prior games, and
the opponent factor uses only defense/league games played earlier that season.
No leakage, same discipline as the game model.

P(stat > line) then prices any prop line (normal around proj_mean; yardage is
roughly bell-shaped -- good enough for v1, refine later).
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW = REPO_ROOT / "data" / "raw" / "player_stats.parquet"
PROC = REPO_ROOT / "data" / "processed"

# stat -> position groups that actually produce it (for filtering + opp factor)
STATS: dict[str, tuple[str, ...]] = {
    "passing_yards": ("QB",),
    "rushing_yards": ("RB", "QB"),
    "receiving_yards": ("WR", "TE", "RB"),
    "receptions": ("WR", "TE", "RB"),
}

ROLL_WINDOW = 10          # player's last N games
ROLL_MIN = 3              # need this many prior games to project
OPP_CLAMP = (0.75, 1.25)  # cap the defensive adjustment


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# empirical per-stat z-distribution (built by scripts/calibrate_props.py), used to
# price P(over) with the real right-skew instead of a symmetric normal.
_ZDIST_FILE = Path(__file__).with_name("prop_distributions.json")
_CLAMP = (0.02, 0.98)


def _load_zdist() -> dict:
    try:
        return json.loads(_ZDIST_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


_ZDIST = _load_zdist()


def _emp_prob_z_over(stat: str, t: float) -> float | None:
    """Empirical P(z > t) for a stat; None if unavailable."""
    d = _ZDIST.get(stat)
    if not d:
        return None
    values, cum, n = d["values"], d["cum"], d["n"]
    idx = bisect.bisect_right(values, t)
    below = cum[idx - 1] if idx > 0 else 0
    p = (n - below) / n
    return min(_CLAMP[1], max(_CLAMP[0], p))


def prob_over(mean: float, sd: float, line: float, stat: str | None = None) -> float:
    """P(stat > line). Uses the empirical z-distribution for `stat` when available
    (captures right-skew); otherwise a normal around (mean, sd)."""
    if sd is None or sd <= 0:
        return 1.0 if mean > line else 0.0
    t = (line - mean) / sd
    if stat:
        p = _emp_prob_z_over(stat, t)
        if p is not None:
            return p
    return 1.0 - _phi(t)


def using_empirical() -> bool:
    return bool(_ZDIST)


def _expanding_prior_mean(df: pl.DataFrame, value: str, group: list[str],
                          out: str) -> pl.DataFrame:
    """Expanding mean of `value` over rows BEFORE the current one, within group
    (ordered by season, week). Shifted so the current game is excluded."""
    df = df.sort(group + ["season", "week"])
    return df.with_columns(
        pl.col(value).cum_sum().over(group).alias("_cs"),
        pl.col(value).cum_count().over(group).alias("_cc"),
    ).with_columns(
        (pl.col("_cs").shift(1).over(group) / pl.col("_cc").shift(1).over(group)
         ).alias(out)
    ).drop("_cs", "_cc")


def opponent_factor(ps: pl.DataFrame, stat: str, positions: tuple[str, ...]) -> pl.DataFrame:
    """Per (season, week, defense, position_group): how much this defense has
    allowed in `stat` to that position group so far this season vs the league,
    as a clamped multiplicative factor. Uses only prior weeks."""
    rel = ps.filter(pl.col("position_group").is_in(list(positions)))
    # allowed by a defense to a position group in one week
    dg = rel.group_by(["season", "week", "opponent_team", "position_group"]).agg(
        allowed=pl.col(stat).sum()
    )
    # defense's season-to-date allowed (prior weeks only)
    dg = _expanding_prior_mean(dg, "allowed",
                               ["season", "opponent_team", "position_group"], "def_td")
    # league season-to-date baseline per position group (prior weeks only)
    dg = _expanding_prior_mean(dg, "allowed", ["season", "position_group"], "lg_td")
    return dg.with_columns(
        pl.when(pl.col("lg_td").is_null() | (pl.col("lg_td") <= 0))
        .then(1.0)
        .otherwise((pl.col("def_td") / pl.col("lg_td")).clip(*OPP_CLAMP))
        .fill_null(1.0)
        .alias("opp_factor")
    ).select(["season", "week", "opponent_team", "position_group", "opp_factor"])


def build_stat(ps: pl.DataFrame, stat: str) -> pl.DataFrame:
    positions = STATS[stat]
    rel = ps.filter(
        (pl.col("season_type") == "REG")
        & pl.col("position_group").is_in(list(positions))
    ).sort(["player_id", "season", "week"])

    # player recent form + game-to-game SD (prior games only)
    prior = pl.col(stat).shift(1).over("player_id")
    rel = rel.with_columns(
        prior.rolling_mean(window_size=ROLL_WINDOW, min_samples=ROLL_MIN)
        .over("player_id").alias("form_mean"),
        prior.rolling_std(window_size=ROLL_WINDOW, min_samples=ROLL_MIN)
        .over("player_id").alias("form_sd"),
        pl.col("week").cum_count().over(["player_id", "season"]).alias("_gp"),
    )

    # opponent defensive adjustment
    of = opponent_factor(ps, stat, positions)
    rel = rel.join(of, on=["season", "week", "opponent_team", "position_group"], how="left")
    rel = rel.with_columns(pl.col("opp_factor").fill_null(1.0))

    # position-group baseline SD to shrink toward when a player has few games
    pos_sd = rel.group_by("position_group").agg(pl.col("form_sd").median().alias("pos_sd"))
    rel = rel.join(pos_sd, on="position_group", how="left")

    rel = rel.with_columns(
        (pl.col("form_mean") * pl.col("opp_factor")).alias("proj_mean"),
    ).with_columns(
        # sd: player's own, else position baseline; floored so we're never
        # absurdly confident on a small sample.
        pl.max_horizontal(
            pl.col("form_sd").fill_null(pl.col("pos_sd")),
            pl.col("pos_sd") * 0.5,
            pl.lit(1.0),
        ).alias("proj_sd")
    )

    return rel.filter(pl.col("proj_mean").is_not_null()).select([
        "player_id", "player_display_name", "position_group", "team",
        "opponent_team", "season", "week", "opp_factor",
        pl.lit(stat).alias("stat"),
        "proj_mean", "proj_sd",
        pl.col(stat).alias("actual"),
    ])


def build_all(save: bool = True) -> pl.DataFrame:
    ps = pl.read_parquet(RAW)
    frames = [build_stat(ps, s) for s in STATS]
    out = pl.concat(frames, how="vertical")
    if save:
        PROC.mkdir(parents=True, exist_ok=True)
        p = PROC / "projections.parquet"
        out.write_parquet(p)
        print(f"[props] {out.height} player-week projections -> {p}")
    return out


if __name__ == "__main__":
    build_all()
