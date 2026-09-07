"""Feature engineering -- Phase 2, step 4.

Turns play-by-play into point-in-time team-strength features and assembles a
model-ready table, one row per game, with leakage-free predictors.

The cardinal rule: every feature for a given game uses only games that finished
BEFORE that game's kickoff. We enforce it by aggregating to one row per
team-game, then within each team (ordered by season, week) taking a rolling mean
that is shift(1)-ed so the current game is excluded.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROC_DIR = REPO_ROOT / "data" / "processed"

# rolling window of prior games used for team-strength features
ROLL_WINDOW = 10
ROLL_MIN = 4          # require this many prior games before a feature is valid


# --------------------------------------------------------------------------
# 1. play-by-play -> one row per (game, team) with offense & defense EPA
# --------------------------------------------------------------------------
def team_game_epa(pbp: pl.DataFrame) -> pl.DataFrame:
    """Aggregate scrimmage plays (pass/run) to per-team offensive and defensive
    EPA/play, success rate, and pass/rush splits for each game."""
    plays = pbp.filter(
        pl.col("epa").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
        & pl.col("posteam").is_not_null()
    )

    offense = plays.group_by(["game_id", "season", "week", "posteam"]).agg(
        off_epa=pl.col("epa").mean(),
        off_sr=pl.col("success").mean(),
        off_pass_epa=pl.col("epa").filter(pl.col("pass") == 1).mean(),
        off_rush_epa=pl.col("epa").filter(pl.col("rush") == 1).mean(),
        off_plays=pl.len(),
    ).rename({"posteam": "team"})

    defense = plays.group_by(["game_id", "defteam"]).agg(
        def_epa=pl.col("epa").mean(),          # EPA allowed per play (lower is better)
        def_sr=pl.col("success").mean(),
    ).rename({"defteam": "team"})

    return offense.join(defense, on=["game_id", "team"], how="inner")


# --------------------------------------------------------------------------
# 2. per-team rolling (point-in-time) strength -- shift(1) excludes current game
# --------------------------------------------------------------------------
_ROLL_COLS = ["off_epa", "def_epa", "off_sr", "def_sr", "off_pass_epa", "off_rush_epa"]


def rolling_team_strength(team_game: pl.DataFrame) -> pl.DataFrame:
    """For each team-game, compute the mean of each metric over the prior
    ROLL_WINDOW games (this game excluded). Crosses season boundaries -- team
    strength has memory -- but a `gp_season` counter lets the model discount
    early-season rows."""
    tg = team_game.sort(["team", "season", "week", "game_id"])

    exprs = []
    for c in _ROLL_COLS:
        exprs.append(
            pl.col(c).shift(1)
            .rolling_mean(window_size=ROLL_WINDOW, min_samples=ROLL_MIN)
            .over("team")
            .alias(f"{c}_r")
        )
    # games played earlier this season (for the away/home team), pre-kickoff
    exprs.append(
        (pl.col("week").cum_count().over(["team", "season"]) - 1).alias("gp_season")
    )
    return tg.with_columns(exprs)


# --------------------------------------------------------------------------
# 3. assemble one row per game: home features, away features, diffs, targets
# --------------------------------------------------------------------------
_FEATURE_BASE = [f"{c}_r" for c in _ROLL_COLS]


def build_game_table(schedules: pl.DataFrame, team_strength: pl.DataFrame) -> pl.DataFrame:
    """Join home & away rolling strength onto each completed game and add
    betting targets. Positive spread_line => home favored; result = home margin."""
    strength = team_strength.select(
        ["game_id", "team", "gp_season", *_FEATURE_BASE]
    )

    home = strength.rename(
        {c: f"home_{c}" for c in ["gp_season", *_FEATURE_BASE]}
    ).rename({"team": "home_team"})
    away = strength.rename(
        {c: f"away_{c}" for c in ["gp_season", *_FEATURE_BASE]}
    ).rename({"team": "away_team"})

    games = schedules.filter(
        pl.col("result").is_not_null()
        & pl.col("spread_line").is_not_null()
        & pl.col("total_line").is_not_null()
    )

    g = (
        games.join(home, on=["game_id", "home_team"], how="left")
        .join(away, on=["game_id", "away_team"], how="left")
    )

    # difference features (home minus away) -- what the model actually leans on
    diffs = []
    for c in _FEATURE_BASE:
        diffs.append((pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"diff_{c}"))

    g = g.with_columns(
        diffs
        + [
            # targets
            (pl.col("result") > pl.col("spread_line")).alias("home_covered"),
            ((pl.col("home_score") + pl.col("away_score")) > pl.col("total_line")).alias("went_over"),
            pl.col("result").alias("home_margin"),
            (pl.col("home_score") + pl.col("away_score")).alias("actual_total"),
            # simple home-field/rest context
            (pl.col("home_rest") - pl.col("away_rest")).alias("rest_diff"),
        ]
    )
    return g


FEATURE_COLS = [f"diff_{c}" for c in _FEATURE_BASE] + ["rest_diff", "spread_line"]


def build(save: bool = True) -> pl.DataFrame:
    pbp = pl.read_parquet(RAW_DIR / "pbp.parquet")
    sched = pl.read_parquet(RAW_DIR / "schedules.parquet")
    tg = team_game_epa(pbp)
    ts = rolling_team_strength(tg)
    game = build_game_table(sched, ts)
    if save:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        out = PROC_DIR / "game_features.parquet"
        game.write_parquet(out)
        print(f"[features] {game.height} games -> {out}")
    return game


if __name__ == "__main__":
    build()
