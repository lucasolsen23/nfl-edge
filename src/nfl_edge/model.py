"""Walk-forward model -- Phase 2, step 5.

A ridge regression that predicts the home margin from EPA-based team-strength
differentials ONLY (the market line is deliberately excluded). That makes it an
independent power rating: betting its disagreement with the closing spread is a
real test of whether EPA carries information the market hasn't already priced.

Walk-forward discipline: to predict week W of season S we train only on games
that kicked off before then, retraining at each (season, week) step. No random
splits, ever.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = REPO_ROOT / "data" / "processed"

# Market-INDEPENDENT features (no spread_line) -> honest "beat the number" test.
MODEL_FEATURES = [
    "diff_off_epa_r", "diff_def_epa_r",
    "diff_off_sr_r", "diff_def_sr_r",
    "diff_off_pass_epa_r", "diff_off_rush_epa_r",
    "rest_diff",
]
TARGET = "home_margin"
MIN_TRAIN = 300          # minimum prior games before we start predicting


def _load() -> pl.DataFrame:
    g = pl.read_parquet(PROC_DIR / "game_features.parquet")
    # keep only rows with all features present and a known outcome
    g = g.drop_nulls(MODEL_FEATURES + [TARGET, "spread_line"])
    return g.sort(["season", "week", "game_id"])


def walk_forward_predict(alpha: float = 5.0) -> pl.DataFrame:
    """Return per-game out-of-sample predictions: game_id, season, week,
    spread_line, pred_margin, home_margin, home_covered."""
    g = _load()
    X_all = g.select(MODEL_FEATURES).to_numpy()
    y_all = g.get_column(TARGET).to_numpy().astype(float)

    # chronological bucket index so "train on everything earlier" is a mask
    buckets = g.select(["season", "week"]).unique(maintain_order=True).with_row_index("b")
    g = g.join(buckets, on=["season", "week"], how="left")
    b = g.get_column("b").to_numpy()

    preds = np.full(len(g), np.nan)
    n_buckets = int(b.max()) + 1
    for cur in range(1, n_buckets):
        train_mask = b < cur
        if train_mask.sum() < MIN_TRAIN:
            continue
        test_mask = b == cur
        scaler = StandardScaler().fit(X_all[train_mask])
        model = Ridge(alpha=alpha).fit(scaler.transform(X_all[train_mask]), y_all[train_mask])
        preds[test_mask] = model.predict(scaler.transform(X_all[test_mask]))

    out = g.select(
        ["game_id", "season", "week", "spread_line", "home_margin", "home_covered"]
    ).with_columns(pl.Series("pred_margin", preds))
    # NaN (not null) marks buckets skipped before MIN_TRAIN -- filter those.
    return out.filter(pl.col("pred_margin").is_not_nan())


def build(alpha: float = 5.0, save: bool = True) -> pl.DataFrame:
    preds = walk_forward_predict(alpha=alpha)
    if save:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        out = PROC_DIR / "predictions.parquet"
        preds.write_parquet(out)
        print(f"[model] {preds.height} out-of-sample predictions -> {out}")
    return preds


if __name__ == "__main__":
    build()
