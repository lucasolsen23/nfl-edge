"""Validate the projection model: accuracy, bias, and distribution calibration.

A prop model is only useful if P(over line) is trustworthy, which requires both
the mean AND the spread to be right. We check:
  * MAE / bias / correlation of the mean projection
  * z = (actual - proj_mean) / proj_sd -- if calibrated, ~50% of actuals beat the
    mean and the tails match a normal (~16% below z=-1, ~84% below z=+1)

Usage: python scripts/eval_props.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from nfl_edge.props import projections as P  # noqa: E402


def main() -> None:
    proj = P.build_all(save=False).drop_nulls(["proj_mean", "proj_sd", "actual"])
    proj = proj.with_columns(
        ((pl.col("actual") - pl.col("proj_mean")) / pl.col("proj_sd")).alias("z")
    )
    rep = proj.group_by("stat").agg(
        n=pl.len(),
        MAE=(pl.col("actual") - pl.col("proj_mean")).abs().mean().round(2),
        bias=(pl.col("actual") - pl.col("proj_mean")).mean().round(2),
        corr=pl.corr("proj_mean", "actual").round(3),
        over_rate=(pl.col("actual") > pl.col("proj_mean")).mean().round(3),   # ~0.50
        z_lt_neg1=(pl.col("z") < -1).mean().round(3),                          # ~0.16
        z_lt_pos1=(pl.col("z") < 1).mean().round(3),                           # ~0.84
    ).sort("stat")

    with pl.Config(tbl_width_chars=200):
        print("\nProjection calibration (2015-2025, walk-forward)\n")
        print(rep)
    print("\nTargets for a calibrated model:")
    print("  over_rate ~ 0.50   z_lt_neg1 ~ 0.16   z_lt_pos1 ~ 0.84")
    print("  (yardage is right-skewed, so small deviations from the normal "
          "tails are expected and fine.)")


if __name__ == "__main__":
    main()
