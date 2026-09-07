"""Calibrate P(over) for props from the empirical residual shape.

Player yardage/receptions are right-skewed: actuals land below the projected
mean more than half the time. Pricing prop lines with a symmetric normal would
systematically overstate 'over', so we measure the empirical distribution of the
standardized residual z = (actual - proj_mean) / proj_sd for each stat and price
P(over) from it. Same idea as the game-model calibration.

Outputs src/nfl_edge/props/prop_distributions.json (per-stat z CDF, 0.05 grid).

    python scripts/calibrate_props.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from nfl_edge.props import projections as P  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "nfl_edge" / "props" / "prop_distributions.json"


def _cdf(zs: list[float]) -> dict:
    c = Counter(round(z, 2) for z in zs)   # 0.01 grid is fine, small enough
    vals = sorted(c)
    cum, run = [], 0
    for v in vals:
        run += c[v]
        cum.append(run)
    return {"values": vals, "cum": cum, "n": run}


def main() -> None:
    proj = P.build_all(save=False).drop_nulls(["proj_mean", "proj_sd", "actual"])
    proj = proj.with_columns(
        ((pl.col("actual") - pl.col("proj_mean")) / pl.col("proj_sd")).alias("z")
    )
    payload = {"generated": "auto"}
    for stat in P.STATS:
        zs = proj.filter(pl.col("stat") == stat)["z"].to_list()
        payload[stat] = _cdf(zs)
        print(f"  {stat:16} n={payload[stat]['n']}  "
              f"median z={sorted(zs)[len(zs)//2]:+.2f}  (negative = right-skew)")
    OUT.write_text(json.dumps(payload))
    print(f"[calibrate_props] -> {OUT}")


if __name__ == "__main__":
    main()
