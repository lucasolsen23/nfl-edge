"""Calibrate the outcome distributions from historical results.

The scanner prices Kalshi spread/total strikes by putting a distribution around
the book's consensus line. A plain normal misses NFL reality -- margins pile up
on key numbers (3, 7), so half-point strikes across those numbers are worth more
than a normal implies. This script measures the ACTUAL distribution of outcomes
around the closing line and ships it as a compact lookup the scanner loads with
no heavy deps.

Outputs src/nfl_edge/edge/distributions.json:
    margin: residuals of (home_margin - spread_line), symmetrized (+/-), so it
            prices both teams' "win by over X".
    total:  residuals of (actual_total - total_line).

Residuals fall on a 0.5-point grid, so we store them as value->cumulative-count
(tiny, exact). Run locally after refreshing schedules; commit the JSON.

    python scripts/calibrate.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHED = REPO_ROOT / "data" / "raw" / "schedules.parquet"
OUT = REPO_ROOT / "src" / "nfl_edge" / "edge" / "distributions.json"


def _cdf(values: list[float]) -> dict:
    """value -> cumulative count (<= value), plus n and sd."""
    c = Counter(round(v, 1) for v in values)
    vals = sorted(c)
    cum, running = [], 0
    for v in vals:
        running += c[v]
        cum.append(running)
    n = running
    mean = sum(values) / n
    sd = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return {"values": vals, "cum": cum, "n": n, "mean": round(mean, 4),
            "sd": round(sd, 4)}


def main() -> None:
    s = pl.read_parquet(SCHED).filter(
        pl.col("result").is_not_null()
        & pl.col("spread_line").is_not_null()
        & pl.col("total_line").is_not_null()
    )
    margin_res = (s["result"] - s["spread_line"]).to_list()
    # symmetrize: we price both "home by >X" and "away by >X"
    margin_res = margin_res + [-r for r in margin_res]
    total_res = ((s["home_score"] + s["away_score"]) - s["total_line"]).to_list()

    payload = {
        "generated": "auto",
        "n_games": s.height,
        "margin": _cdf(margin_res),
        "total": _cdf(total_res),
    }
    OUT.write_text(json.dumps(payload))
    m, t = payload["margin"], payload["total"]
    print(f"[calibrate] {s.height} games -> {OUT}")
    print(f"  margin residuals: n={m['n']} sd={m['sd']} (symmetrized)")
    print(f"  total  residuals: n={t['n']} sd={t['sd']}")


if __name__ == "__main__":
    main()
