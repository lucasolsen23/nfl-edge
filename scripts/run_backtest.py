"""Run the baseline backtest and print + save a units/ROI report.

Usage:
    python scripts/run_backtest.py                 # 2015-2025, REG season
    python scripts/run_backtest.py --start 2010 --end 2025
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from nfl_edge import backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "backtest_reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025, help="inclusive")
    args = ap.parse_args()

    seasons = range(args.start, args.end + 1)
    results = backtest.run(seasons)
    rep = backtest.report(results).sort("roi", descending=True)

    with pl.Config(tbl_rows=50, tbl_width_chars=200):
        print(f"\nBaseline backtest  seasons {args.start}-{args.end}  (REG, flat 1u)\n")
        print(rep)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"baselines_{args.start}_{args.end}.csv"
    rep.write_csv(out)
    print(f"\nsaved -> {out}")

    # Leakage tripwire: baselines should sit near break-even (small negative ROI
    # from the vig). A baseline far in the green is a bug to investigate.
    hot = rep.filter(pl.col("roi") > 0.05)
    if hot.height:
        print("\n[WARN] baseline(s) with ROI > 5% -- investigate for leakage:")
        print(hot.select(["strategy", "roi", "bets"]))


if __name__ == "__main__":
    main()
