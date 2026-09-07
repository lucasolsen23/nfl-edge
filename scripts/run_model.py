"""Train the walk-forward model and backtest it against the closing spread.

Usage:
    python scripts/run_model.py
    python scripts/run_model.py --alpha 5 --thresholds 0.5 1 2 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from nfl_edge import backtest, model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "backtest_reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0])
    args = ap.parse_args()

    preds = model.build(alpha=args.alpha)
    pred_map = dict(zip(preds.get_column("game_id"), preds.get_column("pred_margin")))

    # How good is the power rating at predicting margin vs. the market spread?
    pm = preds.get_column("pred_margin").to_numpy()
    hm = preds.get_column("home_margin").to_numpy()
    sp = preds.get_column("spread_line").to_numpy()
    print("\nPrediction quality (out-of-sample):")
    print(f"  model MAE vs actual margin : {np.mean(np.abs(pm - hm)):.3f}")
    print(f"  spread MAE vs actual margin: {np.mean(np.abs(sp - hm)):.3f}  (market benchmark)")
    print(f"  corr(model, actual)        : {np.corrcoef(pm, hm)[0,1]:.3f}")

    seasons = range(int(preds.get_column("season").min()),
                    int(preds.get_column("season").max()) + 1)

    strategies = [backtest.model_ats_strategy(pred_map, t) for t in args.thresholds]
    results = backtest.run(seasons, strategies=strategies)
    rep = backtest.report(results).sort("roi", descending=True)

    with pl.Config(tbl_rows=50, tbl_width_chars=200):
        print(f"\nModel ATS backtest  seasons {seasons.start}-{seasons.stop-1}  "
              f"alpha={args.alpha}  (flat 1u vs closing spread)\n")
        print(rep)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "model_ats.csv"
    rep.write_csv(out)
    print(f"\nsaved -> {out}")
    print("\nBreak-even at -110 juice is 52.4% win_pct. Beating that on a real "
          "sample = beating the closing line.")


if __name__ == "__main__":
    main()
