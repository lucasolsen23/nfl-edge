"""Live defensive-player props (tackles, sacks) for the upcoming week.

    python scripts/def_props.py                     # top candidates
    python scripts/def_props.py --lines def.csv     # score your lines
    python scripts/def_props.py --lines def.csv --notify discord

CSV format:
    player,stat,line
    Roquan Smith,tackles,8.5
    Micah Parsons,sacks,0.5
Valid stats: tackles, sacks
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from nfl_edge.props import defense as D  # noqa: E402
from nfl_edge.props import scan_props  # noqa: E402
from nfl_edge.edge import notify  # noqa: E402


def _candidates(lp: pl.DataFrame) -> None:
    wk = lp["week"][0] if lp.height else "?"
    print(f"\nTop defensive projections — 2026 week {wk}\n")
    for stat in ["tackles", "sacks"]:
        top = lp.filter(pl.col("stat") == stat).sort("proj_mean", descending=True).head(8)
        print(f"  {stat.upper()}")
        for r in top.to_dicts():
            loc = "vs" if r["is_home"] else "@"
            note = f"  [{r['note']}]" if r.get("note") else ""
            print(f"    {r['player_display_name']:22} {r['team2026']} {loc} "
                  f"{r['opponent_team']:3}  {r['proj_mean']:5.2f}{note}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--lines", type=str)
    ap.add_argument("--breakeven", type=float, default=0.577)
    ap.add_argument("--notify", choices=["console", "discord", "telegram", "email"])
    args = ap.parse_args()

    lp = D.live_def_projections(target_week=args.week)
    if lp.height == 0:
        sys.exit("No upcoming games found.")
    if not args.lines:
        _candidates(lp)
        return

    with open(args.lines, newline="", encoding="utf-8") as f:
        lines = [{"player": r["player"], "stat": r["stat"].strip(),
                  "line": float(r["line"])} for r in csv.DictReader(f)]
    edges = D.score_def_lines(lines, lp)
    body = scan_props.format_props(edges, breakeven=args.breakeven)
    print("\n" + body)

    if args.notify:
        picks = [e for e in edges if e.prob >= args.breakeven]
        if picks:
            notify.build_notifier(args.notify).send(
                f"NFL defensive prop picks ({len(picks)})",
                scan_props.format_props(picks, breakeven=args.breakeven))
            print(f"\n[notify:{args.notify}] sent {len(picks)} pick(s)")


if __name__ == "__main__":
    main()
