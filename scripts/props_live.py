"""Live player-props tool for the upcoming NFL week.

Two modes:
  * no --lines : print the top projections per stat (candidates to look up)
  * --lines FILE : score your prop lines (a CSV of player,stat,line as read off
    PrizePicks/Underdog) into calibrated picks, ranked by confidence

Examples:
    python scripts/props_live.py
    python scripts/props_live.py --lines my_lines.csv --breakeven 0.577
    python scripts/props_live.py --lines my_lines.csv --notify discord

CSV format (header required):
    player,stat,line
    Ja'Marr Chase,receiving_yards,89.5
    Bijan Robinson,rushing_yards,79.5
Valid stats: passing_yards, rushing_yards, receiving_yards, receptions
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from nfl_edge.props import live, scan_props  # noqa: E402
from nfl_edge.edge import notify  # noqa: E402


def _print_candidates(lp: pl.DataFrame) -> None:
    wk = lp["week"][0] if lp.height else "?"
    print(f"\nTop projections — 2026 week {wk}  (no lines given; enter lines to score)\n")
    for stat in ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]:
        top = lp.filter(pl.col("stat") == stat).sort("proj_mean", descending=True).head(8)
        print(f"  {stat.replace('_', ' ').upper()}")
        for r in top.to_dicts():
            loc = "vs" if r["is_home"] else "@"
            print(f"    {r['player_display_name']:24} {r['team2026']} {loc} "
                  f"{r['opponent_team']:3}  {r['proj_mean']:6.1f} +/- {r['proj_sd']:.1f}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--lines", type=str, help="CSV of player,stat,line")
    ap.add_argument("--breakeven", type=float, default=0.577,
                    help="per-leg break-even (2-pick@3x=0.577, 3-pick@5x=0.585)")
    ap.add_argument("--notify", choices=["console", "discord", "telegram", "email"])
    args = ap.parse_args()

    lp = live.live_projections(target_week=args.week)
    if lp.height == 0:
        sys.exit("No upcoming games found (season may be over / data not refreshed).")

    if not args.lines:
        _print_candidates(lp)
        return

    with open(args.lines, newline="", encoding="utf-8") as f:
        lines = [{"player": r["player"], "stat": r["stat"].strip(),
                  "line": float(r["line"])} for r in csv.DictReader(f)]
    edges = scan_props.score_lines(lines, lp)
    body = scan_props.format_props(edges, breakeven=args.breakeven)
    print("\n" + body)

    if args.notify:
        picks = [e for e in edges if e.prob >= args.breakeven]
        if picks:
            n = notify.build_notifier(args.notify)
            n.send(f"NFL prop picks ({len(picks)})",
                   scan_props.format_props(picks, breakeven=args.breakeven))
            print(f"\n[notify:{args.notify}] sent {len(picks)} pick(s)")


if __name__ == "__main__":
    main()
