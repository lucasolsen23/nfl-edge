"""Auto props: pull book player-prop lines and score them against our model.

Finds props where our projection disagrees with the de-vigged book consensus
(mkt_edge), then suggests correlated parlays from the strong picks. Book lines
track PrizePicks closely, so a value pick here is one to look for on your app.

    setx ODDS_API_KEY your_key
    python scripts/props_auto.py                       # offensive markets, upcoming slate
    python scripts/props_auto.py --markets player_pass_yds player_reception_yds
    python scripts/props_auto.py --limit 4             # cap events (saves credits)
    python scripts/props_auto.py --notify discord

Credit cost = (upcoming events) x (markets). The Odds API free tier is 500/mo.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_edge.props import odds_props, live, defense, correlations  # noqa: E402
from nfl_edge.edge import notify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=odds_props.DEFAULT_MARKETS)
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap events fetched (0=all)")
    ap.add_argument("--min-edge", type=float, default=0.04)
    ap.add_argument("--notify", choices=["console", "discord", "telegram", "email"])
    args = ap.parse_args()

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("ODDS_API_KEY not set.")

    events = odds_props.upcoming_events(key, args.days)
    if args.limit:
        events = events[:args.limit]
    print(f"fetching props for {len(events)} events x {len(args.markets)} markets "
          f"(~{len(events)*len(args.markets)} credits)...")
    lines = []
    for e in events:
        lines.extend(odds_props.fetch_event_props(key, e["id"], args.markets))
    print(f"got {len(lines)} book prop lines  (quota remaining: "
          f"{getattr(odds_props.fetch_event_props, 'quota', '?')})")

    off_proj = live.live_projections()
    def_proj = defense.live_def_projections()
    edges = odds_props.score_slate(lines, off_proj, def_proj)

    print("\n" + odds_props.format_slate(edges, min_edge=args.min_edge))

    strong = [e for e in edges if e.mkt_edge >= args.min_edge]
    parlays = correlations.suggest_parlays(strong, min_leg_prob=0.55)
    if parlays:
        print("\n" + correlations.format_parlays(parlays))

    if args.notify and strong:
        notify.build_notifier(args.notify).send(
            f"NFL prop value ({len(strong)})",
            odds_props.format_slate(edges, min_edge=args.min_edge, top=10))
        print(f"\n[notify:{args.notify}] sent")


if __name__ == "__main__":
    main()
