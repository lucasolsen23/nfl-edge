"""Live Kalshi edge scanner.

Compares Kalshi NFL contract prices to de-vigged book-consensus fair value and
alerts on +EV bets.

Setup:
    setx ODDS_API_KEY  your_key           # from the-odds-api.com (free tier)
    # optional, for phone alerts:
    setx TELEGRAM_BOT_TOKEN  ...
    setx TELEGRAM_CHAT_ID    ...

Usage:
    python scripts/scan_kalshi.py                     # one scan, print board
    python scripts/scan_kalshi.py --notify telegram   # + push new edges
    python scripts/scan_kalshi.py --loop 900          # poll every 15 min
    python scripts/scan_kalshi.py --demo              # no key: uses Kalshi mid as
                                                      # pseudo fair-value (format demo)
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_edge.edge import books, kalshi, notify, scan  # noqa: E402
from nfl_edge.edge.teams import norm_abbr  # noqa: E402


def _demo_book_probs(mk):
    """Fabricate 'fair' probs from Kalshi's own de-vigged mid so the pipeline can
    be exercised without an Odds API key. NOT a real edge source."""
    out = {}
    for event, ms in kalshi.group_by_event(mk).items():
        mids = {}
        for m in ms:
            if m.yes_bid is not None and m.yes_ask is not None:
                mids[norm_abbr(m.team_code)] = (m.yes_bid + m.yes_ask) / 2
        if len(mids) == 2:
            tot = sum(mids.values())
            out[frozenset(mids)] = {"probs": {k: v / tot for k, v in mids.items()},
                                    "commence_time": "", "home": "", "away": "",
                                    "n_books": 0}
    return out


def run_once(args) -> None:
    mk = kalshi.nfl_moneyline_markets(status="open")
    if args.demo:
        book_probs = _demo_book_probs(mk)
        print("[demo] using Kalshi mid as pseudo fair-value -- edges here are noise")
    else:
        key = os.environ.get("ODDS_API_KEY")
        if not key:
            sys.exit("ODDS_API_KEY not set. Get a free key at the-odds-api.com, "
                     "or run with --demo to see the format.")
        games = books.fetch_nfl_odds(key)
        book_probs = books.consensus_fair_probs(games)
        q = getattr(books.fetch_nfl_odds, "last_quota", {})
        print(f"[books] {len(book_probs)} games priced  (quota remaining: {q.get('remaining')})")

    rows = scan.scan(mk, book_probs, min_edge=args.min_edge, fee_rate=args.fee_rate)
    print(f"\n{len(rows)} edge(s) >= {args.min_edge*100:.0f}%  "
          f"(scanned {len(kalshi.group_by_event(mk))} games)\n")
    print(scan.format_board(rows))

    if args.notify and rows:
        fresh = notify.filter_new(rows)
        if fresh:
            n = notify.build_notifier(args.notify)
            n.send(f"NFL Kalshi edges ({len(fresh)} new)", scan.format_board(fresh))
            print(f"\n[notify:{args.notify}] sent {len(fresh)} new alert(s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=0.03, help="min EV per contract")
    ap.add_argument("--fee-rate", type=float, default=0.07)
    ap.add_argument("--notify", choices=["console", "telegram", "discord", "email"])
    ap.add_argument("--loop", type=int, default=0, help="poll interval in seconds")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.loop:
        print(f"polling every {args.loop}s (Ctrl+C to stop)")
        while True:
            try:
                run_once(args)
            except Exception as e:  # keep the loop alive across transient errors
                print(f"[error] {e!r}")
            time.sleep(args.loop)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
