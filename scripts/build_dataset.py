"""Pull data from nflreadpy and cache to data/raw/.

Usage:
    python scripts/build_dataset.py            # schedules only (fast)
    python scripts/build_dataset.py --pbp      # also cache play-by-play (large)
"""

import argparse
import sys
from pathlib import Path

# allow running as a plain script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_edge import ingest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1999)
    ap.add_argument("--end", type=int, default=2025, help="inclusive")
    ap.add_argument("--pbp", action="store_true", help="also cache play-by-play (2015+)")
    args = ap.parse_args()

    ingest.cache_schedules(range(args.start, args.end + 1))
    if args.pbp:
        ingest.cache_pbp(range(max(2015, args.start), args.end + 1))
    print("done.")


if __name__ == "__main__":
    main()
