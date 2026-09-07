"""The Odds API client -> de-vigged book-consensus fair probabilities.

Books are the REFERENCE price (we can't bet them in Utah). We strip each book's
vig on the two-way moneyline, then average the fair probabilities across books to
get a consensus true win probability per team. That consensus is what we compare
Kalshi against.

Free tier: 500 requests/month. One /odds call for h2h/us = 1 credit, so polling
every ~30 min through a game week stays well inside the quota.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import requests

from .. import odds as odds_math
from .teams import abbr, norm_abbr

BASE = "https://api.the-odds-api.com/v4"


def fetch_nfl_odds(api_key: str, regions: str = "us", markets: str = "h2h",
                   session: Optional[requests.Session] = None) -> list[dict]:
    s = session or requests.Session()
    r = s.get(
        f"{BASE}/sports/americanfootball_nfl/odds",
        params={"apiKey": api_key, "regions": regions, "markets": markets,
                "oddsFormat": "american"},
        timeout=20,
    )
    r.raise_for_status()
    # remaining-quota headers are handy for logging
    fetch_nfl_odds.last_quota = {
        "remaining": r.headers.get("x-requests-remaining"),
        "used": r.headers.get("x-requests-used"),
    }
    return r.json()


def consensus_fair_probs(games: list[dict]) -> dict[frozenset, dict]:
    """Map frozenset({home_abbr, away_abbr}) -> {
        'probs': {abbr: fair_prob}, 'commence_time', 'home', 'away', 'n_books'}."""
    out: dict[frozenset, dict] = {}
    for g in games:
        ha, aa = abbr(g.get("home_team", "")), abbr(g.get("away_team", ""))
        if not ha or not aa:
            continue
        ha, aa = norm_abbr(ha), norm_abbr(aa)
        fair_acc: dict[str, list[float]] = defaultdict(list)
        for bk in g.get("bookmakers", []):
            h2h = next((m for m in bk.get("markets", []) if m.get("key") == "h2h"), None)
            if not h2h:
                continue
            price = {abbr(o["name"]): o["price"] for o in h2h.get("outcomes", [])
                     if abbr(o.get("name", ""))}
            if ha in price and aa in price:
                fh, fa = odds_math.devig_two_way(price[ha], price[aa])
                fair_acc[ha].append(fh)
                fair_acc[aa].append(fa)
        if ha not in fair_acc or aa not in fair_acc:
            continue
        ph = sum(fair_acc[ha]) / len(fair_acc[ha])
        pa = sum(fair_acc[aa]) / len(fair_acc[aa])
        total = ph + pa
        out[frozenset({ha, aa})] = {
            "probs": {ha: ph / total, aa: pa / total},
            "commence_time": g.get("commence_time"),
            "home": ha, "away": aa,
            "n_books": len(fair_acc[ha]),
        }
    return out
