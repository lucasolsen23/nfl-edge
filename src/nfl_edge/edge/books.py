"""The Odds API client -> de-vigged book-consensus fair probabilities.

Books are the REFERENCE price (we can't bet them in Utah). We strip each book's
vig on the two-way moneyline, then average the fair probabilities across books to
get a consensus true win probability per team. That consensus is what we compare
Kalshi against.

Free tier: 500 requests/month. One /odds call for h2h/us = 1 credit, so polling
every ~30 min through a game week stays well inside the quota.
"""

from __future__ import annotations

import datetime as _dt
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


def _game_date(g: dict) -> Optional[_dt.date]:
    ct = g.get("commence_time")
    if not ct:
        return None
    try:
        return _dt.date.fromisoformat(ct[:10])
    except ValueError:
        return None


def _one_game_consensus(g: dict) -> Optional[dict]:
    """Reduce a single Odds API game to one consensus record, or None if it
    can't be priced. IMPORTANT: keyed with a date so multi-week feeds don't
    collide (the Odds API returns the whole season)."""
    ha, aa = abbr(g.get("home_team", "")), abbr(g.get("away_team", ""))
    if not ha or not aa:
        return None
    ha, aa = norm_abbr(ha), norm_abbr(aa)

    fair_acc: dict[str, list[float]] = defaultdict(list)
    margin_acc: dict[str, list[float]] = defaultdict(list)
    total_acc: list[float] = []
    for bk in g.get("bookmakers", []):
        for mk in bk.get("markets", []):
            key = mk.get("key")
            if key == "h2h":
                price = {abbr(o["name"]): o["price"] for o in mk.get("outcomes", [])
                         if abbr(o.get("name", ""))}
                if ha in price and aa in price:
                    fh, fa = odds_math.devig_two_way(price[ha], price[aa])
                    fair_acc[ha].append(fh)
                    fair_acc[aa].append(fa)
            elif key == "spreads":
                for o in mk.get("outcomes", []):
                    ab = abbr(o.get("name", ""))
                    if ab and o.get("point") is not None:
                        # a team laying -3 is expected to win by 3 -> margin = -point
                        margin_acc[norm_abbr(ab)].append(-float(o["point"]))
            elif key == "totals":
                o = next((x for x in mk.get("outcomes", [])
                          if x.get("name", "").lower() == "over"), None)
                if o and o.get("point") is not None:
                    total_acc.append(float(o["point"]))

    if ha not in fair_acc or aa not in fair_acc:
        return None
    ph, pa = sum(fair_acc[ha]) / len(fair_acc[ha]), sum(fair_acc[aa]) / len(fair_acc[aa])
    tot = ph + pa
    rec = {
        "date": _game_date(g),
        "probs": {ha: ph / tot, aa: pa / tot},
        "commence_time": g.get("commence_time"),
        "home": ha, "away": aa, "n_books": len(fair_acc[ha]),
    }
    if ha in margin_acc and aa in margin_acc:
        rec["margin"] = {ha: sum(margin_acc[ha]) / len(margin_acc[ha]),
                         aa: sum(margin_acc[aa]) / len(margin_acc[aa])}
    if total_acc:
        rec["total"] = sum(total_acc) / len(total_acc)
    return rec


def consensus(games: list[dict]) -> dict[frozenset, list[dict]]:
    """Consensus per game, grouped by team-pair. Because the Odds API returns
    the whole season, each team-pair maps to a LIST of weekly games (each with
    its 'date'); callers pick the one whose date matches the Kalshi contract.

    frozenset({home_abbr, away_abbr}) -> [ {date, probs, margin?, total?, ...}, ... ]
    """
    out: dict[frozenset, list[dict]] = defaultdict(list)
    for g in games:
        rec = _one_game_consensus(g)
        if rec is None:
            continue
        out[frozenset({rec["home"], rec["away"]})].append(rec)
    return dict(out)


# backward-compatible alias (moneyline callers)
consensus_fair_probs = consensus
