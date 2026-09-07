"""Kalshi public market-data client (no auth required for reads).

Verified live against the production API:
    base = https://api.elections.kalshi.com/trade-api/v2
    GET /markets?series_ticker=KXNFLGAME&status=open   -> no credentials needed

NFL series:
    KXNFLGAME   single-game winner (moneyline)   <- Phase-3 focus
    KXNFLSPREAD spread
    KXNFLTOTAL  total (over/under)

Each game is an EVENT (e.g. KXNFLGAME-26SEP21NYGLAR) with two markets, one per
team (...-NYG, ...-LAR), each a YES/NO "Team wins" contract. Prices are in the
*_dollars fields (e.g. yes_ask_dollars = 0.55 means 55c to back that team).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"

SERIES = {
    "moneyline": "KXNFLGAME",
    "spread": "KXNFLSPREAD",
    "total": "KXNFLTOTAL",
}


@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: str
    team_code: str            # ticker suffix, nflverse-style abbr (KC, NYG, ...)
    team_name: str            # yes_sub_title
    yes_ask: Optional[float]  # $ to BUY yes (back this team)
    yes_bid: Optional[float]  # $ you'd receive to sell yes
    last_price: Optional[float]
    close_time: Optional[str]
    status: str
    volume: Optional[float]


def _f(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def get_markets(series_ticker: str, status: str = "open",
                session: Optional[requests.Session] = None) -> list[dict]:
    """Fetch all markets for a series, following cursor pagination."""
    s = session or requests.Session()
    out: list[dict] = []
    cursor = None
    while True:
        params = {"series_ticker": series_ticker, "limit": 1000, "status": status}
        if cursor:
            params["cursor"] = cursor
        r = s.get(f"{BASE}/markets", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        out.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def nfl_moneyline_markets(status: str = "open") -> list[KalshiMarket]:
    """Normalized per-team win contracts for upcoming NFL games."""
    raw = get_markets(SERIES["moneyline"], status=status)
    markets: list[KalshiMarket] = []
    for m in raw:
        ticker = m.get("ticker", "")
        team_code = ticker.rsplit("-", 1)[-1] if "-" in ticker else ""
        markets.append(KalshiMarket(
            ticker=ticker,
            event_ticker=m.get("event_ticker", ""),
            team_code=team_code,
            team_name=m.get("yes_sub_title", ""),
            yes_ask=_f(m.get("yes_ask_dollars")),
            yes_bid=_f(m.get("yes_bid_dollars")),
            last_price=_f(m.get("last_price_dollars")),
            close_time=m.get("close_time"),
            status=m.get("status", ""),
            volume=_f(m.get("volume_fp")),
        ))
    return markets


def group_by_event(markets: list[KalshiMarket]) -> dict[str, list[KalshiMarket]]:
    events: dict[str, list[KalshiMarket]] = {}
    for m in markets:
        events.setdefault(m.event_ticker, []).append(m)
    return events


# --------------------------------------------------------------------------
# Spread & total ladders. Each contract is a single strike:
#   spread: "TEAM wins by over floor_strike points"  (strike_type 'greater')
#   total:  "over floor_strike points scored"
# --------------------------------------------------------------------------
@dataclass
class KalshiLadderMarket:
    ticker: str
    event_ticker: str
    kind: str                 # 'spread' or 'total'
    team_code: str            # spread only (nflverse abbr); '' for totals
    floor_strike: float       # the line (e.g. 3.5, 47.5)
    yes_ask: Optional[float]  # $ to buy YES (team covers / game goes over)
    yes_bid: Optional[float]
    last_price: Optional[float]
    close_time: Optional[str]
    status: str
    volume: Optional[float]


_LEAD_ALPHA = re.compile(r"^[A-Za-z]+")


def _ladder_markets(series_key: str, kind: str, status: str) -> list[KalshiLadderMarket]:
    raw = get_markets(SERIES[series_key], status=status)
    out: list[KalshiLadderMarket] = []
    for m in raw:
        fs = _f(m.get("floor_strike"))
        if fs is None:
            continue
        team = ""
        if kind == "spread":
            suffix = m.get("ticker", "").rsplit("-", 1)[-1]   # e.g. 'KC8'
            mt = _LEAD_ALPHA.match(suffix)
            team = mt.group(0).upper() if mt else ""
        out.append(KalshiLadderMarket(
            ticker=m.get("ticker", ""),
            event_ticker=m.get("event_ticker", ""),
            kind=kind,
            team_code=team,
            floor_strike=fs,
            yes_ask=_f(m.get("yes_ask_dollars")),
            yes_bid=_f(m.get("yes_bid_dollars")),
            last_price=_f(m.get("last_price_dollars")),
            close_time=m.get("close_time"),
            status=m.get("status", ""),
            volume=_f(m.get("volume_fp")),
        ))
    return out


def nfl_spread_markets(status: str = "open") -> list[KalshiLadderMarket]:
    """'TEAM wins by over X.5 points' contracts across all upcoming games."""
    return _ladder_markets("spread", "spread", status)


def nfl_total_markets(status: str = "open") -> list[KalshiLadderMarket]:
    """'over X.5 points scored' contracts across all upcoming games."""
    return _ladder_markets("total", "total", status)
