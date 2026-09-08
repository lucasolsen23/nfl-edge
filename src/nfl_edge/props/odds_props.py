"""Auto prop-line source: The Odds API player props (book lines).

PrizePicks' own API is Cloudflare-blocked (and ToS-risky), so we pull player
props from the books via The Odds API. Two wins:
  * real prop lines, auto-fetched (they track PrizePicks closely) -- no typing
  * de-vigged book consensus = a fair value, so we score where OUR model
    disagrees with the whole market (mkt_edge), the same logic as the game scanner

Cost: player props are per-event. Each market x region = 1 credit PER event, so
we fetch only the upcoming slate and let the caller pick markets. Set the key in
the ODDS_API_KEY env var.
"""

from __future__ import annotations

import datetime as dt
import os
from collections import Counter, defaultdict
from typing import Optional

import requests

from .. import odds as odds_math
from . import projections as P
from . import defense as D
from .scan_props import PropEdge

BASE = "https://api.the-odds-api.com/v4"

# Odds API market key -> our stat key
MARKET_MAP = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
    "player_tackles_assists": "tackles",
    "player_sacks": "sacks",
}
DEFAULT_MARKETS = ["player_pass_yds", "player_rush_yds",
                   "player_reception_yds", "player_receptions"]


def _norm(name: str) -> str:
    return name.lower().replace(".", "").replace("'", "").strip()


def upcoming_events(api_key: str, days: int = 8) -> list[dict]:
    r = requests.get(f"{BASE}/sports/americanfootball_nfl/events",
                     params={"apiKey": api_key}, timeout=20)
    r.raise_for_status()
    now = dt.datetime.now(dt.timezone.utc)
    horizon = now + dt.timedelta(days=days)
    out = []
    for e in r.json():
        try:
            ct = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        except (ValueError, KeyError):
            continue
        if now <= ct <= horizon:
            out.append(e)
    return out


def fetch_event_props(api_key: str, event_id: str, markets: list[str],
                      session: Optional[requests.Session] = None) -> list[dict]:
    s = session or requests.Session()
    r = s.get(f"{BASE}/sports/americanfootball_nfl/events/{event_id}/odds",
              params={"apiKey": api_key, "regions": "us",
                      "markets": ",".join(markets), "oddsFormat": "american"},
              timeout=25)
    if r.status_code != 200:
        return []
    fetch_event_props.quota = r.headers.get("x-requests-remaining")
    return consensus_from_bookmakers(r.json().get("bookmakers", []))


def consensus_from_bookmakers(bookmakers: list[dict]) -> list[dict]:
    """Pure parser: bookmakers -> consensus lines with de-vigged over prob.
    Consensus line = the modal point; book_fair_over = mean de-vigged over prob
    across books offering that line."""
    # per (player, stat): list of (point, over_price, under_price) across books
    acc: dict[tuple, list] = defaultdict(list)
    for bk in bookmakers:
        for mk in bk.get("markets", []):
            stat = MARKET_MAP.get(mk.get("key"))
            if not stat:
                continue
            sides: dict[str, dict] = defaultdict(dict)
            for o in mk.get("outcomes", []):
                sides[o["description"]][o["name"].lower()] = (o.get("price"), o.get("point"))
            for player, sd in sides.items():
                if "over" in sd and "under" in sd and sd["over"][1] is not None:
                    acc[(player, stat)].append(
                        (sd["over"][1], sd["over"][0], sd["under"][0]))
    lines = []
    for (player, stat), rows in acc.items():
        point = Counter(r[0] for r in rows).most_common(1)[0][0]
        fair = [odds_math.devig_two_way(o, u)[0]
                for pt, o, u in rows if pt == point and o and u]
        if not fair:
            continue
        lines.append({"player": player, "stat": stat, "line": float(point),
                      "book_fair_over": sum(fair) / len(fair), "n_books": len(fair)})
    return lines


def fetch_slate_props(api_key: str, markets: list[str] | None = None,
                      days: int = 8) -> tuple[list[dict], dict]:
    markets = markets or DEFAULT_MARKETS
    events = upcoming_events(api_key, days)
    all_lines = []
    for e in events:
        all_lines.extend(fetch_event_props(api_key, e["id"], markets))
    meta = {"events": len(events), "markets": len(markets),
            "est_credits": len(events) * len(markets),
            "quota_remaining": getattr(fetch_event_props, "quota", None)}
    return all_lines, meta


# --------------------------------------------------------------------------
# Score book lines against our model: edge = our prob - de-vigged book prob
# --------------------------------------------------------------------------
def _latest(df):
    return {} if df.height == 0 else {
        (_norm(r["player_display_name"]), r["stat"]): r
        for r in df.sort(["season", "week"]).group_by(["player_id", "stat"]).last().to_dicts()
    }


# The book line is a SHARP consensus, so we treat it as a strong prior and only
# lean our model's way partway. And when our raw model disagrees with the book by
# more than MAX_DISAGREE, that's almost always our model being wrong (small sample
# / role change), not a real edge -- so we drop it.
BLEND_WEIGHT = 0.40          # weight on our model; rest defers to the book
MAX_DISAGREE = 0.22          # raw |model_over - book_over| above this = skip


def score_slate(lines: list[dict], off_proj, def_proj,
                blend: float = BLEND_WEIGHT) -> list[PropEdge]:
    look = {**_latest(off_proj), **_latest(def_proj)}
    edges: list[PropEdge] = []
    for ln in lines:
        r = look.get((_norm(ln["player"]), ln["stat"]))
        if not r:
            continue
        stat, mean, sd = ln["stat"], r["proj_mean"], r["proj_sd"]
        model_over = (D.prob_over(stat, mean, sd, ln["line"]) if stat in D.DEF_STATS
                      else P.prob_over(mean, sd, ln["line"], stat=stat))
        book_over = ln["book_fair_over"]
        if abs(model_over - book_over) > MAX_DISAGREE:
            continue                                   # implausible -> model error
        # market-anchored probability
        final_over = blend * model_over + (1 - blend) * book_over
        if final_over >= book_over:
            pick, prob, book_pick = "OVER", final_over, book_over
        else:
            pick, prob, book_pick = "UNDER", 1 - final_over, 1 - book_over
        edges.append(PropEdge(
            player=r["player_display_name"], stat=stat, line=ln["line"], pick=pick,
            prob=round(prob, 4), proj_mean=round(mean, 2), proj_sd=round(sd, 2),
            opponent=r["opponent_team"], season=r["season"], week=r["week"],
            note=r.get("note", "") or "", opp_factor=r.get("opp_factor", 1.0),
            player_id=r["player_id"], team=r.get("team2026", ""),
            book_prob=round(book_pick, 4), mkt_edge=round(prob - book_pick, 4),
        ))
    edges.sort(key=lambda e: e.mkt_edge, reverse=True)
    return edges


def format_slate(edges: list[PropEdge], min_edge: float = 0.04, top: int = 20) -> str:
    picks = [e for e in edges if e.mkt_edge >= min_edge][:top]
    if not picks:
        return "No props where the model beats the book by the threshold."
    out = [f"Model vs book — value picks (edge >= {min_edge*100:.0f} pts)\n"]
    for i, e in enumerate(picks, 1):
        note = f"  [{e.note}]" if e.note else ""
        out.append(
            f"#{i} {e.player} — {e.stat.replace('_', ' ')}: {e.pick} {e.line} vs {e.opponent}{note}\n"
            f"    model {e.prob*100:.0f}%  vs book {e.book_prob*100:.0f}%  "
            f"=> edge +{e.mkt_edge*100:.1f} pts")
    return "\n".join(out)
