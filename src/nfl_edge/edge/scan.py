"""Edge scan: Kalshi price vs book-consensus fair value.

Covers all three NFL market types on Kalshi:
    ML     -- "TEAM wins"        vs de-vigged book moneyline
    SPREAD -- "TEAM by over X"   vs P(margin > X) from consensus spread + normal
    TOTAL  -- "over X points"    vs P(total  > X) from consensus total  + normal

For every contract: fair value of a $1 payout (the book-implied probability) minus
the all-in Kalshi cost (ask + fee) is the expected value per contract. We also
report a full-Kelly stake (size a FRACTION of it).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from . import dist
from .fees import fee_per_contract
from .kalshi import KalshiMarket, KalshiLadderMarket, event_date, group_by_event
from .teams import norm_abbr

# Kalshi ticker date vs book commence date can differ by up to a day for
# night games (UTC rollover); 3 days is safe and never spans two NFL weeks.
_MATCH_TOL_DAYS = 3


@dataclass
class EdgeRow:
    market: str               # 'ML' | 'SPREAD' | 'TOTAL'
    event: str
    selection: str            # human label, e.g. 'KC', 'KC by >3.5', 'Over 47.5'
    team: str                 # abbr for ML/SPREAD, '' for TOTAL
    opponent: str
    line: Optional[float]     # strike for SPREAD/TOTAL, None for ML
    fair_prob: float
    kalshi_ask: float
    fee: float
    cost: float
    edge: float               # fair_prob - cost (EV per $1 contract)
    edge_pct: float
    kelly: float              # full-Kelly fraction (cap this!)
    n_books: int
    close_time: str
    ticker: str

    def key(self) -> str:
        return self.ticker


def kelly_fraction(fair: float, cost: float) -> float:
    """Kelly for a binary contract costing `cost` that pays $1: f = (p - c)/(1 - c)."""
    if cost >= 1.0:
        return 0.0
    return max(0.0, (fair - cost) / (1.0 - cost))


def _row(market, event, selection, team, opponent, line, fair, ask, fee_rate,
         n_books, close_time, ticker) -> EdgeRow:
    fee = fee_per_contract(ask, fee_rate)
    cost = ask + fee
    edge = fair - cost
    return EdgeRow(
        market=market, event=event, selection=selection, team=team,
        opponent=opponent, line=line, fair_prob=round(fair, 4), kalshi_ask=ask,
        fee=round(fee, 4), cost=round(cost, 4), edge=round(edge, 4),
        edge_pct=round(edge * 100, 2), kelly=round(kelly_fraction(fair, cost), 4),
        n_books=n_books, close_time=close_time or "", ticker=ticker,
    )


def find_game(cons: dict, event_ticker: str) -> Optional[dict]:
    """Find the consensus record for a Kalshi event, matching on BOTH team codes
    present in the ticker AND the closest date. This is what prevents the
    multi-week Odds API feed from pairing a contract with the wrong week."""
    tdate = event_date(event_ticker)
    best, best_diff = None, 10 ** 9
    for key, recs in cons.items():
        if not all(code in event_ticker for code in key):
            continue
        for rec in recs:
            d = rec.get("date")
            diff = abs((d - tdate).days) if (d and tdate) else 10 ** 9
            if diff < best_diff:
                best, best_diff = rec, diff
    return best if best is not None and best_diff <= _MATCH_TOL_DAYS else None


# ---- moneyline ------------------------------------------------------------
def scan(kalshi_markets: list[KalshiMarket], cons: dict,
         min_edge: float = 0.03, fee_rate: float = 0.07) -> list[EdgeRow]:
    rows: list[EdgeRow] = []
    for event, ms in group_by_event(kalshi_markets).items():
        codes = [norm_abbr(m.team_code) for m in ms]
        game = find_game(cons, event)
        if not game or len(codes) < 2:
            continue
        for m in ms:
            code = norm_abbr(m.team_code)
            fair = game.get("probs", {}).get(code)
            if fair is None or m.yes_ask is None:
                continue
            opp = next((c for c in codes if c != code), "?")
            rows.append(_row("ML", event, code, code, opp, None, fair, m.yes_ask,
                             fee_rate, game.get("n_books", 0), m.close_time, m.ticker))
    return _finish(rows, min_edge)


# ---- spreads --------------------------------------------------------------
def scan_spreads(markets: list[KalshiLadderMarket], cons: dict,
                 min_edge: float = 0.03, fee_rate: float = 0.07,
                 sd: float = dist.MARGIN_SD) -> list[EdgeRow]:
    rows: list[EdgeRow] = []
    for m in markets:
        if m.yes_ask is None or not m.team_code:
            continue
        team = norm_abbr(m.team_code)
        game = find_game(cons, m.event_ticker)
        if not game or team not in (game.get("home"), game.get("away")):
            continue
        margin = game.get("margin", {}).get(team)
        if margin is None:
            continue
        opp = game["home"] if game["away"] == team else game["away"]
        fair = dist.prob_margin_over(margin, m.floor_strike, sd)
        rows.append(_row("SPREAD", m.event_ticker, f"{team} by >{m.floor_strike}",
                         team, opp, m.floor_strike, fair, m.yes_ask, fee_rate,
                         game.get("n_books", 0), m.close_time, m.ticker))
    return _finish(rows, min_edge)


# ---- totals ---------------------------------------------------------------
def scan_totals(markets: list[KalshiLadderMarket], cons: dict,
                min_edge: float = 0.03, fee_rate: float = 0.07,
                sd: float = dist.TOTAL_SD) -> list[EdgeRow]:
    rows: list[EdgeRow] = []
    for m in markets:
        if m.yes_ask is None:
            continue
        game = find_game(cons, m.event_ticker)
        if not game or game.get("total") is None:
            continue
        fair = dist.prob_total_over(game["total"], m.floor_strike, sd)
        rows.append(_row("TOTAL", m.event_ticker, f"Over {m.floor_strike}",
                         "", "", m.floor_strike, fair, m.yes_ask, fee_rate,
                         game.get("n_books", 0), m.close_time, m.ticker))
    return _finish(rows, min_edge)


def scan_all(ml, spreads, totals, cons, min_edge=0.03, fee_rate=0.07) -> list[EdgeRow]:
    rows = (scan(ml, cons, min_edge, fee_rate)
            + scan_spreads(spreads, cons, min_edge, fee_rate)
            + scan_totals(totals, cons, min_edge, fee_rate))
    rows.sort(key=lambda r: r.edge, reverse=True)
    return rows


def _finish(rows: list[EdgeRow], min_edge: float) -> list[EdgeRow]:
    rows.sort(key=lambda r: r.edge, reverse=True)
    return [r for r in rows if r.edge >= min_edge]


def format_board(rows: list[EdgeRow]) -> str:
    if not rows:
        return "No edges above threshold."
    lines = [f"{'MKT':6} {'SELECTION':16} {'FAIR':>6} {'ASK':>5} {'COST':>6} "
             f"{'EDGE':>6} {'KELLY':>6}  CLOSE"]
    for r in rows:
        lines.append(
            f"{r.market:6} {r.selection:16} {r.fair_prob:6.3f} {r.kalshi_ask:5.2f} "
            f"{r.cost:6.3f} {r.edge_pct:5.1f}% {r.kelly*100:5.1f}%  {r.close_time[:10]}"
        )
    return "\n".join(lines)


def rows_to_dicts(rows: list[EdgeRow]) -> list[dict]:
    return [asdict(r) for r in rows]
