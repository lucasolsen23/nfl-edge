"""Edge scan: Kalshi price vs book-consensus fair value.

For each Kalshi "Team wins" contract we compare the de-vigged book probability
(true value of $1 of payout) against the all-in Kalshi cost (ask + fee). A
positive difference is expected value per contract. We also report a
fractional-Kelly stake so sizing is grounded, not vibes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .fees import fee_per_contract, total_cost
from .kalshi import KalshiMarket, group_by_event
from .teams import norm_abbr


@dataclass
class EdgeRow:
    event: str
    team: str
    opponent: str
    fair_prob: float          # de-vigged book consensus
    kalshi_ask: float         # $ to buy YES
    fee: float                # $ per contract
    cost: float               # ask + fee
    edge: float               # fair_prob - cost  (EV per $1 contract)
    edge_pct: float           # edge * 100
    kelly: float              # full-Kelly fraction of bankroll (cap this!)
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


def scan(kalshi_markets: list[KalshiMarket],
         book_probs: dict,
         min_edge: float = 0.03,
         fee_rate: float = 0.07) -> list[EdgeRow]:
    rows: list[EdgeRow] = []
    for event, ms in group_by_event(kalshi_markets).items():
        codes = [norm_abbr(m.team_code) for m in ms]
        key = frozenset(codes)
        book = book_probs.get(key)
        if not book or len(codes) < 2:
            continue
        for m in ms:
            code = norm_abbr(m.team_code)
            fair = book["probs"].get(code)
            if fair is None or m.yes_ask is None:
                continue
            fee = fee_per_contract(m.yes_ask, fee_rate)
            cost = m.yes_ask + fee
            edge = fair - cost
            opp = next((c for c in codes if c != code), "?")
            rows.append(EdgeRow(
                event=event, team=code, opponent=opp,
                fair_prob=round(fair, 4), kalshi_ask=m.yes_ask,
                fee=round(fee, 4), cost=round(cost, 4),
                edge=round(edge, 4), edge_pct=round(edge * 100, 2),
                kelly=round(kelly_fraction(fair, cost), 4),
                n_books=book["n_books"], close_time=m.close_time or "",
                ticker=m.ticker,
            ))
    rows.sort(key=lambda r: r.edge, reverse=True)
    return [r for r in rows if r.edge >= min_edge]


def format_board(rows: list[EdgeRow]) -> str:
    if not rows:
        return "No edges above threshold."
    lines = [f"{'TEAM':4} {'vs':4} {'FAIR':>6} {'ASK':>5} {'COST':>6} "
             f"{'EDGE':>6} {'KELLY':>6}  CLOSE"]
    for r in rows:
        lines.append(
            f"{r.team:4} {r.opponent:4} {r.fair_prob:6.3f} {r.kalshi_ask:5.2f} "
            f"{r.cost:6.3f} {r.edge_pct:5.1f}% {r.kelly*100:5.1f}%  {r.close_time[:10]}"
        )
    return "\n".join(lines)


def rows_to_dicts(rows: list[EdgeRow]) -> list[dict]:
    return [asdict(r) for r in rows]
