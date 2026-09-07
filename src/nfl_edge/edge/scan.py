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
from .teams import name as team_name, norm_abbr, short_name

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
    home: str                 # home team abbr
    away: str                 # away team abbr
    line: Optional[float]     # strike for SPREAD/TOTAL, None for ML
    fair_prob: float
    kalshi_ask: float
    fee: float
    cost: float
    edge: float               # fair_prob - cost (EV per $1 contract)
    edge_pct: float
    kelly: float              # full-Kelly fraction (cap this!)
    n_books: int
    close_time: str           # Kalshi settlement time (NOT kickoff)
    kickoff: str              # book commence_time -- the real game time
    ticker: str

    def key(self) -> str:
        return self.ticker

    # -- human-readable helpers -------------------------------------------
    def matchup(self) -> str:
        """Away @ Home in abbreviations, e.g. 'TB @ NO'."""
        if self.away and self.home:
            return f"{self.away} @ {self.home}"
        return self.event

    def where(self) -> str:
        """Which side of the bet's team is at home."""
        if self.team and self.team == self.home:
            return "home"
        if self.team and self.team == self.away:
            return "away"
        return ""

    def game_date(self) -> str:
        import datetime as _dt
        d = None
        try:
            d = _dt.date.fromisoformat((self.kickoff or "")[:10])
        except (ValueError, TypeError):
            d = event_date(self.event)          # fall back to the Kalshi ticker date
        if not d:
            return (self.kickoff or self.close_time or "")[:10]
        return d.strftime("%a %b ") + str(d.day)   # 'Sun Sep 14' (portable)

    def bet_text(self) -> str:
        """Plain-English description of the wager."""
        if self.market == "ML":
            return f"{team_name(self.team)} win outright"
        if self.market == "SPREAD":
            need = int(self.line + 0.5)   # >7.5 means win by 8+
            opp = short_name(self.opponent)
            return f"{team_name(self.team)} beat {opp} by {need}+ pts"
        if self.market == "TOTAL":
            pts = int(self.line + 0.5)
            return f"Combined score {pts}+ (Over {self.line})"
        return self.selection


def kelly_fraction(fair: float, cost: float) -> float:
    """Kelly for a binary contract costing `cost` that pays $1: f = (p - c)/(1 - c)."""
    if cost >= 1.0:
        return 0.0
    return max(0.0, (fair - cost) / (1.0 - cost))


def _row(market, event, selection, team, opponent, home, away, line, fair, ask,
         fee_rate, n_books, close_time, kickoff, ticker) -> EdgeRow:
    fee = fee_per_contract(ask, fee_rate)
    cost = ask + fee
    edge = fair - cost
    return EdgeRow(
        market=market, event=event, selection=selection, team=team,
        opponent=opponent, home=home or "", away=away or "", line=line,
        fair_prob=round(fair, 4), kalshi_ask=ask, fee=round(fee, 4),
        cost=round(cost, 4), edge=round(edge, 4), edge_pct=round(edge * 100, 2),
        kelly=round(kelly_fraction(fair, cost), 4), n_books=n_books,
        close_time=close_time or "", kickoff=kickoff or "", ticker=ticker,
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
            rows.append(_row("ML", event, code, code, opp,
                             game.get("home"), game.get("away"), None, fair,
                             m.yes_ask, fee_rate, game.get("n_books", 0),
                             m.close_time, game.get("commence_time"), m.ticker))
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
                         team, opp, game.get("home"), game.get("away"),
                         m.floor_strike, fair, m.yes_ask, fee_rate,
                         game.get("n_books", 0), m.close_time,
                         game.get("commence_time"), m.ticker))
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
                         "", "", game.get("home"), game.get("away"),
                         m.floor_strike, fair, m.yes_ask, fee_rate,
                         game.get("n_books", 0), m.close_time,
                         game.get("commence_time"), m.ticker))
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


_MARKET_LABEL = {"ML": "MONEYLINE", "SPREAD": "SPREAD", "TOTAL": "TOTAL"}


def format_board(rows: list[EdgeRow], limit: int | None = None) -> str:
    """Human-readable edge list for Discord / console. One labeled block per bet:
    what the game is, who's home, the bet in plain English, and the numbers."""
    if not rows:
        return "No edges above threshold right now."
    shown = rows if limit is None else rows[:limit]
    blocks = []
    for i, r in enumerate(shown, 1):
        ask_c = int(round(r.kalshi_ask * 100))
        quarter = r.kelly / 4 * 100                 # practical quarter-Kelly stake
        home_note = f"  ({r.team} at home)" if r.where() == "home" else \
                    f"  ({r.team} on road)" if r.where() == "away" else ""
        blocks.append(
            f"#{i}  {_MARKET_LABEL.get(r.market, r.market)}   {r.matchup()}   {r.game_date()}\n"
            f"    Bet: {r.bet_text()}{home_note}\n"
            f"    Buy at {ask_c}c  ->  fair value {r.fair_prob*100:.0f}%   "
            f"|  EDGE +{r.edge_pct:.1f}%   |  stake ~{quarter:.1f}% of bankroll"
        )
    out = "\n\n".join(blocks)
    if limit is not None and len(rows) > limit:
        out += f"\n\n...and {len(rows) - limit} more."
    out += ("\n\nHOW TO READ: '@' = away team @ home team.  EDGE = expected profit "
            "per $1 staked.  'stake' = suggested bet size (quarter-Kelly); never bet "
            "more than you'd lose comfortably.")
    return out


def rows_to_dicts(rows: list[EdgeRow]) -> list[dict]:
    return [asdict(r) for r in rows]
