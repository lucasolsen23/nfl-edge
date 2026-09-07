"""Score prop lines against the projection model.

Given lines (player, stat, line) -- typed in from PrizePicks/Underdog, or from a
book props feed -- we compute the calibrated P(over)/P(under), pick the stronger
side, and rank by confidence.

PrizePicks / Underdog are pick'em: you need each leg above a break-even that
depends on the payout. Common power-play break-evens (per leg, if legs were
independent):
    2 picks @ 3x  -> 57.7%      3 picks @ 5x  -> 58.5%
    4 picks @ 10x -> 56.2%      5 picks @ 20x -> 54.9%
So only legs whose model probability clears the break-even are +EV -- and
correlated legs (e.g. a QB's yards + his team winning) beat independent pricing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import polars as pl

from . import projections as P

PICKEM_BREAKEVEN = {2: 0.577, 3: 0.585, 4: 0.562, 5: 0.549, 6: 0.560}


@dataclass
class PropEdge:
    player: str
    stat: str
    line: float
    pick: str            # 'OVER' or 'UNDER'
    prob: float          # model probability of the picked side
    proj_mean: float
    proj_sd: float
    opponent: str
    season: int
    week: int
    note: str = ""
    opp_factor: float = 1.0

    def edge_vs(self, breakeven: float) -> float:
        return round(self.prob - breakeven, 4)


def latest_projections(projections: pl.DataFrame) -> pl.DataFrame:
    """Most recent projection per (player, stat) -- a proxy for current form.
    (A true live scan projects the upcoming opponent; this uses the last game's.)"""
    return (projections.sort(["season", "week"])
            .group_by(["player_id", "stat"]).last())


def score_lines(lines: list[dict], projections: pl.DataFrame) -> list[PropEdge]:
    """lines: [{'player': name, 'stat': 'receiving_yards', 'line': 45.5}, ...]"""
    latest = latest_projections(projections)
    edges: list[PropEdge] = []
    for ln in lines:
        stat = ln["stat"]
        name = ln["player"].lower()
        row = latest.filter(
            (pl.col("stat") == stat)
            & (pl.col("player_display_name").str.to_lowercase() == name)
        )
        if row.is_empty():
            continue
        r = row.to_dicts()[0]
        mean, sd = r["proj_mean"], r["proj_sd"]
        p_over = P.prob_over(mean, sd, ln["line"], stat=stat)
        pick, prob = ("OVER", p_over) if p_over >= 0.5 else ("UNDER", 1 - p_over)
        edges.append(PropEdge(
            player=r["player_display_name"], stat=stat, line=ln["line"],
            pick=pick, prob=round(prob, 4), proj_mean=round(mean, 1),
            proj_sd=round(sd, 1), opponent=r["opponent_team"],
            season=r["season"], week=r["week"],
            note=r.get("note", "") or "", opp_factor=r.get("opp_factor", 1.0),
        ))
    edges.sort(key=lambda e: e.prob, reverse=True)
    return edges


def format_props(edges: list[PropEdge], breakeven: float = 0.577) -> str:
    if not edges:
        return "No prop picks (no matching projections)."
    lines = [f"Prop picks (model vs line) — need > {breakeven*100:.0f}% per leg to profit\n"]
    for i, e in enumerate(edges, 1):
        flag = "✅" if e.prob >= breakeven else "  "
        note = f"  [{e.note}]" if e.note else ""
        lines.append(
            f"{flag} #{i} {e.player} — {e.stat.replace('_', ' ')}\n"
            f"     {e.pick} {e.line}  (proj {e.proj_mean}±{e.proj_sd}, vs {e.opponent}){note}\n"
            f"     model {e.prob*100:.0f}%   edge vs break-even {e.edge_vs(breakeven)*100:+.1f} pts"
        )
    return "\n".join(lines)


def edges_to_dicts(edges: list[PropEdge]) -> list[dict]:
    return [asdict(e) for e in edges]
