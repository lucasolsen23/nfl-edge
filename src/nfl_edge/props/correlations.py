"""Correlated same-game parlays for pick'em (PrizePicks / Underdog).

Pick'em platforms price each leg independently and pay a fixed multiplier. A
parlay is +EV only if the JOINT hit probability beats the break-even (1/payout).
Positively-correlated legs bet in the SAME direction hit together more often than
independence implies -> the parlay is better than it looks. Opposite-direction
bets on correlated stats hit together LESS -> worse. This module gets the joint
probability right so you stack the good combos and avoid the traps.

Correlations are the empirical residual correlations measured from 2015-2025
projections (scripts/estimate_correlations.py), i.e. what's left AFTER our model,
which is exactly the part the platform isn't pricing.

Method: a Gaussian copula. Each leg's performance is a latent standard normal;
we orient it toward "hit" (+ for OVER, - for UNDER) so the correlation sign
follows the bet direction, then Monte-Carlo the joint probability that all legs
clear their lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

# empirical residual correlations (see estimate_correlations.py)
CORR_SAME_PLAYER_RECYDS_REC = 0.75      # a player's rec yards <-> receptions
CORR_SAME_PLAYER_OTHER = 0.40           # same player, other stat pair (prior)
CORR_QB_PASS_TEAMMATE_REC = 0.21        # QB pass yds <-> his WR/TE receiving
CORR_OPP_QB_QB = 0.09                   # opposing QBs (shootout)
# everything else (two WRs, RB<->QB, cross-game) ~ 0

# typical PrizePicks power-play payouts (flex/standard vary; override as needed)
POWER_PAYOUTS = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 25.0}

_RECV = {"receiving_yards", "receptions"}


@dataclass
class Leg:
    player_id: str
    team: str
    opponent: str
    stat: str
    side: str            # 'OVER' or 'UNDER'
    prob: float          # model P(hit) for this leg
    season: int
    week: int
    label: str = ""

    def same_game(self, o: "Leg") -> bool:
        return (self.season, self.week) == (o.season, o.week) and \
               {self.team, self.opponent} == {o.team, o.opponent}


def base_correlation(a: Leg, b: Leg) -> float:
    """Correlation of the two underlying PERFORMANCES (ignoring bet direction)."""
    if not a.same_game(b):
        return 0.0
    if a.player_id == b.player_id:
        if {a.stat, b.stat} == _RECV:
            return CORR_SAME_PLAYER_RECYDS_REC
        return CORR_SAME_PLAYER_OTHER
    if a.team == b.team:
        stats = {a.stat, b.stat}
        if "passing_yards" in stats and (stats & _RECV):
            return CORR_QB_PASS_TEAMMATE_REC
        return 0.0
    # opposing players
    if a.stat == "passing_yards" and b.stat == "passing_yards":
        return CORR_OPP_QB_QB
    return 0.0


def correlation_matrix(legs: list[Leg]) -> np.ndarray:
    """Correlation of the direction-oriented latent variables: base correlation
    times the product of bet-direction signs (OVER=+1, UNDER=-1)."""
    n = len(legs)
    sign = [1.0 if lg.side == "OVER" else -1.0 for lg in legs]
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = base_correlation(legs[i], legs[j]) * sign[i] * sign[j]
            R[i, j] = R[j, i] = r
    return R


def _psd(R: np.ndarray) -> np.ndarray:
    """Nudge to the nearest usable PSD matrix if needed (for Cholesky)."""
    try:
        np.linalg.cholesky(R)
        return R
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(R)
        w = np.clip(w, 1e-6, None)
        R2 = V @ np.diag(w) @ V.T
        d = np.sqrt(np.diag(R2))
        return R2 / np.outer(d, d)


def parlay_probability(legs: list[Leg], n: int = 200_000, seed: int = 0) -> float:
    """Monte-Carlo joint probability that every leg hits, via a Gaussian copula."""
    if not legs:
        return 0.0
    R = _psd(correlation_matrix(legs))
    L = np.linalg.cholesky(R)
    thr = np.array([NormalDist().inv_cdf(1 - lg.prob) for lg in legs])
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(legs))) @ L.T
    return float(np.mean(np.all(z > thr, axis=1)))


def independent_probability(legs: list[Leg]) -> float:
    p = 1.0
    for lg in legs:
        p *= lg.prob
    return p


@dataclass
class ParlayEval:
    legs: list[Leg]
    joint: float
    independent: float
    payout: float

    @property
    def ev(self) -> float:                     # EV per $1 staked
        return self.joint * self.payout - 1.0

    @property
    def breakeven(self) -> float:
        return 1.0 / self.payout

    @property
    def lift(self) -> float:                    # how much correlation helps
        return self.joint - self.independent


def evaluate_parlay(legs: list[Leg], payout: float | None = None) -> ParlayEval:
    pay = payout if payout is not None else POWER_PAYOUTS.get(len(legs), 1.0)
    return ParlayEval(legs, parlay_probability(legs), independent_probability(legs), pay)


# --------------------------------------------------------------------------
# Suggest the best parlays from a slate of scored single-leg picks
# --------------------------------------------------------------------------
def leg_from(edge) -> Leg:
    """Build a Leg from a scan_props.PropEdge (duck-typed)."""
    return Leg(
        player_id=getattr(edge, "player_id", edge.player), team=edge.team,
        opponent=edge.opponent, stat=edge.stat, side=edge.pick, prob=edge.prob,
        season=edge.season, week=edge.week,
        label=f"{edge.player} {edge.pick} {edge.line} {edge.stat.replace('_', ' ')}",
    )


def suggest_parlays(edges, sizes=(2, 3), payouts=None, min_leg_prob: float = 0.55,
                    top: int = 8) -> list[ParlayEval]:
    """Rank +EV parlays from scored picks. Only legs with prob >= min_leg_prob are
    used, and every parlay must contain at least one correlated pair (the point)."""
    import itertools
    payouts = payouts or POWER_PAYOUTS
    legs = [leg_from(e) for e in edges if e.prob >= min_leg_prob]
    evals: list[ParlayEval] = []
    for size in sizes:
        for combo in itertools.combinations(legs, size):
            if not any(base_correlation(a, b) != 0.0
                       for a, b in itertools.combinations(combo, 2)):
                continue                      # require genuine correlation
            evals.append(evaluate_parlay(list(combo), payouts.get(size)))
    evals = [e for e in evals if e.ev > 0]
    evals.sort(key=lambda e: e.ev, reverse=True)
    return evals[:top]


def format_parlays(evals: list[ParlayEval]) -> str:
    if not evals:
        return "No +EV correlated parlays found in these legs."
    out = ["Correlated parlays (joint prob accounts for same-game correlation)\n"]
    for i, e in enumerate(evals, 1):
        out.append(
            f"#{i}  {len(e.legs)}-leg @ {e.payout:g}x   "
            f"hit {e.joint*100:.1f}%  (indep {e.independent*100:.1f}%, "
            f"+{e.lift*100:.1f} from correlation)   EV {e.ev*100:+.1f}%"
        )
        for lg in e.legs:
            out.append(f"     - {lg.label}  ({lg.prob*100:.0f}%)")
    return "\n".join(out)
