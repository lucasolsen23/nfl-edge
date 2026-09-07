"""Turn a book's consensus line into a fair probability for any Kalshi strike.

Kalshi offers a fine ladder of spread/total strikes; a book gives one main line.
We model the outcome as a distribution around the consensus line and read the
probability at each strike:

    P(margin > strike) = P(residual > strike - mean_margin)
    P(total  > strike) = P(residual > strike - mean_total)

The residual distribution is EMPIRICAL -- measured from 2015-2025 closing-line
residuals (scripts/calibrate.py -> distributions.json). This matters because NFL
margins pile up on key numbers (3, 7), so a plain normal misprices half-point
strikes across those numbers. If the calibration file is missing we fall back to
a normal with the empirical SD. Pure-stdlib (json, bisect, math) so the Actions
scanner loads it with no heavy deps.
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

# empirical closing-line residual SDs (2015-2025) -- also the normal fallback
MARGIN_SD = 12.7
TOTAL_SD = 13.2

# clamp empirical probabilities away from 0/1 so deep-tail strikes (beyond the
# observed data) can't manufacture a fake 100%-fair edge.
_CLAMP = (0.02, 0.98)


# --------------------------------------------------------------------------
# Normal model (fallback)
# --------------------------------------------------------------------------
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def prob_over(mean: float, strike: float, sd: float) -> float:
    """P(outcome > strike) for outcome ~ Normal(mean, sd)."""
    if sd <= 0:
        return 1.0 if mean > strike else 0.0
    return _phi((mean - strike) / sd)


# --------------------------------------------------------------------------
# Empirical model (preferred)
# --------------------------------------------------------------------------
_DIST_FILE = Path(__file__).with_name("distributions.json")


def _load() -> dict:
    try:
        return json.loads(_DIST_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


_DIST = _load()


def _emp_prob_over(key: str, t: float) -> float | None:
    """Empirical P(residual > t) for 'margin' or 'total'; None if unavailable."""
    d = _DIST.get(key)
    if not d:
        return None
    values, cum, n = d["values"], d["cum"], d["n"]
    idx = bisect.bisect_right(values, t)          # # of distinct grid values <= t
    below = cum[idx - 1] if idx > 0 else 0         # count of residuals <= t
    p = (n - below) / n
    return min(_CLAMP[1], max(_CLAMP[0], p))


def prob_margin_over(mean_margin: float, strike: float, sd: float = MARGIN_SD,
                     empirical: bool = True) -> float:
    """P(a team wins by more than `strike`), given its expected margin."""
    if empirical:
        p = _emp_prob_over("margin", strike - mean_margin)
        if p is not None:
            return p
    return prob_over(mean_margin, strike, sd)


def prob_total_over(mean_total: float, strike: float, sd: float = TOTAL_SD,
                    empirical: bool = True) -> float:
    """P(combined score exceeds `strike`), given the expected total."""
    if empirical:
        p = _emp_prob_over("total", strike - mean_total)
        if p is not None:
            return p
    return prob_over(mean_total, strike, sd)


def using_empirical() -> bool:
    return bool(_DIST.get("margin") and _DIST.get("total"))
