"""Turn a book's consensus line into a fair probability for any Kalshi strike.

Kalshi offers a fine ladder of spread/total strikes; a book gives one main line.
So we model the outcome as normal around the consensus line and read off the
probability at each Kalshi strike:

    P(margin > strike) = Phi((mean_margin - strike) / sd_margin)
    P(total  > strike) = Phi((mean_total  - strike) / sd_total)

The SDs are estimated empirically from 2015-2025 closing-line residuals
(see scripts / README): margin SD ~= 12.7, total SD ~= 13.2. Both residual means
are ~0, i.e. closing lines are unbiased -- which is exactly why they make a good
mean. No scipy dependency: Phi is built from math.erf.
"""

from __future__ import annotations

import math

# empirical closing-line residual SDs (2015-2025, n=3028)
MARGIN_SD = 12.7
TOTAL_SD = 13.2


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def prob_over(mean: float, strike: float, sd: float) -> float:
    """P(outcome > strike) for outcome ~ Normal(mean, sd)."""
    if sd <= 0:
        return 1.0 if mean > strike else 0.0
    return _phi((mean - strike) / sd)


def prob_margin_over(mean_margin: float, strike: float, sd: float = MARGIN_SD) -> float:
    """P(a team wins by more than `strike`), given its expected margin."""
    return prob_over(mean_margin, strike, sd)


def prob_total_over(mean_total: float, strike: float, sd: float = TOTAL_SD) -> float:
    """P(combined score exceeds `strike`), given the expected total."""
    return prob_over(mean_total, strike, sd)
