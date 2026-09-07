"""Kalshi trading fees.

Kalshi charges a per-contract fee that peaks at a 50c price and vanishes near the
extremes, using the form:

    fee = ceil( rate * C * P * (1 - P) )   [in dollars, rounded up to the cent]

where P is the contract price in dollars (0-1) and C the number of contracts.
The general rate is 0.07; some high-volume sports markets use 0.035. Because the
fee is real money out of every +EV calc, we model it explicitly and keep the
rate configurable rather than hand-waving it.

These are estimates for EV screening -- always reconcile against Kalshi's own fee
schedule before sizing real trades.
"""

from __future__ import annotations

import math

DEFAULT_RATE = 0.07


def fee_per_contract(price: float, rate: float = DEFAULT_RATE) -> float:
    """Fee in dollars for ONE contract at `price` (dollars, 0-1)."""
    if price is None or not (0.0 < price < 1.0):
        return 0.0
    raw = rate * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0     # round up to the next cent


def total_cost(price: float, rate: float = DEFAULT_RATE) -> float:
    """All-in cost to enter one YES contract: price + fee (dollars)."""
    return price + fee_per_contract(price, rate)
