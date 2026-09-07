"""Edge engine -- Phase 2/3.

Finds +EV bets on Kalshi (the only NFL execution venue available in Utah) by
comparing Kalshi contract prices to de-vigged book consensus fair value.

Books here are a *reference price*, not a place to bet: strip their vig to learn
the true probability, then buy it cheaper on Kalshi when the exchange disagrees.
"""
