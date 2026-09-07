# nfl-edge

An NFL betting model and market-edge toolkit. The goal is not "beat Vegas" — NFL
sides and totals are among the most efficient markets in the world. The goal is a
**disciplined, honest pipeline**: pull real data, engineer leakage-free features,
grade every prediction against the **closing line** in units and ROI, and only
claim an edge that survives that test.

Built Polars-native (no pandas/pyarrow) so it runs anywhere without native-DLL
headaches.

## Phase 1 status: ✅ foundation complete

| Piece | File | State |
|-------|------|-------|
| Data ingestion (nflreadpy → parquet) | `src/nfl_edge/ingest.py` | done |
| Odds math (implied prob, de-vig, grading, CLV) | `src/nfl_edge/odds.py` | done, 14 unit tests |
| Backtest harness (walk-forward, units/ROI) | `src/nfl_edge/backtest.py` | done |
| Baseline validation | `outputs/backtest_reports/` | done |

## Phase 2 status: 🟡 in progress

| Piece | File | State |
|-------|------|-------|
| Point-in-time EPA features (leakage-checked) | `src/nfl_edge/features.py` | done |
| Walk-forward ridge power rating | `src/nfl_edge/model.py` | done |
| Model → ATS backtest vs closing line | `scripts/run_model.py` | done |
| Live Kalshi edge scanner + Discord alerts | `src/nfl_edge/edge/`, `.github/workflows/scan.yml` | done |
| Calibration report (Brier, reliability) | `evaluate.py` | next |

## Live edge scanner (Utah-legal: Kalshi only)

Since Utah bans sportsbooks, books are used as a **fair-value reference**, not a
place to bet: de-vig the book consensus to a true win probability, then buy it
cheaper on **Kalshi** (a CFTC exchange) when the price disagrees.

```bash
python scripts/scan_kalshi.py                 # print the edge board
python scripts/scan_kalshi.py --notify telegram --loop 900   # poll + phone alerts
python scripts/scan_kalshi.py --demo          # see the format without any API key
```

Covers all three NFL markets:

| market | Kalshi series | fair value from |
|--------|---------------|-----------------|
| moneyline | `KXNFLGAME` | de-vigged book h2h consensus |
| spread | `KXNFLSPREAD` | P(margin > strike), normal around consensus spread |
| total | `KXNFLTOTAL` | P(total > strike), normal around consensus total |

- Kalshi reads need **no credentials** (public market data).
- Spreads/totals are priced with a **distribution model** whose SDs are estimated
  empirically from 2015–2025 closing-line residuals (margin σ≈12.7, total σ≈13.2;
  both residual means ≈0, i.e. closing lines are unbiased). See `edge/dist.py`.
- Edge = `fair − (kalshi_ask + fee)`, with a fractional-**Kelly** stake per bet.
- Always-on via **GitHub Actions cron** → **Telegram**. See [SETUP.md](SETUP.md).

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows
pip install -r requirements.txt

python scripts/build_dataset.py --pbp      # cache schedules + play-by-play
python -m pytest -q                        # 14/14 green
python scripts/run_backtest.py             # baseline units/ROI, 2015–2025
python scripts/run_model.py                # walk-forward model vs closing line
```

## Data (audited, not assumed)

Source: [`nflreadpy`](https://pypi.org/project/nflreadpy/) — the maintained
successor to the **deprecated** `nfl_data_py` (archived Sept 2025). Returns Polars.

`load_schedules()` gives one row per game since 1999 with `spread_line`,
`total_line`, moneylines, side odds, `result`, `roof/temp/wind`, and rest days.

Coverage audit (`load_schedules`, 1999–2025):

| Field | Coverage |
|-------|----------|
| spread_line, total_line | 100% since **1999** |
| moneylines, side odds | complete since **2010** (patchy 2006–09) |
| results | 100% (2025 complete through the Feb 2026 Super Bowl) |

Verified semantics (asserted at ingest time in `_sanity_schedules`):
- `spread_line > 0` ⇒ **home** favored by that many points
- `result` = home margin = `home_score − away_score`
- `total` = `home_score + away_score`

## The two rules that keep it honest

1. **Walk-forward, never a random split.** All state (team strength, etc.) is
   updated only *after* a game is graded, so every pick uses pre-kickoff info.
2. **Grade in units/ROI against the closing line, never raw accuracy.** You need
   only **52.4%** ATS to beat −110 juice; win rate alone is noise.

## CLV is a live-ops metric, not a backtest output

`load_schedules` provides **one line per game — the close**. There is no opening
line or per-book entry price, so there is no entry-vs-close delta to compute
historically. **Closing Line Value is measured live** (Phase 3): log the line you
actually took during the week, then compare to the number that closed. The
`clv_*` helpers in `odds.py` exist for that live loop. In the backtest, beating
the closing line *is* the hard test — and it's the honest one.

## Baseline backtest — the leakage check (2015–2025, REG, flat 1u)

Before any model, we grade dumb strategies. A correct harness makes them all
lose ~the vig. If a dumb baseline showed real profit, that's a bug, not alpha.

| strategy | market | bets | win% | ROI |
|----------|--------|-----:|-----:|----:|
| always_away_ats | ats | 2895 | 51.2% | −0.6% |
| always_under | total | 2895 | 51.1% | −0.8% |
| home_underdog_ats | ats | 1109 | 50.2% | −2.9% |
| better_ppg_diff_ats | ats | 2356 | 48.7% | −5.0% |
| always_home_ats | ats | 2895 | 48.8% | −5.2% |
| always_over | total | 2895 | 48.9% | −5.3% |

Reading it: nothing clears break-even (no leakage). The *pattern* is real —
fading the public (away/under) loses least, chasing home/over loses most — and
**betting the statistically "better" team ATS loses**, because team strength is
already priced into the spread. That's the whole motivation for a model that
beats the *number*.

## Phase 2 model — the honest result (2016–2025, out-of-sample)

A walk-forward ridge builds a team power rating from EPA differentials **only**
(the market line is excluded), then bets ATS when its projected margin disagrees
with the closing spread by more than a threshold.

Prediction quality vs. the market:

| metric | model | closing spread |
|--------|------:|---------------:|
| MAE vs actual margin | 10.33 | **9.86** |
| corr with actual margin | 0.36 | — |

ATS results by edge threshold (flat 1u vs the close, break-even = 52.4%):

| threshold | bets | win% | ROI |
|----------:|-----:|-----:|----:|
| 0.5 | 2153 | 50.4% | −2.5% |
| 1.0 | 1867 | 50.6% | −2.2% |
| 2.0 | 1359 | 51.7% | −0.2% |
| 3.0 |  947 | 51.7% | −0.3% |

**Reading it:** the closing spread predicts margins *better* than the EPA model
(9.86 < 10.33 MAE), and the model tops out at 51.7% ATS — real signal (>50%, and
it sharpens as the threshold rises) but short of the 52.4% needed to beat the
vig. This is the expected, correct result: **a "predict the game better" model
does not beat an efficient closing line.** It is the launch point for the edges
that don't require out-predicting the market — softer markets and book-vs-Kalshi
price gaps.

## Roadmap

- **Phase 2 — Edge engine + Kalshi.** No-vig fair-value calculator (already in
  `odds.py`), Kalshi-vs-book discrepancy scanner (Kalshi = CFTC exchange, no house
  vig — compare contract implied prob vs. book de-vigged prob), key-number logic,
  offseason roster-turnover features from rosters/snaps.
- **Phase 3 — Weekly ops.** Live line pull, prediction log, weekly grading,
  **CLV tracking**, calibration (Brier, reliability curves), Streamlit/Tableau
  dashboard.
- **Phase 4 — Expansion.** Playoff/Super Bowl futures, then player props (softer,
  less efficient markets).

Model work itself (`features.py` point-in-time EPA, `model.py` walk-forward
logistic/ridge, `evaluate.py` calibration) is Phase 2+ and hangs off this
foundation.

## Bankroll note

Even a real edge carries brutal variance over an 18-week season (~272 games).
Flat/fractional staking only, and treat every backtested "edge" as guilty of
overfitting until it survives out-of-sample and live grading.
