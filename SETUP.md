# Setup — live Kalshi edge alerts to Discord

The scanner reads Kalshi with **no key** (public data). You only need two things:
a free **Odds API key** (book fair-value reference) and a **Discord webhook** (alerts).

## 1. Odds API key (fair-value reference)

1. Sign up at <https://the-odds-api.com> (free tier: 500 requests/month).
2. Copy your API key from the dashboard.

## 2. Discord webhook (the alert channel)

1. In Discord, pick (or create) a server and a channel for alerts — e.g. `#nfl-edges`.
2. Hover the channel → gear icon (**Edit Channel**) → **Integrations** →
   **Webhooks** → **New Webhook**.
3. Name it (e.g. "Kalshi Scanner"), then **Copy Webhook URL**. That URL is the
   secret — it looks like `https://discord.com/api/webhooks/123.../abc...`.

## 3. Test it locally first (Windows PowerShell)

```powershell
$env:ODDS_API_KEY        = "your_odds_key"
$env:DISCORD_WEBHOOK_URL = "your_webhook_url"

# one scan, print the board:
python scripts/scan_kalshi.py

# scan + push a Discord alert on any new edge >= 3%:
python scripts/scan_kalshi.py --notify discord

# poll every 15 minutes while your PC is on:
python scripts/scan_kalshi.py --loop 900 --notify discord
```

No key yet? See the format with fake data: `python scripts/scan_kalshi.py --demo`.

## 4. Always-on in the cloud (GitHub Actions)

Once the repo is on GitHub (VS Code → Publish to GitHub → **Private**):

1. Repo **Settings → Secrets and variables → Actions → New repository secret**.
   Add two secrets (names must match exactly):
   - `ODDS_API_KEY`
   - `DISCORD_WEBHOOK_URL`
2. Go to the **Actions** tab, enable workflows if prompted.
3. Open **kalshi-edge-scan** → **Run workflow** to test it immediately.
4. After that it runs on its own every 4 hours, Thu–Mon (see `.github/workflows/scan.yml`).

**Secrets never live in the code** — they're read from environment variables, and
GitHub encrypts the repo secrets. Nothing sensitive is committed.

## Quota math

The Odds API free tier is 500 requests/month. Each scan fetches
`h2h,spreads,totals` = **3 credits**. The default schedule (~every 4h, Thu–Mon)
spends ~390/month. To poll harder, upgrade the Odds API plan, or run moneyline
only (`--markets h2h`, 1 credit) more frequently. Kalshi is free and unlimited,
so the Kalshi price side can be polled as often as you like.

## Tuning

- `--min-edge 0.03` → only alert on ≥3% EV. Raise it to cut noise.
- `--fee-rate 0.07` → Kalshi fee rate; check against Kalshi's live fee schedule.
- Dedup: you're only pinged on a **new** edge or one that grew ≥2 points
  (`notify.filter_new`), so a standing edge won't spam you.
