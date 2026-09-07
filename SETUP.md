# Setup — live Kalshi edge alerts to Telegram

The scanner reads Kalshi with **no key** (public data). You only need two things:
a free **Odds API key** (book fair-value reference) and a **Telegram bot** (alerts).

## 1. Odds API key (fair-value reference)

1. Sign up at <https://the-odds-api.com> (free tier: 500 requests/month).
2. Copy your API key from the dashboard.

## 2. Telegram bot (the alert channel)

1. In Telegram, message **@BotFather** → send `/newbot` → follow prompts →
   copy the **bot token** it gives you (looks like `12345678:AA...`).
2. Send any message to your new bot (so it can see you).
3. Get your **chat ID**: message **@userinfobot**, or open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and read
   `"chat":{"id": ...}`.

## 3. Test it locally first (Windows PowerShell)

```powershell
$env:ODDS_API_KEY      = "your_odds_key"
$env:TELEGRAM_BOT_TOKEN = "your_bot_token"
$env:TELEGRAM_CHAT_ID   = "your_chat_id"

# one scan, print the board:
python scripts/scan_kalshi.py

# scan + push a Telegram alert on any new edge >= 3%:
python scripts/scan_kalshi.py --notify telegram

# poll every 15 minutes while your PC is on:
python scripts/scan_kalshi.py --loop 900 --notify telegram
```

No key yet? See the format with fake data: `python scripts/scan_kalshi.py --demo`.

## 4. Always-on in the cloud (GitHub Actions)

Once the repo is on GitHub (VS Code → Publish to GitHub → **Private**):

1. Repo **Settings → Secrets and variables → Actions → New repository secret**.
   Add three secrets (names must match exactly):
   - `ODDS_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
2. Go to the **Actions** tab, enable workflows if prompted.
3. Open **kalshi-edge-scan** → **Run workflow** to test it immediately.
4. After that it runs on its own every 2 hours, Thu–Mon (see `.github/workflows/scan.yml`).

**Secrets never live in the code** — they're read from environment variables, and
GitHub encrypts the repo secrets. Nothing sensitive is committed.

## Quota math

The Odds API free tier is 500 requests/month; each scan uses 1. The default
schedule (~every 2h, Thu–Mon) spends ~250/month. To poll Sundays harder, add a
second `cron` line in the workflow, or upgrade the Odds API plan. Kalshi is free
and unlimited, so the Kalshi price side can be polled as often as you like.

## Tuning

- `--min-edge 0.03` → only alert on ≥3% EV. Raise it to cut noise.
- `--fee-rate 0.07` → Kalshi fee rate; check against Kalshi's live fee schedule.
- Dedup: you're only pinged on a **new** edge or one that grew ≥2 points
  (`notify.filter_new`), so a standing edge won't spam you.
