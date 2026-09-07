"""Notification dispatch for live edge alerts.

Pluggable channels so you can pick whatever reaches your phone fastest:
    - console  (always; for local runs / debugging)
    - telegram (recommended: free, instant push to phone, no SMS cost)
    - discord  (webhook, also instant)
    - email    (SMTP; universal but slower)

Secrets come from environment variables, never the repo. A small JSON state file
dedups alerts so the same standing edge doesn't re-ping every poll -- you're only
notified on a NEW edge or a materially larger one.
"""

from __future__ import annotations

import json
import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import Protocol

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_FILE = REPO_ROOT / "outputs" / "notified.json"


class Notifier(Protocol):
    def send(self, subject: str, body: str) -> None: ...


class ConsoleNotifier:
    def send(self, subject: str, body: str) -> None:
        print(f"\n=== {subject} ===\n{body}\n")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token, self.chat_id = token, chat_id

    def send(self, subject: str, body: str) -> None:
        requests.post(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            json={"chat_id": self.chat_id, "text": f"*{subject}*\n{body}",
                  "parse_mode": "Markdown"},
            timeout=15,
        ).raise_for_status()


class DiscordNotifier:
    def __init__(self, webhook_url: str):
        self.url = webhook_url

    def send(self, subject: str, body: str) -> None:
        requests.post(self.url, json={"content": f"**{subject}**\n```\n{body}\n```"},
                      timeout=15).raise_for_status()


class EmailNotifier:
    def __init__(self, host, port, user, password, to_addr):
        self.host, self.port, self.user, self.password, self.to = (
            host, int(port), user, password, to_addr)

    def send(self, subject: str, body: str) -> None:
        msg = MIMEText(body)
        msg["Subject"], msg["From"], msg["To"] = subject, self.user, self.to
        with smtplib.SMTP(self.host, self.port) as srv:
            srv.starttls()
            srv.login(self.user, self.password)
            srv.send_message(msg)


def build_notifier(channel: str) -> Notifier:
    """Construct the configured notifier from env vars. `channel` in
    {console, telegram, discord, email}."""
    channel = (channel or "console").lower()
    if channel == "telegram":
        return TelegramNotifier(os.environ["TELEGRAM_BOT_TOKEN"],
                                os.environ["TELEGRAM_CHAT_ID"])
    if channel == "discord":
        return DiscordNotifier(os.environ["DISCORD_WEBHOOK_URL"])
    if channel == "email":
        return EmailNotifier(os.environ["SMTP_HOST"], os.environ.get("SMTP_PORT", 587),
                             os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"],
                             os.environ["ALERT_EMAIL_TO"])
    return ConsoleNotifier()


# --------------------------------------------------------------------------
# Dedup: only alert on new tickers, or when edge grows by >= bump (pts).
# --------------------------------------------------------------------------
def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def filter_new(rows: list, bump: float = 0.02) -> list:
    """Return only rows worth alerting: unseen ticker, or edge up by >= bump."""
    state = _load_state()
    fresh = []
    for r in rows:
        prev = state.get(r.ticker)
        if prev is None or (r.edge - prev) >= bump:
            fresh.append(r)
            state[r.ticker] = r.edge
    _save_state(state)
    return fresh
