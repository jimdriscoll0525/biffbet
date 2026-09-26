"""Weekly RSI email via Resend. Plain text, no action links.

send() never raises: a missing RESEND_API_KEY or an HTTP failure is logged
and reported as False so the review command always completes.
"""
from __future__ import annotations

import requests

from mlb_value_bot.utils import get_env, get_logger

log = get_logger("rsi.email")

_ENDPOINT = "https://api.resend.com/emails"
_TIMEOUT = 20


def send(subject: str, text: str, cfg: dict) -> bool:
    key = get_env("RESEND_API_KEY")
    if not key:
        log.warning("RESEND_API_KEY not set; RSI email not sent (subject: %s)", subject)
        return False
    email_cfg = (cfg or {}).get("email") or {}
    to = email_cfg.get("to")
    sender = email_cfg.get("from") or "BiffBet RSI <onboarding@resend.dev>"
    if not to:
        log.warning("rsi email.to not configured; email not sent")
        return False
    payload = {"from": sender, "to": [to] if isinstance(to, str) else list(to),
               "subject": subject, "text": text}
    try:
        resp = requests.post(_ENDPOINT, headers={"Authorization": "Bearer " + key},
                             json=payload, timeout=_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - email must never break the run
        log.warning("RSI email failed: %s", exc)
        return False
    if resp.status_code >= 300:
        log.warning("RSI email rejected (%s): %s", resp.status_code, resp.text[:300])
        return False
    log.info("RSI email sent to %s", to)
    return True
