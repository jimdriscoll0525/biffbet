"""Email alert for the BiffBet weekly retro.

Reads the newest storage/retro/retro_report_*.json and docs/abilities/
ledger.json and emails Jim when there is something worth a human look:

  * a candidate cleared the evidence bar this run (new or repeat trend), or
  * the ledger holds anything in status "proposed" or "retire-recommended"
    (i.e. a decision is waiting on Jim).

Otherwise it exits quietly (set RETRO_EMAIL_ALWAYS=1 in .env to get a
"ran, nothing found" heartbeat instead, so silence never means "broken").

Like retro_analysis.py this is a deliberately import-free sidecar: stdlib
only, read-only against every file it touches, and it never imports
mlb_value_bot modules. SMTP credentials come from .env:

    RETRO_SMTP_HOST=smtp.gmail.com
    RETRO_SMTP_PORT=587
    RETRO_SMTP_USER=you@example.com
    RETRO_SMTP_PASSWORD=app-password-here
    RETRO_EMAIL_TO=jim@financewithjim.com      (optional; this is the default)
    RETRO_EMAIL_FROM=you@example.com           (optional; defaults to USER)
    RETRO_EMAIL_ALWAYS=1                       (optional heartbeat mode)

Usage (from the repo root):
    python .claude/skills/self-improve/scripts/notify_trends.py
    python ... notify_trends.py --dry-run       # print instead of send
"""
from __future__ import annotations

import argparse
import json
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RETRO_DIR = REPO_ROOT / "storage" / "retro"
LEDGER = REPO_ROOT / "docs" / "abilities" / "ledger.json"
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_TO = "jim@financewithjim.com"

# Ledger statuses that mean "a decision is waiting on Jim".
ACTIONABLE_STATUSES = {"proposed", "retire-recommended"}


def _load_env(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser for .env (no quoting rules, no expansion)."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _newest_report() -> tuple[Path, dict] | tuple[None, None]:
    reports = sorted(RETRO_DIR.glob("retro_report_*.json"))
    if not reports:
        return None, None
    path = reports[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_ledger() -> dict:
    if not LEDGER.exists():
        return {"abilities": []}
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _fmt_candidate(c: dict) -> str:
    clv = {True: "CLV agrees", False: "CLV DISAGREES", None: "no CLV (pass pool)"}[
        c.get("clv_agrees")]
    bonf = "Bonferroni-significant" if c.get("bonferroni_significant") \
        else "NOT Bonferroni-significant"
    return (
        f"* {c['key']} — {c['direction']}, {c['wins']}-{c['losses']} "
        f"({c['hit_rate']:.0%} vs {c['breakeven_hit_rate']:.0%} breakeven), "
        f"flat ROI {c['flat_roi']:+.1%}, t={c['t_stat']}, p={c['p_value']}\n"
        f"    {bonf} · {clv}"
    )


def build_email(report_path: Path, report: dict, ledger: dict) -> tuple[str, str, bool]:
    """Returns (subject, body, actionable)."""
    candidates = report.get("candidates", [])
    waiting = [a for a in ledger.get("abilities", [])
               if a.get("status") in ACTIONABLE_STATUSES]
    watches = [a for a in ledger.get("abilities", [])
               if a.get("status") == "watch"]

    lines: list[str] = []
    lines.append(f"BiffBet weekly retro — {report.get('run_date')}")
    lines.append(f"Report: {report_path}")
    rc = report.get("row_counts", {})
    lines.append(
        f"Rows: {rc.get('moneyline_bets', 0)} ML bets / "
        f"{rc.get('moneyline_passes', 0)} ML passes / "
        f"{rc.get('totals_bets', 0)} totals bets / "
        f"{rc.get('totals_passes', 0)} totals passes · "
        f"{report.get('cells_tested')} cells tested")
    lines.append("")

    if waiting:
        lines.append(f"AWAITING YOUR APPROVAL ({len(waiting)}):")
        for a in waiting:
            doc = a.get("proposal_doc", "(no doc)")
            lines.append(f"* [{a['status']}] {a['key']} — {doc}")
        lines.append("")

    if candidates:
        lines.append(f"Trends clearing the evidence bar this run ({len(candidates)}):")
        for c in candidates:
            lines.append(_fmt_candidate(c))
        lines.append("")
    else:
        lines.append("No candidates cleared the evidence bar this run.")
        lines.append("")

    if watches:
        lines.append("Ledger watch list:")
        for a in watches:
            lines.append(
                f"* {a['key']} — {a.get('consecutive_hits', 0)} consecutive hit(s), "
                f"first seen {a.get('first_seen')}")
        lines.append("")

    lines.append("Remember: promotion needs 2+ consecutive weekly hits; nothing")
    lines.append("changes in the engine without your explicit approval.")
    lines.append("Run /self-improve in Claude Code to review or act on any of this.")

    actionable = bool(candidates or waiting)
    if waiting:
        subject = f"[BiffBet retro] {len(waiting)} proposal(s) awaiting approval"
    elif candidates:
        subject = f"[BiffBet retro] {len(candidates)} trend(s) cleared the bar"
    else:
        subject = "[BiffBet retro] ran clean — no trends this week"
    return subject, "\n".join(lines), actionable


def send(env: dict[str, str], subject: str, body: str, dry_run: bool) -> int:
    to_addr = env.get("RETRO_EMAIL_TO", DEFAULT_TO)
    if dry_run:
        print(f"--- DRY RUN (would send to {to_addr}) ---")
        print(f"Subject: {subject}")
        print()
        print(body)
        return 0

    host = env.get("RETRO_SMTP_HOST")
    user = env.get("RETRO_SMTP_USER")
    password = env.get("RETRO_SMTP_PASSWORD")
    if not (host and user and password):
        print("notify_trends: RETRO_SMTP_HOST/USER/PASSWORD not set in .env — "
              "email skipped. Add them to enable alerts.", file=sys.stderr)
        return 1

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = env.get("RETRO_EMAIL_FROM", user)
    msg["To"] = to_addr
    msg.set_content(body)

    port = int(env.get("RETRO_SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    print(f"notify_trends: sent '{subject}' to {to_addr}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the email instead of sending it.")
    args = ap.parse_args(argv)

    report_path, report = _newest_report()
    if report is None:
        print("notify_trends: no retro report found — run retro_analysis.py "
              "first.", file=sys.stderr)
        return 2
    ledger = _load_ledger()

    subject, body, actionable = build_email(report_path, report, ledger)
    env = _load_env(ENV_FILE)

    if not actionable and env.get("RETRO_EMAIL_ALWAYS") != "1":
        print("notify_trends: nothing actionable and RETRO_EMAIL_ALWAYS != 1 — "
              "no email.")
        return 0
    return send(env, subject, body, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
