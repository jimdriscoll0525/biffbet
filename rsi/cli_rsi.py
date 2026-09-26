"""RSI CLI -- `python -m mlb_value_bot.rsi <command>`.

  review [--sport all|mlb|mlb_totals|football|griffbet] [--since] [--dry-run] [--no-email]
  email [--run-id N]          re-send the summary email for a stored review run
  shadow-stats                refresh champion-vs-challenger stats for approved proposals
  versions                    rolling rollback check on active model versions
  import-ledger PATH          one-time import of docs/abilities/ledger.json

Heavy imports are deferred so `--help` stays fast and tests never touch the
network.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click

from mlb_value_bot.rsi import load_rsi_config
from mlb_value_bot.utils import get_logger

log = get_logger("rsi.cli")

_SPORTS = ("all", "mlb", "mlb_totals", "football", "griffbet")


@click.group()
def cli() -> None:
    """RSI - the engine's weekly self-improvement loop."""


def _send_summary(summary: dict, cfg: dict, run_id: int | None) -> bool:
    from mlb_value_bot.rsi import email as _email
    from mlb_value_bot.rsi.report import render_email

    subject, text = render_email(summary, cfg.get("site_url", "https://biffbet.com/proposals"))
    try:
        ok = _email.send(subject, text, cfg)
    except Exception as exc:  # noqa: BLE001 - email never fails the command
        log.warning("email failed: %s", exc)
        ok = False
    if ok and run_id is not None:
        try:
            from mlb_value_bot.rsi import supa
            supa.patch_rows("rsi_review_runs", {"id": run_id},
                            {"email_sent_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
        except Exception as exc:  # noqa: BLE001
            log.warning("could not stamp email_sent_at: %s", exc)
    return ok


@cli.command()
@click.option("--sport", type=click.Choice(_SPORTS), default="all", show_default=True)
@click.option("--since", type=str, default=None, help="Only rows with date >= YYYY-MM-DD.")
@click.option("--dry-run", is_flag=True, help="Compute and print; write nothing.")
@click.option("--no-email", is_flag=True, help="Skip the summary email.")
@click.option("--run-date", type=str, default=None, help="Override the run date (YYYY-MM-DD).")
def review(sport: str, since: str | None, dry_run: bool, no_email: bool, run_date: str | None) -> None:
    """Weekly review: mine the history, reconcile proposals, email the summary."""
    from mlb_value_bot.rsi.report import render_email
    from mlb_value_bot.rsi.review import run_review

    cfg = load_rsi_config()
    result = run_review([sport], since=since, dry_run=dry_run, run_date=run_date, cfg=cfg)
    subject, text = render_email(result.summary, cfg.get("site_url", ""))
    click.echo(text)
    if result.errors:
        click.echo("Errors: " + "; ".join(result.errors), err=True)
    if dry_run or no_email:
        return
    email_cfg = cfg.get("email") or {}
    if not email_cfg.get("always", True) and not result.summary.get("pending"):
        click.echo("Nothing pending and email.always is false; email skipped.")
        return
    ok = _send_summary(result.summary, cfg, result.run_id)
    click.echo("Email sent." if ok else "Email NOT sent (see log).")


@cli.command(name="email")
@click.option("--run-id", type=int, default=None, help="rsi_review_runs.id (default: latest).")
def email_cmd(run_id: int | None) -> None:
    """Re-send the summary email for a stored review run."""
    from mlb_value_bot.rsi import supa

    cfg = load_rsi_config()
    if run_id is None:
        rows = supa.get_rows("rsi_review_runs", order="id.desc", limit=1)
    else:
        rows = supa.get_rows("rsi_review_runs", {"id": run_id}, limit=1)
    if not rows:
        raise click.ClickException("no review run found")
    run = rows[0]
    summary = run.get("summary") or {"run_date": run.get("run_date"), "sports": [], "pending": []}
    ok = _send_summary(summary, cfg, run.get("id"))
    click.echo(f"Run {run.get('id')} ({run.get('run_date')}): " + ("email sent." if ok else "email NOT sent."))


@cli.command(name="shadow-stats")
def shadow_stats_cmd() -> None:
    """Refresh champion-vs-challenger stats for every approved proposal."""
    from mlb_value_bot.rsi import shadow_stats

    cfg = load_rsi_config()
    out = shadow_stats.refresh_all(cfg)
    if not out:
        click.echo("No approved proposals in shadow.")
    for s in out:
        g = s.get("gate") or {}
        click.echo(f"[{s['engine']}] #{s['proposal_id']} {s.get('title')} ({s.get('challenger_tag')}): "
                   f"champion CLV {s['champion'].get('avg_clv')} vs challenger {s['challenger'].get('avg_clv')} "
                   f"- {'GATE OPEN' if g.get('enabled') else 'closed: ' + '; '.join(g.get('reasons') or [])}")


@cli.command()
@click.option("--no-write", is_flag=True, help="Compute only; do not patch rsi_model_versions.")
def versions(no_write: bool) -> None:
    """Rolling rollback check on the active model versions."""
    from mlb_value_bot.rsi import versions as _versions

    cfg = load_rsi_config()
    out = _versions.rolling_check_all(cfg, write=not no_write)
    if not out:
        click.echo("No active versions.")
    for v in out:
        r = v.get("rolling") or {}
        flag = f"FLAGGED: {v.get('rollback_reason')}" if v.get("rollback_flagged") else "ok"
        click.echo(f"[{v['engine']}] {v['tag']}: n={r.get('n')} avg CLV {r.get('avg_clv')} "
                   f"({r.get('clv_metric')}) flat ROI {r.get('flat_roi')} - {flag}")


@cli.command(name="import-ledger")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Show the rows that would be inserted.")
def import_ledger_cmd(path: Path, dry_run: bool) -> None:
    """One-time import of docs/abilities/ledger.json (+ proposals/*.md)."""
    from mlb_value_bot.rsi.import_ledger import import_ledger

    out = import_ledger(path, path.parent / "proposals", dry_run=dry_run)
    click.echo(f"imported={out['imported']} skipped={out['skipped']}"
               + (" (dry run)" if dry_run else ""))
    if dry_run:
        for r in out["rows"]:
            click.echo(f"  {r['engine']:10s} {r['status']:9s} {r['finding_key']}")


# --- runtime half (added by the runtime agent) -------------------------------
# `grade`  -- settle rsi_shadow_picks against results and re-price CLV.
# `state`  -- print the active version + approved challengers per engine.
# Do not implement here; rsi/state.py, rsi/shadow.py, rsi/shadow_grade.py own them.
