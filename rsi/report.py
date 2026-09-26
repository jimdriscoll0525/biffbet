"""Review summary + the weekly plain-text email.

build_summary() reduces a ReviewResult (or anything duck-typed like it) to a
JSON-able dict that is stored on rsi_review_runs.summary and rendered by
render_email(). The email carries counts, one line per pending decision,
one line per shadow, rollback warnings and the single site link -- no
action links, decisions are made on the site.
"""
from __future__ import annotations

from typing import Any

_CLV_UNIT = {"clv_pct": "%", "clv_pp": "pp", "clv_blended_vs_sharp": "pp"}
RUN_CMD = "python -m mlb_value_bot.rsi review --sport all"


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def clv_text(avg_clv, clv_metric, n=None) -> str:
    if avg_clv is None:
        return "CLV n/a"
    unit = _CLV_UNIT.get(clv_metric or "", "")
    s = f"CLV {avg_clv:+.2f}{unit}"
    return f"{s} (n={n})" if n is not None else s


def headline(stats: dict | None) -> str:
    """'n=101, CLV +0.80% (n=95), ROI +4.9%, hit 57%, 2026-05-01 to 2026-09-20'."""
    if not stats:
        return "no stats"
    parts = [f"n={stats.get('settled', stats.get('n', 0))}",
             clv_text(stats.get("avg_clv"), stats.get("clv_metric"), stats.get("clv_tracked"))]
    roi = stats.get("flat_roi")
    parts.append(f"ROI {roi * 100:+.1f}%" if roi is not None else "ROI n/a")
    hit = stats.get("hit_rate")
    parts.append(f"hit {hit * 100:.0f}%" if hit is not None else "hit n/a")
    if stats.get("date_from") and stats.get("date_to"):
        parts.append(f"{stats['date_from']} to {stats['date_to']}")
    return ", ".join(parts)


def build_summary(result: Any) -> dict:
    sports_out: list[dict] = []
    pending: list[dict] = []
    for sp in _get(result, "sports", []) or []:
        changes = _get(sp, "changes", []) or []
        new_watches = sum(1 for c in changes if _get(c, "is_new"))
        newly_pending = sum(1 for c in changes
                            if _get(c, "status_after") == "pending" and _get(c, "status_before") != "pending")
        dropped = sum(1 for c in changes if _get(c, "status_after") == "dropped"
                      and _get(c, "status_before") != "dropped")
        pools = {}
        for pool, base in (_get(sp, "pools", {}) or {}).items():
            pools[pool] = {k: base.get(k) for k in ("rows", "settled", "wins", "losses", "flat_roi",
                                                   "avg_clv", "clv_metric", "clv_tracked")}
        sports_out.append({
            "sport": _get(sp, "sport"), "engine": _get(sp, "engine"),
            "rows": int(_get(sp, "rows", 0) or 0), "settled": int(_get(sp, "settled", 0) or 0),
            "cells_tested": int(_get(sp, "cells_tested", 0) or 0),
            "findings": len(_get(sp, "findings", []) or []),
            "new_watches": new_watches, "newly_pending": newly_pending, "dropped": dropped,
            "pools": pools, "error": _get(sp, "error"),
        })
        for p in _get(sp, "proposals_after", []) or []:
            if p.get("status") == "pending":
                pending.append({
                    "id": p.get("id"), "engine": p.get("engine"), "sport": p.get("sport"),
                    "finding_key": p.get("finding_key"), "title": p.get("title"),
                    "kind": p.get("kind"), "confidence": p.get("confidence"),
                    "headline": headline(p.get("latest_stats")),
                })
    shadows = []
    for s in _get(result, "shadows", []) or []:
        shadows.append({
            "proposal_id": s.get("proposal_id"), "title": s.get("title"), "engine": s.get("engine"),
            "challenger_tag": s.get("challenger_tag"), "champion": s.get("champion"),
            "challenger": s.get("challenger"), "gate": s.get("gate"),
            "days_in_shadow": s.get("days_in_shadow"), "window_complete": s.get("window_complete"),
        })
    rollbacks = [{"tag": v.get("tag"), "engine": v.get("engine"), "rolling": v.get("rolling"),
                  "rollback_reason": v.get("rollback_reason")}
                 for v in (_get(result, "rollbacks", []) or []) if v.get("rollback_flagged")]
    return {
        "run_date": _get(result, "run_date"), "dry_run": bool(_get(result, "dry_run", False)),
        "sports": sports_out, "pending": pending, "shadows": shadows, "rollbacks": rollbacks,
        "errors": list(_get(result, "errors", []) or []),
    }


def _side_line(label: str, s: dict | None) -> str:
    s = s or {}
    roi = s.get("flat_roi")
    return (f"{label} {clv_text(s.get('avg_clv'), s.get('clv_metric'), s.get('n_settled'))}, "
            f"ROI {roi * 100:+.1f}%" if roi is not None else
            f"{label} {clv_text(s.get('avg_clv'), s.get('clv_metric'), s.get('n_settled'))}, ROI n/a")


def render_email(summary: dict, site_url: str) -> tuple[str, str]:
    run_date = summary.get("run_date") or ""
    pending = summary.get("pending") or []
    rollbacks = summary.get("rollbacks") or []
    subject = f"BiffBet RSI review {run_date}: {len(pending)} pending decision(s)"
    if rollbacks:
        subject += f", {len(rollbacks)} rollback warning(s)"
    if summary.get("dry_run"):
        subject = "[dry run] " + subject

    lines: list[str] = [f"BiffBet RSI weekly review - {run_date}", ""]
    for sp in summary.get("sports") or []:
        if sp.get("error"):
            lines.append(f"{sp.get('sport')}: ERROR {sp['error']}")
            continue
        lines.append(f"{sp.get('sport')}: {sp.get('rows', 0)} rows analysed, {sp.get('settled', 0)} settled, "
                     f"{sp.get('cells_tested', 0)} cells tested, {sp.get('new_watches', 0)} new watch(es), "
                     f"{sp.get('newly_pending', 0)} newly pending, {sp.get('dropped', 0)} dropped")
    lines.append("")
    lines.append(f"Pending decisions ({len(pending)})")
    if not pending:
        lines.append("  none - nothing cleared the bar this week, which is a fine outcome.")
    for p in pending:
        lines.append(f"  - [{p.get('engine')}] {p.get('title')} ({p.get('confidence')} confidence): {p.get('headline')}")
    lines.append("")
    shadows = summary.get("shadows") or []
    lines.append(f"In shadow ({len(shadows)})")
    if not shadows:
        lines.append("  none")
    for s in shadows:
        g = s.get("gate") or {}
        status = "GATE OPEN" if g.get("enabled") else "gate closed: " + "; ".join(g.get("reasons") or [])
        lines.append(f"  - [{s.get('engine')}] {s.get('title')} ({s.get('challenger_tag')}): "
                     f"{_side_line('champion', s.get('champion'))} vs "
                     f"{_side_line('challenger', s.get('challenger'))} - {status}")
    lines.append("")
    lines.append(f"Rollback warnings ({len(rollbacks)})")
    if not rollbacks:
        lines.append("  none")
    for r in rollbacks:
        lines.append(f"  - [{r.get('engine')}] {r.get('tag')}: {r.get('rollback_reason')}")
    if summary.get("errors"):
        lines.append("")
        lines.append("Errors")
        for e in summary["errors"]:
            lines.append(f"  - {e}")
    lines += ["", f"Decide on the site: {site_url}", "",
              f"Run manually: {RUN_CMD}"]
    return subject, "\n".join(lines)
