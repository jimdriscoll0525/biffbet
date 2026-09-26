"""Reconcile this run's findings against the existing rsi_proposals rows.

Pure: takes findings + existing rows + the run date, returns ProposalChange
objects (the upsert row, the events to append, the before/after status).
The review module does the I/O.

Lifecycle (from the old SKILL.md, now mechanical):
  * new finding                        -> watch, hits=1, first_seen=run_date
  * watch, present, settled grew       -> hits+1 (a stale re-run never counts)
  * hits >= consecutive_runs_for_pending AND (bonferroni OR clv_agrees)
                                       -> pending (+ bar_met, pending events)
  * present but bar not met            -> stays watch
  * absent this run (watch / pending)  -> misses+1, hits=0; misses >= 2 -> dropped
  * rejected / dropped / promoted / approved reappearing
                                       -> evidence appended only
  * snoozed with snoozed_until <= run_date, present -> pending (unsnoozed)
  * calendar-only dims (month)         -> never leave watch
Confidence: high = bonferroni AND clv_agrees; medium = clv_agrees AND hits >= 2;
low otherwise. Overlap: same pool + direction, nested price dims (fav_size
within odds_bucket within side_type) -> keep the sharpest cell, fold the
broader ones into evidence[-1]["corroborating_cells"].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mlb_value_bot.rsi.segments import CALENDAR_DIMS, NESTED_PRICE_DIMS, price_range
from mlb_value_bot.rsi.stats import Candidate

ENGINE_STATUSES = {"watch", "pending", "snoozed"}          # the engine may move these
DECIDED_STATUSES = {"approved", "promoted", "rejected", "dropped"}
_ROW_SKIP = {"id", "created_at"}


@dataclass
class ProposalChange:
    engine: str
    finding_key: str
    row: dict                         # upsert payload (no id)
    events: list[dict]                # [{"event": str, "payload": dict}]
    is_new: bool
    status_before: str | None
    status_after: str
    proposal_id: int | None = None
    note: str = ""
    finding: dict | None = field(default=None, repr=False)


def _finding_dict(f: Candidate | dict) -> dict:
    return f.to_dict() | {"suggestion": f.suggestion, "corroborating": f.corroborating} \
        if isinstance(f, Candidate) else dict(f)


def _nan_none(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def latest_stats(f: dict) -> dict:
    return {
        "n": f.get("rows"), "settled": f.get("settled"), "wins": f.get("wins"),
        "losses": f.get("losses"), "hit_rate": f.get("hit_rate"), "flat_roi": f.get("flat_roi"),
        "avg_clv": f.get("avg_clv"), "clv_metric": f.get("clv_metric"),
        "clv_tracked": f.get("clv_tracked"), "t_stat": f.get("t_stat"),
        "p_value": f.get("p_value"), "bonferroni": bool(f.get("bonferroni_significant")),
        "clv_agrees": f.get("clv_agrees"), "date_from": f.get("date_from"),
        "date_to": f.get("date_to"), "delta_clv": _nan_none(f.get("delta_clv")),
        "delta_roi": _nan_none(f.get("delta_roi")),
    }


def evidence_entry(f: dict, run_date: str) -> dict:
    keep = ("rows", "settled", "wins", "losses", "pushes", "hit_rate", "hit_rate_wilson_95",
            "breakeven_hit_rate", "flat_pl_units", "flat_roi", "t_stat", "p_value",
            "avg_ev_pct", "avg_clv", "clv_metric", "clv_tracked", "clv_positive",
            "date_from", "date_to", "direction", "delta_clv", "delta_roi", "preregistered")
    entry = {"run_date": run_date}
    entry.update({k: _nan_none(f.get(k)) for k in keep if k in f})
    entry["bonferroni_significant"] = bool(f.get("bonferroni_significant"))
    entry["clv_agrees"] = f.get("clv_agrees")
    entry["cells_tested"] = f.get("cells_tested")
    if f.get("corroborating"):
        entry["corroborating_cells"] = [
            {"key": c.get("key"), "settled": c.get("settled"), "flat_roi": c.get("flat_roi"),
             "avg_clv": c.get("avg_clv"), "t_stat": c.get("t_stat")}
            for c in f["corroborating"]]
    return entry


def confidence_for(f: dict, hits: int) -> str:
    if f.get("bonferroni_significant") and f.get("clv_agrees"):
        return "high"
    if f.get("clv_agrees") and hits >= 2:
        return "medium"
    return "low"


def fold_overlaps(findings: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Return (kept findings, {folded_key: kept_key}). Nested price cells in the
    same pool+direction collapse onto the narrowest one."""
    folded: dict[str, str] = {}
    by_group: dict[tuple, list[dict]] = {}
    for f in findings:
        if f.get("dimension") in NESTED_PRICE_DIMS and price_range(f["dimension"], f["value"]):
            by_group.setdefault((f.get("engine"), f["pool"], f["direction"]), []).append(f)
    for group in by_group.values():
        ranged = [(price_range(f["dimension"], f["value"]), f) for f in group]
        for (ra, a) in ranged:
            for (rb, b) in ranged:
                if a is b or a["key"] in folded:
                    continue
                inside = rb[0] <= ra[0] and ra[1] <= rb[1] and (rb != ra)
                if inside:
                    # a is narrower than b -> b folds into a
                    if b["key"] not in folded:
                        folded[b["key"]] = a["key"]
                        a.setdefault("corroborating", []).append(b)
    kept = [f for f in findings if f["key"] not in folded]
    # A folded key may itself have collected corroborators; move them up.
    for f in kept:
        extra: list[dict] = []
        for c in f.get("corroborating") or []:
            extra.extend(c.get("corroborating") or [])
        if extra:
            f["corroborating"] = list(f.get("corroborating") or []) + extra
    return kept, folded


def _base_row(existing: dict | None) -> dict:
    if not existing:
        return {}
    return {k: v for k, v in existing.items() if k not in _ROW_SKIP}


def _last_settled(existing: dict | None) -> int | None:
    for entry in reversed((existing or {}).get("evidence") or []):
        if isinstance(entry, dict) and entry.get("settled") is not None:
            return int(entry["settled"])
    return None


def reconcile(findings: list[Candidate | dict], existing: list[dict], run_date: str,
              cfg: dict) -> list[ProposalChange]:
    runs_for_pending = int(cfg.get("consecutive_runs_for_pending", 2))
    found = [_finding_dict(f) for f in findings]
    found, folded = fold_overlaps(found)
    by_key: dict[tuple[str, str], dict] = {(e["engine"], e["finding_key"]): e for e in existing}
    present: set[tuple[str, str]] = set()
    changes: list[ProposalChange] = []

    for f in found:
        key = (f.get("engine", ""), f["key"])
        present.add(key)
        ex = by_key.get(key)
        entry = evidence_entry(f, run_date)
        sug = f.get("suggestion") or {}
        events: list[dict] = [{"event": "seen", "payload": {"run_date": run_date,
                                                              "settled": f.get("settled")}}]
        if ex is None:
            hits = 1
            status = "watch"
            row = {
                "engine": f.get("engine"), "sport": f.get("sport") or f.get("engine"),
                "finding_key": f["key"], "kind": sug.get("kind", "insight"),
                "title": sug.get("title") or f["key"],
                "description": sug.get("description") or f["key"],
                "suggested_change": sug.get("suggested_change"), "overlay": sug.get("overlay"),
                "status": status, "confidence": confidence_for(f, hits),
                "direction": f.get("direction"), "latest_stats": latest_stats(f),
                "evidence": [entry], "consecutive_hits": hits, "consecutive_misses": 0,
                "first_seen": run_date, "last_seen": run_date,
            }
            changes.append(ProposalChange(f.get("engine", ""), f["key"], row, events, True,
                                          None, status, finding=f))
            continue

        row = _base_row(ex)
        before = ex.get("status") or "watch"
        status = before
        hits = int(ex.get("consecutive_hits") or 0)
        misses = 0
        evidence = list(ex.get("evidence") or []) + [entry]
        grown = (_last_settled(ex) is None) or (int(f.get("settled") or 0) > _last_settled(ex))
        note = ""
        if before in ("watch", "pending"):
            if grown:
                hits += 1
            else:
                note = "settled count did not grow; hit not counted"
            bar = bool(f.get("bonferroni_significant") or f.get("clv_agrees"))
            calendar = f.get("dimension") in CALENDAR_DIMS
            if before == "watch" and hits >= runs_for_pending and bar and not calendar:
                status = "pending"
                events.append({"event": "bar_met", "payload": {"run_date": run_date, "hits": hits,
                                                                "bonferroni": bool(f.get("bonferroni_significant")),
                                                                "clv_agrees": f.get("clv_agrees")}})
                events.append({"event": "pending", "payload": {"run_date": run_date}})
            elif before == "watch" and calendar:
                note = "calendar-only dimension never leaves watch"
            row.update({"title": sug.get("title") or ex.get("title"),
                        "description": sug.get("description") or ex.get("description"),
                        "suggested_change": sug.get("suggested_change", ex.get("suggested_change")),
                        "overlay": sug.get("overlay", ex.get("overlay")),
                        "kind": sug.get("kind", ex.get("kind") or "insight"),
                        "confidence": confidence_for(f, hits)})
        elif before == "snoozed":
            if grown:
                hits += 1
            until = ex.get("snoozed_until")
            if until and str(until) <= run_date:
                status = "pending"
                events.append({"event": "unsnoozed", "payload": {"run_date": run_date}})
                row["snoozed_until"] = None
            row["confidence"] = confidence_for(f, hits)
        else:
            # approved / promoted / rejected / dropped: evidence only.
            misses = int(ex.get("consecutive_misses") or 0)
            note = f"{before}: evidence appended only"
        row.update({"status": status, "latest_stats": latest_stats(f), "evidence": evidence,
                    "consecutive_hits": hits, "consecutive_misses": misses,
                    "last_seen": run_date, "direction": f.get("direction") or ex.get("direction")})
        changes.append(ProposalChange(ex["engine"], ex["finding_key"], row, events, False,
                                      before, status, proposal_id=ex.get("id"), note=note,
                                      finding=f))

    # Folded findings whose proposal already exists: evidence only, no hit/miss.
    for f_key, kept_key in folded.items():
        for (engine, fk), ex in by_key.items():
            if fk != f_key or (engine, fk) in present:
                continue
            present.add((engine, fk))
            row = _base_row(ex)
            entry = {"run_date": run_date, "folded_into": kept_key, "counts_toward_persistence": False}
            row.update({"evidence": list(ex.get("evidence") or []) + [entry], "last_seen": run_date})
            changes.append(ProposalChange(engine, fk, row,
                                          [{"event": "seen", "payload": {"run_date": run_date,
                                                                          "folded_into": kept_key}}],
                                          False, ex.get("status"), ex.get("status") or "watch",
                                          proposal_id=ex.get("id"), note="folded"))

    # Absent this run.
    for key, ex in by_key.items():
        if key in present or ex.get("status") not in ENGINE_STATUSES:
            continue
        if ex.get("last_seen") and str(ex["last_seen"]) >= run_date:
            continue                       # already reconciled for this run date
        row = _base_row(ex)
        misses = int(ex.get("consecutive_misses") or 0) + 1
        before = ex.get("status")
        status = before
        events: list[dict] = []
        if misses >= 2 and before in ("watch", "pending"):
            status = "dropped"
            events.append({"event": "dropped", "payload": {"run_date": run_date, "misses": misses}})
        row.update({"status": status, "consecutive_hits": 0, "consecutive_misses": misses,
                    "evidence": list(ex.get("evidence") or []) + [
                        {"run_date": run_date, "absent": True, "counts_toward_persistence": False}]})
        changes.append(ProposalChange(key[0], key[1], row, events, False, before, status,
                                      proposal_id=ex.get("id"), note="absent"))
    return changes


def apply_changes(existing: list[dict], changes: list[ProposalChange]) -> list[dict]:
    """The proposal rows as they will look after `changes` (for dry runs and
    the summary)."""
    merged: dict[tuple[str, str], dict] = {(e["engine"], e["finding_key"]): dict(e) for e in existing}
    for ch in changes:
        key = (ch.engine, ch.finding_key)
        base: dict[str, Any] = merged.get(key, {})
        base = dict(base)
        base.update(ch.row)
        if ch.proposal_id is not None:
            base["id"] = ch.proposal_id
        merged[key] = base
    return list(merged.values())
