"""One-time import of the old docs/abilities ledger into rsi_proposals.

Idempotent on (engine, finding_key): a key already present in rsi_proposals
is skipped, so re-running never duplicates rows or 'imported' events.

Status mapping (ledger -> rsi_proposals):
    active -> promoted (version_id = the engine's seed version)
    watch -> watch, rejected -> rejected, dropped -> dropped,
    retire-recommended -> pending, proposed -> pending, retired -> rejected
config_keys become the overlay, valued from the CURRENT yaml (the ledger's
abilities are already live in config), and the proposal doc's markdown
becomes the description.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlb_value_bot.rsi.overlay import flatten
from mlb_value_bot.utils import get_logger

log = get_logger("rsi.import_ledger")

STATUS_MAP = {
    "active": "promoted", "watch": "watch", "rejected": "rejected", "dropped": "dropped",
    "retire-recommended": "pending", "proposed": "pending", "retired": "rejected",
}
SEED_TAG = {"mlb": "biff_v1", "mlb_totals": "totals_v1", "football": "matchup_v1"}


def engine_for_key(key: str) -> str:
    prefix = key.split("|", 1)[0]
    if prefix.startswith("totals"):
        return "mlb_totals"
    if prefix.startswith("fb"):
        return "football"
    if prefix.startswith("griff"):
        return "griffbet"
    return "mlb"


def sport_for(engine: str, key: str) -> str:
    if engine == "football":
        low = key.lower()
        if "cfb" in low:
            return "cfb"
        if "nfl" in low:
            return "nfl"
        return "football"
    return engine


def _overlay_from_keys(config_keys: list[str] | None, flat_cfg: dict) -> dict | None:
    if not config_keys:
        return None
    overlay = {}
    for k in config_keys:
        overlay[k] = flat_cfg.get(k)
    return overlay


def _direction(entry: dict) -> str:
    d = entry.get("direction")
    return d if d in ("positive", "negative") else "structural"


def _stats_from_evidence(evidence: list[dict]) -> dict:
    for e in reversed(evidence or []):
        if isinstance(e, dict) and e.get("settled") is not None:
            return {
                "n": e.get("rows"), "settled": e.get("settled"), "wins": e.get("wins"),
                "losses": e.get("losses"), "hit_rate": e.get("hit_rate"),
                "flat_roi": e.get("flat_roi"), "avg_clv": e.get("avg_clv_pct", e.get("avg_clv")),
                "clv_metric": "clv_pct", "clv_tracked": e.get("clv_tracked"),
                "t_stat": e.get("t_stat"), "p_value": e.get("p_value"),
                "bonferroni": bool(e.get("bonferroni_significant")),
                "clv_agrees": e.get("clv_agrees"), "date_from": None, "date_to": e.get("run_date"),
            }
    return {"n": None, "settled": None, "imported": True}


def _read_doc(proposals_dir: Path, rel: str | None) -> str | None:
    if not rel:
        return None
    name = Path(rel).name
    path = proposals_dir / name
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def build_rows(ledger: dict, proposals_dir: Path, configs: dict[str, dict],
               version_ids: dict[str, int | None]) -> list[dict]:
    """Pure: the rsi_proposals rows the ledger maps to."""
    flat = {engine: flatten(cfg) for engine, cfg in configs.items()}
    rows: list[dict] = []
    for a in ledger.get("abilities") or []:
        key = a["key"]
        engine = engine_for_key(key)
        status = STATUS_MAP.get(a.get("status"), "watch")
        overlay = _overlay_from_keys(a.get("config_keys"), flat.get(engine, {}))
        doc = _read_doc(proposals_dir, a.get("proposal_doc"))
        notes = a.get("notes") or ""
        title = key
        if doc:
            first = next((ln.strip("# ").strip() for ln in doc.splitlines() if ln.startswith("#")), None)
            title = first or key
        elif notes:
            title = notes.split(". ")[0].rstrip(".")[:120]
        description = doc or notes or key
        evidence = list(a.get("evidence") or [])
        dates = [e.get("run_date") for e in evidence if isinstance(e, dict) and e.get("run_date")]
        first_seen = a.get("first_seen") or (min(dates) if dates else ledger.get("updated"))
        last_seen = a.get("last_seen") or (max(dates) if dates else first_seen)
        row = {
            "engine": engine, "sport": sport_for(engine, key), "finding_key": key,
            "kind": "overlay" if overlay else "insight", "title": title,
            "description": description, "suggested_change": (
                "config keys: " + ", ".join(a["config_keys"]) if a.get("config_keys") else None),
            "overlay": overlay, "status": status,
            "confidence": "high" if status == "promoted" else "low",
            "direction": _direction(a), "latest_stats": _stats_from_evidence(evidence),
            "evidence": evidence + [{"run_date": ledger.get("updated"), "imported_from": "docs/abilities/ledger.json",
                                     "counts_toward_persistence": False,
                                     "kill_criteria": a.get("kill_criteria")}],
            "consecutive_hits": int(a.get("consecutive_hits") or 0),
            "consecutive_misses": int(a.get("consecutive_misses") or 0),
            "first_seen": first_seen, "last_seen": last_seen,
            "decided_at": None, "decision_reason": None, "version_id": None,
        }
        decided = a.get("activated") or a.get("rejected") or a.get("dropped") or a.get("proposed")
        if decided and status in ("promoted", "rejected", "dropped", "pending"):
            row["decided_at"] = f"{decided}T00:00:00Z"
            row["decision_reason"] = f"imported from ledger (status {a.get('status')})"
        if status == "promoted":
            row["version_id"] = version_ids.get(SEED_TAG.get(engine, ""))
        rows.append(row)
    return rows


def _load_configs() -> dict[str, dict]:
    from mlb_value_bot.football import load_football_config
    from mlb_value_bot.utils import load_config

    base = load_config()
    return {"mlb": base, "mlb_totals": base, "football": load_football_config(), "griffbet": {}}


def import_ledger(ledger_path: str | Path, proposals_dir: str | Path | None = None,
                  dry_run: bool = False) -> dict:
    """Import the ledger; returns {imported, skipped, rows}."""
    from mlb_value_bot.rsi import supa

    ledger_path = Path(ledger_path)
    proposals_dir = Path(proposals_dir) if proposals_dir else ledger_path.parent / "proposals"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    version_ids: dict[str, int | None] = {}
    existing_keys: set[tuple[str, str]] = set()
    if not dry_run:
        for v in supa.get_rows("rsi_model_versions", select="id,tag"):
            version_ids[v["tag"]] = v["id"]
        existing_keys = {(p["engine"], p["finding_key"])
                         for p in supa.get_rows("rsi_proposals", select="engine,finding_key")}
    rows = build_rows(ledger, proposals_dir, _load_configs(), version_ids)
    to_insert = [r for r in rows if (r["engine"], r["finding_key"]) not in existing_keys]
    skipped = len(rows) - len(to_insert)
    if dry_run:
        return {"imported": len(to_insert), "skipped": skipped, "rows": to_insert, "dry_run": True}
    imported = 0
    for r in to_insert:
        created = supa.insert_returning("rsi_proposals", r)
        pid = created.get("id")
        if pid is not None:
            supa.insert_rows("rsi_proposal_events", [{
                "proposal_id": pid, "event": "imported", "actor": "engine",
                "payload": {"source": str(ledger_path), "ledger_status": None,
                            "imported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}}])
        imported += 1
    log.info("Imported %d ledger item(s), skipped %d already present.", imported, skipped)
    return {"imported": imported, "skipped": skipped, "rows": to_insert}


def as_json(value: Any) -> str:
    return json.dumps(value, indent=1, default=str)
