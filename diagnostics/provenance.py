"""Root-cause classification for unavailable model components.

For each component that reports available=False, resolve WHY from signals that
already exist (no model change, no re-fetch):
  * the analysis itself  — TBD starter (pitcher name is None), first-pitch time,
    lineup-status objects, the component's own available/note;
  * cascade rules        — recent form is PITCHING-ONLY (keyed on the named
    starter), so a TBD starter zeroes form on that side BY CASCADE;
  * lineup timing        — before the posting window => not-posted (legitimate),
    inside the window + feed error => fetch failure;
  * the fetch ledger     — slate-wide source outcomes (FanGraphs 403, Statcast
    timeout, etc.) captured output-neutrally during the run.

Reason codes: UPSTREAM_ENTITY_MISSING | FETCH_FAILED | STALE | NOT_APPLICABLE | OK.
The result is merged into reasoning_json as an additive `data_health` block plus
per-component reason fields — additive only; nothing else in the row changes.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from mlb_value_bot.data import fetch_ledger
from mlb_value_bot.utils import get_logger

log = get_logger("diagnostics.provenance")

UPSTREAM = "UPSTREAM_ENTITY_MISSING"
FETCH_FAILED = "FETCH_FAILED"
STALE = "STALE"
NOT_APPLICABLE = "NOT_APPLICABLE"
OK = "OK"

# reason_code -> UI tone. Legitimate absences read as expected (neutral); real
# problems read as danger (red).
_TONE = {UPSTREAM: "expected", NOT_APPLICABLE: "expected", OK: "ok",
         FETCH_FAILED: "danger", STALE: "danger"}

_FETCH_OUTCOMES = {"http_error", "rate_limited", "timeout", "error"}
_SEVERITY = {"http_error": 5, "rate_limited": 5, "timeout": 5, "error": 4,
             "stale": 3, "empty": 2, "ok": 1}


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _worst(substr: str) -> dict | None:
    """Most-severe ledger entry whose source/key matches substr (or None)."""
    matches = fetch_ledger.find(substr)
    if not matches:
        return None
    return max(matches, key=lambda e: (_SEVERITY.get(e.get("outcome"), 0), bool(e.get("stale"))))


def _prov(reason: str, label: str, *, sources=None, http_status=None,
          cascaded_from=None, detail=None) -> dict:
    return {
        "reason_code": reason,
        "reason_label": label,
        "reason_tone": _TONE.get(reason, "expected"),
        "sources_attempted": sources or [],
        "http_status": http_status,
        "cascaded_from": cascaded_from,
        "detail": detail,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _hours_until_first_pitch(game_datetime: str | None, now: datetime) -> float | None:
    if not game_datetime:
        return None
    try:
        fp = datetime.fromisoformat(str(game_datetime).replace("Z", "+00:00"))
        if fp.tzinfo is None:
            fp = fp.replace(tzinfo=timezone.utc)
        return (fp - now).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return None


def _lineup_side_reason(lu, hours_until: float | None, window: float) -> tuple[str, str]:
    """(reason_code, detail) for one side's lineup status."""
    if lu is None:
        return NOT_APPLICABLE, "lineup feature unavailable/disabled"
    status = getattr(lu, "status", None)
    if status == "confirmed":
        return OK, "confirmed"
    if status == "unavailable":
        return FETCH_FAILED, "lineup feed unavailable"
    # projected: not posted yet -- legitimate either way (too early, or posted-incomplete)
    if hours_until is not None and hours_until > window:
        return UPSTREAM, f"not posted yet ({hours_until:.1f}h before first pitch)"
    return UPSTREAM, "lineup not yet posted"


def classify_analysis(analysis, config: dict, now: datetime | None = None) -> tuple[dict, dict]:
    """Return (data_health, {component_name: provenance}). Pure read-only."""
    now = _now(now)
    window = float(config.get("lineup", {}).get("hours_before_first_pitch_to_check", 3.0))
    hours_until = _hours_until_first_pitch(getattr(analysis, "game_datetime", None), now)

    home_tbd = getattr(analysis, "home_pitcher", None) in (None, "", "TBD")
    away_tbd = getattr(analysis, "away_pitcher", None) in (None, "", "TBD")
    tbd_sides = [s for s, t in (("home", home_tbd), ("away", away_tbd)) if t]

    fg_pitch = _worst("fg_pitching")
    fg_bat = _worst("fg_team_batting")

    def fg_sources(*extra):
        srcs = []
        if fg_pitch:
            srcs.append({"source": "fangraphs_pitching", "outcome": fg_pitch.get("outcome"),
                         "http_status": fg_pitch.get("http_status")})
        srcs.extend(extra)
        return srcs

    components = getattr(getattr(analysis, "wp", None), "components", []) or []
    by_name = {c.name: c for c in components}
    starter_available = bool(getattr(by_name.get("starter"), "available", False))
    # A genuine Statcast/FanGraphs fetch FAILURE leaves a hard-error entry in the
    # ledger (exception-based). An "empty" return = the pitcher simply has no /
    # too little data (thin sample) -- legitimate, NOT a fetch failure.
    # FanGraphs 403 is the PERMANENT baseline here (Statcast is the intended
    # fallback), so only a Statcast hard error is a real starter/form fetch
    # failure -- not FanGraphs being blocked.
    statcast_hard = any(e.get("outcome") in _FETCH_OUTCOMES for e in fetch_ledger.find("statcast"))
    out: dict[str, dict] = {}

    for c in components:
        name, available, note = c.name, c.available, (c.note or "")
        if available:
            # Available but possibly STALE if its primary source was served stale.
            if name in ("starter", "bullpen", "park") and fg_pitch and fg_pitch.get("stale"):
                out[name] = _prov(STALE, "STALE DATA", sources=fg_sources(),
                                  http_status=fg_pitch.get("http_status"),
                                  detail="served from stale cache")
            else:
                out[name] = _prov(OK, "")
            continue

        if name == "starter":
            if tbd_sides:
                out[name] = _prov(UPSTREAM, "PITCHER NOT NAMED",
                                  detail=f"starter TBD ({'/'.join(tbd_sides)})")
            elif statcast_hard:
                # Named starters but no rate, AND Statcast (the working source)
                # actually errored.
                out[name] = _prov(FETCH_FAILED, "FETCH FAILED",
                                  sources=fg_sources({"source": "statcast", "outcome": "error"}),
                                  detail="named starters, no rate resolved, Statcast errored")
            else:
                # Named but no rate and no fetch error -> debut / unmatched id; the
                # data simply isn't there yet. Legitimate, not a failure.
                out[name] = _prov(NOT_APPLICABLE, "NO RATE DATA",
                                  detail="named starter but no rate (debut/unmatched); sources OK")
        elif name == "form":
            if tbd_sides:
                out[name] = _prov(UPSTREAM, "NO STARTER NAMED", cascaded_from="starter",
                                  detail=f"recent form is pitching-only; starter TBD ({'/'.join(tbd_sides)})")
            elif starter_available:
                # Starter rate DID resolve -> Statcast/FanGraphs worked; form is
                # missing only for lack of enough RECENT batted balls (call-up /
                # IL return / early season). Legitimate thin sample, not a failure.
                out[name] = _prov(NOT_APPLICABLE, "THIN RECENT DATA", cascaded_from="starter",
                                  detail="recent windows below sample floor; season rate available")
            elif statcast_hard:
                out[name] = _prov(FETCH_FAILED, "FETCH FAILED",
                                  sources=[{"source": "statcast", "outcome": "error"}],
                                  detail="recent form missing and Statcast errored")
            else:
                out[name] = _prov(NOT_APPLICABLE, "NO RECENT DATA",
                                  detail="no recent Statcast windows; sources OK")
        elif name == "lineup":
            hr, dh = _lineup_side_reason(getattr(analysis, "home_lineup_status", None), hours_until, window)
            ar, da = _lineup_side_reason(getattr(analysis, "away_lineup_status", None), hours_until, window)
            # Component reason = the more severe of the two sides.
            worst = max((hr, dh), (ar, da), key=lambda rd: _TONE_RANK(rd[0]))
            reason, detail = worst
            if reason == FETCH_FAILED:
                out[name] = _prov(FETCH_FAILED, "FETCH FAILED", detail=f"H:{dh}; A:{da}")
            elif reason == NOT_APPLICABLE:
                out[name] = _prov(NOT_APPLICABLE, "LINEUP N/A", detail=f"H:{dh}; A:{da}")
            else:
                out[name] = _prov(UPSTREAM, "LINEUP NOT POSTED YET", detail=f"H:{dh}; A:{da}")
        elif name == "bullpen_fatigue":
            # No status object on the analysis; classify from the ledger. Only a
            # genuine MLB feed error is FETCH_FAILED; otherwise it's a legitimate
            # absence (no recent games / insufficient sample).
            mlb_err = _worst("/v1/")
            if mlb_err and mlb_err.get("outcome") in _FETCH_OUTCOMES:
                out[name] = _prov(FETCH_FAILED, "FETCH FAILED",
                                  sources=[{"source": "mlb_stats_api", "outcome": mlb_err.get("outcome"),
                                            "http_status": mlb_err.get("http_status")}],
                                  http_status=mlb_err.get("http_status"), detail=note)
            else:
                out[name] = _prov(NOT_APPLICABLE, "NO BULLPEN DATA", detail=note)
        elif name in ("bullpen", "park"):
            # FanGraphs primary + MLB-API proxy both empty -> genuine fetch gap.
            http = fg_pitch.get("http_status") if (name == "bullpen" and fg_pitch) else (
                fg_bat.get("http_status") if (name == "park" and fg_bat) else None)
            primary = fg_pitch if name == "bullpen" else fg_bat
            srcs = []
            if primary:
                srcs.append({"source": "fangraphs", "outcome": primary.get("outcome"),
                             "http_status": primary.get("http_status")})
            srcs.append({"source": "mlb_stats_proxy", "outcome": "empty/failed"})
            out[name] = _prov(FETCH_FAILED, "FETCH FAILED", sources=srcs, http_status=http, detail=note)
        else:
            out[name] = _prov(NOT_APPLICABLE, "N/A", detail=note)

    # Game-level summary.
    expected = sum(1 for p in out.values() if p["reason_code"] in (UPSTREAM, NOT_APPLICABLE))
    failed = sum(1 for p in out.values() if p["reason_code"] == FETCH_FAILED)
    stale = sum(1 for p in out.values() if p["reason_code"] == STALE)
    data_health = {
        "expected_absences": expected,
        "fetch_failures": failed,
        "stale": stale,
        "ok": sum(1 for p in out.values() if p["reason_code"] == OK),
        "computed_at": now.isoformat(timespec="seconds"),
    }
    return data_health, out


def _TONE_RANK(reason: str) -> int:
    return {FETCH_FAILED: 3, STALE: 3, NOT_APPLICABLE: 1, UPSTREAM: 2, OK: 0}.get(reason, 0)


# --- persistence: merge into reasoning_json (additive) -----------------------
def _merge_into_reasoning(reasoning: dict, data_health: dict, per_comp: dict) -> dict:
    reasoning = dict(reasoning or {})
    reasoning["data_health"] = data_health
    comps = reasoning.get("components")
    if isinstance(comps, list):
        for c in comps:
            prov = per_comp.get(c.get("name"))
            if prov:
                c["reason_code"] = prov["reason_code"]
                c["reason_label"] = prov["reason_label"]
                c["reason_tone"] = prov["reason_tone"]
                if prov.get("cascaded_from"):
                    c["cascaded_from"] = prov["cascaded_from"]
                if prov.get("sources_attempted"):
                    c["sources_attempted"] = prov["sources_attempted"]
                if prov.get("http_status") is not None:
                    c["fetch_http_status"] = prov["http_status"]
    return reasoning


def annotate_slate(analyses, db_path, table: str, game_date: str,
                   config: dict, now: datetime | None = None) -> dict:
    """Classify each analysis and merge the provenance block into its stored
    reasoning_json row. Additive: only adds data_health + per-component reason
    fields; never touches model values, EV, or picks. Returns a summary."""
    summary = {"games": 0, "fetch_failures": 0, "stale": 0, "details": []}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for a in analyses:
            if getattr(a, "wp", None) is None:
                continue
            data_health, per_comp = classify_analysis(a, config, now)
            row = conn.execute(
                f"SELECT reasoning_json FROM {table} WHERE date=? AND game_id=?",
                (game_date, a.game_id),
            ).fetchone()
            if row is None:
                continue
            try:
                reasoning = json.loads(row["reasoning_json"] or "{}")
            except json.JSONDecodeError:
                reasoning = {}
            merged = _merge_into_reasoning(reasoning, data_health, per_comp)
            conn.execute(
                f"UPDATE {table} SET reasoning_json=? WHERE date=? AND game_id=?",
                (json.dumps(merged), game_date, a.game_id),
            )
            summary["games"] += 1
            summary["fetch_failures"] += data_health["fetch_failures"]
            summary["stale"] += data_health["stale"]
            if data_health["fetch_failures"] or data_health["stale"]:
                bad = {n: p["reason_code"] for n, p in per_comp.items()
                       if p["reason_code"] in (FETCH_FAILED, STALE)}
                summary["details"].append(
                    {"game": f"{a.away_team} @ {a.home_team}", "problems": bad})
        conn.commit()
    finally:
        conn.close()
    return summary
