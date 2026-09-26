"""Rollback monitor for ACTIVE model versions.

rolling_check() looks at the last rolling_window[engine] settled champion
picks stamped with the version's tag and flags the version when its avg CLV
falls below rollback.clv_floor or below the promotion baseline by more than
rollback.clv_drop_vs_baseline. It patches `rolling` / `rollback_flagged` /
`rollback_reason` and NEVER touches `status` -- rolling back is Jim's button.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from mlb_value_bot.rsi.segments import SETTLED
from mlb_value_bot.utils import get_logger

log = get_logger("rsi.versions")


def _r(x, nd: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, nd)


def evaluate_rolling(rows: pd.DataFrame, version: dict, cfg: dict) -> dict:
    """Pure: {rolling, rollback_flagged, rollback_reason} for `version` from the
    champion pick rows (view shape, any order; the last `window` settled by
    date are used)."""
    engine = version.get("engine", "mlb")
    window = int((cfg.get("rolling_window") or {}).get(engine, 75))
    rb = cfg.get("rollback") or {}
    floor = float(rb.get("clv_floor", 0.0))
    drop = float(rb.get("clv_drop_vs_baseline", 0.5))

    df = rows.copy() if rows is not None else pd.DataFrame()
    if not df.empty:
        df = df[df["result"].fillna("pending").astype(str).isin(SETTLED)]
        df = df.sort_values("date").tail(window)
    clv = pd.to_numeric(df["clv"], errors="coerce").dropna() if not df.empty else pd.Series(dtype=float)
    flat = pd.to_numeric(df["flat_pl_units"], errors="coerce").dropna() if not df.empty else pd.Series(dtype=float)
    clv_metric = None
    if not df.empty and "clv_metric" in df.columns and df["clv_metric"].notna().any():
        clv_metric = str(df["clv_metric"].dropna().iloc[0])
    rolling = {
        "n": int(len(df)), "window": window, "avg_clv": _r(clv.mean()) if len(clv) else None,
        "clv_metric": clv_metric or ("clv_pct" if engine == "mlb" else "clv_pp"),
        "clv_tracked": int(len(clv)), "flat_roi": _r(flat.mean()) if len(flat) else None,
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    flagged, reason = False, None
    min_n = int((cfg.get("min_sample") or {}).get(engine, 75))
    if rolling["avg_clv"] is not None and rolling["clv_tracked"] >= min(min_n, window):
        if rolling["avg_clv"] < floor:
            flagged, reason = True, (f"rolling avg CLV {rolling['avg_clv']:+.3f} below floor {floor:+.2f} "
                                     f"over last {rolling['n']} settled picks")
        baseline = version.get("baseline") or {}
        base_clv = baseline.get("avg_clv") if isinstance(baseline, dict) else None
        if base_clv is not None and rolling["avg_clv"] < float(base_clv) - drop:
            flagged = True
            reason = (reason + "; " if reason else "") + (
                f"rolling avg CLV {rolling['avg_clv']:+.3f} is more than {drop} below the "
                f"promotion baseline {float(base_clv):+.3f}")
    return {"rolling": rolling, "rollback_flagged": flagged, "rollback_reason": reason}


def rolling_check(version: dict, cfg: dict, rows: pd.DataFrame | None = None,
                  write: bool = True) -> dict:
    """Evaluate one version (fetching its champion picks unless `rows` is given)
    and patch rolling / rollback_flagged / rollback_reason. Never changes status."""
    if rows is None:
        from mlb_value_bot.rsi import supa

        rows = supa.fetch_scored_games(
            engine=version["engine"],
            extra_filters={"decision": "pick", "model_version": version["tag"],
                           "result": ("in", sorted(SETTLED))})
    result = evaluate_rolling(rows, version, cfg)
    if write:
        from mlb_value_bot.rsi import supa

        supa.patch_rows("rsi_model_versions", {"id": version["id"]}, dict(result))
    return {"id": version.get("id"), "engine": version.get("engine"), "tag": version.get("tag"),
            "status": version.get("status"), **result}


def rolling_check_all(cfg: dict, write: bool = True) -> list[dict]:
    from mlb_value_bot.rsi import supa

    out: list[dict] = []
    for v in supa.get_rows("rsi_model_versions", {"status": "active"}, order="id.asc"):
        try:
            out.append(rolling_check(v, cfg, write=write))
        except Exception as exc:  # noqa: BLE001
            log.warning("rolling check failed for %s: %s", v.get("tag"), exc)
    return out
