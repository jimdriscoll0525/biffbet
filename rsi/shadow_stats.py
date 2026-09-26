"""Champion vs challenger comparison for approved proposals running in shadow.

Champion rows = rsi_scored_games picks stamped with the engine's ACTIVE tag
since the shadow started; challenger rows = rsi_shadow_picks for the
proposal's challenger_tag (is_value true). Both sides get the same metric
bundle, an overlap breakdown, and a holdout slice (rsi.holdout twin of the
view's is_holdout). gate() then says whether the Promote button may open;
refresh_all() patches every approved proposal's shadow_stats.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone

import pandas as pd

from mlb_value_bot.rsi.holdout import is_holdout, sport_key_for
from mlb_value_bot.rsi.segments import SETTLED
from mlb_value_bot.utils import get_logger

log = get_logger("rsi.shadow")

_MARKET_FOR_ENGINE = {"mlb": "moneyline", "mlb_totals": "total", "griffbet": "moneyline"}


def _r(x, nd: int = 4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, nd)


def side_stats(df: pd.DataFrame, clv_col: str, flat_col: str, clv_metric: str) -> dict:
    """n_picks, n_settled, avg_clv, clv_positive_share, flat_roi, hit_rate, W-L."""
    if df is None or df.empty:
        return {"n_picks": 0, "n_settled": 0, "wins": 0, "losses": 0, "avg_clv": None,
                "clv_metric": clv_metric, "clv_tracked": 0, "clv_positive_share": None,
                "flat_roi": None, "hit_rate": None}
    results = df["result"].fillna("pending").astype(str)
    settled = df[results.isin(SETTLED)]
    wins = int((results.loc[settled.index] == "win").sum())
    n = len(settled)
    clv = pd.to_numeric(df[clv_col], errors="coerce").dropna() if clv_col in df.columns else pd.Series(dtype=float)
    flat = pd.to_numeric(settled[flat_col], errors="coerce").dropna() if flat_col in settled.columns else pd.Series(dtype=float)
    return {
        "n_picks": int(len(df)), "n_settled": n, "wins": wins, "losses": n - wins,
        "avg_clv": _r(clv.mean()) if len(clv) else None, "clv_metric": clv_metric,
        "clv_tracked": int(len(clv)),
        "clv_positive_share": _r((clv > 0).mean()) if len(clv) else None,
        "flat_roi": _r(flat.mean()) if len(flat) else None,
        "hit_rate": _r(wins / n) if n else None,
    }


def _game_keys(df: pd.DataFrame, key_col: str) -> set[tuple]:
    if df is None or df.empty:
        return set()
    return {(str(d), str(g), str(m)) for d, g, m in zip(df["date"], df[key_col], df["market"])}


def compare_frames(proposal: dict, champion: pd.DataFrame, challenger: pd.DataFrame,
                   cfg: dict, today: date | None = None) -> dict:
    """Pure comparison of two already-fetched frames."""
    today = today or date.today()
    engine = proposal.get("engine")
    clv_metric = "clv_pct" if engine in ("mlb", "griffbet") else "clv_pp"
    if champion is not None and not champion.empty and "clv_metric" in champion.columns:
        m = champion["clv_metric"].dropna()
        if len(m):
            clv_metric = str(m.iloc[0])
    champ = champion.copy() if champion is not None else pd.DataFrame()
    chal = challenger.copy() if challenger is not None else pd.DataFrame()
    holdout_pct = int(cfg.get("holdout_pct", 20))

    if not champ.empty and "is_holdout" not in champ.columns:
        champ["is_holdout"] = [is_holdout(sport_key_for(engine, s, l), g, holdout_pct)
                               for s, l, g in zip(champ.get("sport"), champ.get("league"), champ["game_key"])]
    if not chal.empty:
        chal["is_holdout"] = [is_holdout(sport_key_for(engine, s, l), g, holdout_pct)
                              for s, l, g in zip(chal.get("sport"), chal.get("league"), chal["game_id"])]

    champ_keys = _game_keys(champ, "game_key")
    chal_keys = _game_keys(chal, "game_id")
    overlap = champ_keys & chal_keys

    def _slice(df: pd.DataFrame, hold: bool) -> pd.DataFrame:
        if df.empty:
            return df
        return df[df["is_holdout"].astype(bool) == hold]

    started = proposal.get("shadow_started_at")
    ends = proposal.get("shadow_ends_at")
    days = None
    if started:
        days = (today - date.fromisoformat(str(started)[:10])).days
    window_complete = bool(ends) and today >= date.fromisoformat(str(ends)[:10])
    return {
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "champion_tag": proposal.get("_champion_tag"),
        "challenger_tag": proposal.get("challenger_tag"),
        "champion": side_stats(champ, "clv", "flat_pl_units", clv_metric),
        "challenger": side_stats(chal, "clv", "flat_pl", clv_metric),
        "overlap": len(overlap),
        "only_champion": len(champ_keys - chal_keys),
        "only_challenger": len(chal_keys - champ_keys),
        "holdout": {
            "champion": side_stats(_slice(champ, True), "clv", "flat_pl_units", clv_metric),
            "challenger": side_stats(_slice(chal, True), "clv", "flat_pl", clv_metric),
        },
        "days_in_shadow": days,
        "window_complete": window_complete,
        "shadow_started_at": started, "shadow_ends_at": ends,
    }


def gate(stats: dict, cfg: dict, engine: str | None = None) -> dict:
    """{"enabled": bool, "reasons": [...]} -- every failing condition listed."""
    reasons: list[str] = []
    engine = engine or stats.get("engine") or "mlb"
    min_n = int((cfg.get("min_sample") or {}).get(engine, 75))
    hold_min = int(cfg.get("holdout_min_n", 10))
    champ, chal = stats.get("champion") or {}, stats.get("challenger") or {}
    if int(chal.get("n_settled") or 0) < min_n:
        reasons.append(f"challenger has {chal.get('n_settled') or 0} settled picks; needs {min_n}")
    c_clv, h_clv = chal.get("avg_clv"), champ.get("avg_clv")
    if c_clv is None or h_clv is None:
        reasons.append("CLV not available on both sides")
    elif not c_clv > h_clv:
        reasons.append(f"challenger avg CLV {c_clv:+.3f} does not beat champion {h_clv:+.3f}")
    hold = stats.get("holdout") or {}
    hc, hh = hold.get("challenger") or {}, hold.get("champion") or {}
    if int(hc.get("n_settled") or 0) < hold_min or int(hh.get("n_settled") or 0) < hold_min:
        reasons.append("holdout sample too small")
    elif hc.get("avg_clv") is None or hh.get("avg_clv") is None:
        reasons.append("holdout CLV not available on both sides")
    elif hc["avg_clv"] < hh["avg_clv"]:
        reasons.append(f"holdout: challenger avg CLV {hc['avg_clv']:+.3f} below champion {hh['avg_clv']:+.3f}")
    if not stats.get("window_complete"):
        reasons.append("shadow window not complete")
    return {"enabled": not reasons, "reasons": reasons}


def active_tag(engine: str) -> str | None:
    from mlb_value_bot.rsi import supa

    rows = supa.get_rows("rsi_model_versions", {"engine": engine, "status": "active"}, limit=1)
    return rows[0]["tag"] if rows else None


def compare(proposal: dict, cfg: dict) -> dict:
    """Fetch both sides for one approved proposal and compare them."""
    from mlb_value_bot.rsi import supa

    engine = proposal["engine"]
    tag = active_tag(engine)
    since = str(proposal.get("shadow_started_at") or proposal.get("decided_at") or "")[:10] or None
    extra: dict = {"decision": "pick"}
    if tag:
        extra["model_version"] = tag
    if engine == "football" and proposal.get("sport") in ("nfl", "cfb"):
        extra["league"] = proposal["sport"]
    market = _MARKET_FOR_ENGINE.get(engine)
    if market:
        extra["market"] = market
    champ = supa.fetch_scored_games(engine=engine, since=since, extra_filters=extra)
    chal_rows = supa.get_rows("rsi_shadow_picks",
                              {"challenger_tag": proposal.get("challenger_tag"), "is_value": True},
                              order="date.asc")
    chal = pd.DataFrame(chal_rows)
    stats = compare_frames({**proposal, "_champion_tag": tag}, champ, chal, cfg)
    stats["engine"] = engine
    stats["gate"] = gate(stats, cfg, engine)
    return stats


def refresh_all(cfg: dict) -> list[dict]:
    """Recompute + patch shadow_stats for every approved proposal."""
    from mlb_value_bot.rsi import supa

    out: list[dict] = []
    approved = supa.get_rows("rsi_proposals", {"status": "approved"}, order="id.asc")
    for p in approved:
        try:
            stats = compare(p, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad proposal must not stop the rest
            log.warning("shadow stats failed for proposal %s: %s", p.get("id"), exc)
            continue
        supa.patch_rows("rsi_proposals", {"id": p["id"]},
                        {"shadow_stats": stats, "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})
        supa.insert_rows("rsi_proposal_events", [
            {"proposal_id": p["id"], "event": "shadow_stats", "actor": "engine",
             "payload": {"gate": stats["gate"], "champion": stats["champion"],
                         "challenger": stats["challenger"]}}])
        out.append({"proposal_id": p["id"], "title": p.get("title"), "engine": p.get("engine"),
                    "challenger_tag": p.get("challenger_tag"), **stats})
    return out
