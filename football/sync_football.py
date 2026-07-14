"""Football <-> Supabase sync — the only football module that talks to
Supabase. Reuses supabase_sync's proven plumbing (_credentials/_post/_clean/
_get_all) by import; pushes football's OWN tables only:

  * football_recommendations    (upsert on_conflict=league,date,game_id,market)
  * football_snapshot           (upsert on_conflict=scope)
  * football_team_drive_stats   (upsert on_conflict=league,season,team) — the
    live totals model's priors (live_total_v1; read by the biffbet site)

Pull rebuilds the local SQLite store from Supabase (the ephemeral CI box's
source of truth), mirroring pull_totals_recommendations.
"""
from __future__ import annotations

import json
from datetime import date as _date
from datetime import datetime, timezone

from mlb_value_bot.football.tracking import football_performance, football_store
from mlb_value_bot.sync.supabase_sync import _clean, _credentials, _get_all, _post
from mlb_value_bot.utils import get_logger

log = get_logger("football.sync")

_BATCH = 200

_PUSH_COLS = [
    "league", "date", "week", "game_id", "market", "home_team", "away_team",
    "pick_side", "line", "bet_odds", "decimal_odds", "model_prob",
    "market_prob_devigged", "p_push", "ev_pct", "adjusted_ev_pct", "flat_stake",
    "confidence", "tier", "stability", "edge_score", "archetype",
    "projected_margin", "projected_total", "paper", "model_tag",
    "opening_line", "opening_price", "opening_devig_p_side",
    "closing_line", "closing_price", "sharp_close_line",
    "sharp_close_devig_p_side", "clv_pp", "result", "home_score", "away_score",
    "profit_loss", "is_value", "created_at", "updated_at",
]


def _rec_rows(since: str | None) -> list[dict]:
    df = football_store.to_dataframe(since=since)
    if df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        row = {c: _clean(r[c]) for c in _PUSH_COLS if c in df.columns}
        row["paper"] = bool(r["paper"])
        row["is_value"] = bool(r["is_value"])
        raw = r.get("reasoning_json")
        row["reasoning"] = json.loads(raw) if isinstance(raw, str) and raw else None
        rows.append(row)
    return rows


def push_recommendations(url: str, key: str, since: str | None = None) -> int:
    rows = _rec_rows(since)
    for start in range(0, len(rows), _BATCH):
        _post(url, key, "football_recommendations", rows[start:start + _BATCH],
              on_conflict="league,date,game_id,market")
    log.info("Pushed %d football recommendation row(s)", len(rows))
    return len(rows)


def push_snapshot(url: str, key: str, config: dict) -> int:
    scopes = football_performance.compute_snapshot(config)
    payload = [{"scope": scope, "payload": data, "updated_at": data.get("computed_at")}
               for scope, data in scopes.items()]
    if payload:
        _post(url, key, "football_snapshot", payload, on_conflict="scope")
    log.info("Pushed %d football snapshot scope(s)", len(payload))
    return len(payload)


def push_drive_stats(url: str, key: str, config: dict) -> int:
    """Upsert the live totals model's drive-stat priors (NFL for v1). Runs
    year-round: nflverse is free, so this is NOT behind the season_window
    odds-credit gate."""
    from mlb_value_bot.football import season_for_date
    from mlb_value_bot.football.pipeline_football import _infer_week, build_drive_stats

    today = _date.today().isoformat()
    season = season_for_date(today)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = 0
    for league in ("nfl",):
        week = _infer_week(config, league, season, today)
        stats = build_drive_stats(league, season, week, config)
        if stats.empty:
            log.info("No %s drive stats for season %d (feed gap?)", league, season)
            continue
        rows = []
        for team, r in stats.iterrows():
            row = {c: _clean(r[c]) for c in stats.columns}
            row.update({"league": league, "season": season, "team": str(team),
                        "games": int(r["games"]), "updated_at": now})
            rows.append(row)
        for start in range(0, len(rows), _BATCH):
            _post(url, key, "football_team_drive_stats", rows[start:start + _BATCH],
                  on_conflict="league,season,team")
        total += len(rows)
    log.info("Pushed %d drive-stat row(s)", total)
    return total


def push_all(config: dict, since: str | None = None) -> dict:
    url, key = _credentials()
    out = {
        "recommendations": push_recommendations(url, key, since),
        "snapshot_scopes": push_snapshot(url, key, config),
    }
    # Tolerant: a missing table (schema not applied yet) must not fail the
    # recommendations sync — same contract as the MLB totals push.
    try:
        out["drive_stats"] = push_drive_stats(url, key, config)
    except Exception as exc:  # noqa: BLE001
        log.warning("drive-stats sync skipped (%s) — apply supabase/schema.sql "
                    "football_team_drive_stats?", exc)
        out["drive_stats"] = 0
    return out


# --- live totals (live_total_v1) helpers --------------------------------------
# The live tables are WRITTEN by the biffbet site's server routes; the engine
# only reads them back for grading and patches the graded fields. Kept here so
# this module stays the single football<->Supabase seam.

def fetch_live_table(table: str) -> list[dict]:
    """Read an entire live-totals table (football_live_recommendations /
    football_live_snapshots)."""
    url, key = _credentials()
    return _get_all(url, key, table)


def patch_live_recommendation(rec_id: int, fields: dict) -> None:
    """Update one graded recommendation row by id."""
    import requests

    url, key = _credentials()
    resp = requests.patch(
        f"{url}/rest/v1/football_live_recommendations?id=eq.{int(rec_id)}",
        headers={"apikey": key, "Authorization": f"Bearer {key}",
                 "Content-Type": "application/json", "Prefer": "return=minimal"},
        data=json.dumps({k: _clean(v) for k, v in fields.items()}, default=str),
        timeout=30,
    )
    if resp.status_code >= 300:
        raise RuntimeError(
            f"Supabase live-rec patch failed ({resp.status_code}): {resp.text[:300]}")


def pull_recommendations() -> int:
    """Rebuild the local store from Supabase (upsert by the unique key)."""
    url, key = _credentials()
    remote = _get_all(url, key, "football_recommendations")
    if not remote:
        return 0
    football_store.init_db()
    n = 0
    with football_store.connect() as conn:
        for r in remote:
            reasoning = r.get("reasoning")
            conn.execute(
                """
                INSERT INTO football_recommendations (
                    league, date, week, game_id, market, home_team, away_team, pick_side,
                    line, bet_odds, decimal_odds, model_prob, market_prob_devigged, p_push,
                    ev_pct, adjusted_ev_pct, flat_stake, confidence, tier, stability,
                    edge_score, archetype, projected_margin, projected_total, paper,
                    model_tag, reasoning_json, opening_line, opening_price,
                    opening_devig_p_side, closing_line, closing_price, sharp_close_line,
                    sharp_close_devig_p_side, clv_pp, result, home_score, away_score,
                    profit_loss, is_value, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(league, date, game_id, market) DO UPDATE SET
                    result=excluded.result, profit_loss=excluded.profit_loss,
                    home_score=excluded.home_score, away_score=excluded.away_score,
                    closing_line=excluded.closing_line, closing_price=excluded.closing_price,
                    sharp_close_line=excluded.sharp_close_line,
                    sharp_close_devig_p_side=excluded.sharp_close_devig_p_side,
                    clv_pp=excluded.clv_pp, is_value=excluded.is_value,
                    updated_at=excluded.updated_at
                """,
                (r.get("league"), r.get("date"), r.get("week"), r.get("game_id"),
                 r.get("market"), r.get("home_team"), r.get("away_team"),
                 r.get("pick_side"), r.get("line"), r.get("bet_odds"),
                 r.get("decimal_odds"), r.get("model_prob"),
                 r.get("market_prob_devigged"), r.get("p_push"), r.get("ev_pct"),
                 r.get("adjusted_ev_pct"), r.get("flat_stake"), r.get("confidence"),
                 r.get("tier"), r.get("stability"), r.get("edge_score"),
                 r.get("archetype"), r.get("projected_margin"), r.get("projected_total"),
                 1 if r.get("paper") else 0, r.get("model_tag") or "matchup_v1",
                 json.dumps(reasoning) if reasoning is not None else None,
                 r.get("opening_line"), r.get("opening_price"),
                 r.get("opening_devig_p_side"), r.get("closing_line"),
                 r.get("closing_price"), r.get("sharp_close_line"),
                 r.get("sharp_close_devig_p_side"), r.get("clv_pp"),
                 r.get("result") or "pending", r.get("home_score"), r.get("away_score"),
                 r.get("profit_loss"), 1 if r.get("is_value") else 0,
                 r.get("created_at"), r.get("updated_at")),
            )
            n += 1
    log.info("Pulled %d football recommendation row(s) from Supabase", n)
    return n
