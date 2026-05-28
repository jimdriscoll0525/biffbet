"""Push the engine's tracking data to Supabase (Postgres) for the public site.

The BiffBet website (Next.js) is a thin, read-only view over two Supabase tables:

  * recommendations      — mirror of the local SQLite `recommendations` table
  * performance_snapshot  — precomputed PerformanceReport (overall + segments)

This module reads what `tracking/*` already produced and upserts it via the
Supabase PostgREST API. It is the ONLY place that talks to Supabase from Python,
and it never touches the model — `today`/`results` run exactly as before, then
`sync` mirrors the result to the cloud.

Auth: uses the SERVICE-ROLE key (server-side only, bypasses RLS). Set in .env:
    SUPABASE_URL=https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY=<service-role secret>   # NEVER ship this to the browser

Idempotent: upserts on the natural keys, so re-running is safe and cheap.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from mlb_value_bot.tracking import performance as perf
from mlb_value_bot.tracking import recommendations as recs
from mlb_value_bot.utils import get_env, get_logger

log = get_logger("sync.supabase")

# PostgREST request tuning.
_BATCH = 500            # rows per upsert request (recommendations)
_TIMEOUT = 30           # seconds per HTTP call

# Columns we push to the `recommendations` table. We deliberately DROP the SQLite
# `id` (Supabase owns its own identity) and send `reasoning` as parsed jsonb
# instead of the raw `reasoning_json` string.
_REC_COLUMNS = (
    "date", "game_id", "home_team", "away_team", "recommended_side",
    "model_prob", "market_prob_devigged", "american_odds", "decimal_odds",
    "ev_pct", "kelly_stake", "confidence", "opening_line", "closing_line",
    "clv_pct", "result", "profit_loss", "is_value",
    "created_at", "updated_at",
)


class SupabaseConfigError(RuntimeError):
    """Raised when the Supabase URL / service key are not configured."""


@dataclass
class SyncResult:
    recommendations: int
    performance_scopes: int


# --- credentials -------------------------------------------------------------
def _credentials() -> tuple[str, str]:
    """Resolve (base_url, service_key) from the environment or raise."""
    url = get_env("SUPABASE_URL")
    # Newer projects issue a "secret" key (sb_secret_...); older ones a
    # service_role key. Both bypass RLS so the engine can write.
    key = (
        get_env("SUPABASE_SERVICE_KEY")
        or get_env("SUPABASE_SERVICE_ROLE_KEY")
        or get_env("SUPABASE_SECRET_KEY")
    )
    if not url or not key:
        raise SupabaseConfigError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_SECRET_KEY) "
            "in your .env (Supabase Dashboard -> Project Settings -> API Keys)."
        )
    return url.rstrip("/"), key


def _headers(key: str, *, upsert_on: str | None) -> dict[str, str]:
    prefer = "return=minimal"
    if upsert_on:
        prefer = f"resolution=merge-duplicates,{prefer}"
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


# --- JSON sanitation ---------------------------------------------------------
def _clean(value: Any) -> Any:
    """Make a value JSON-safe for PostgREST.

    pandas/NumPy leak NaN (invalid JSON) and numpy scalar types; convert NaN/NaT
    to None and coerce numpy scalars to native Python types, recursively.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    # numpy scalar -> python scalar
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _clean(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _post(url: str, key: str, table: str, rows: list[dict], *, on_conflict: str) -> None:
    """Upsert a batch of rows into `table`, merging on `on_conflict` columns."""
    if not rows:
        return
    endpoint = f"{url}/rest/v1/{table}?on_conflict={on_conflict}"
    resp = requests.post(
        endpoint,
        headers=_headers(key, upsert_on=on_conflict),
        data=json.dumps(rows, default=str),
        timeout=_TIMEOUT,
    )
    if resp.status_code >= 300:
        raise RuntimeError(
            f"Supabase upsert into {table} failed ({resp.status_code}): {resp.text[:500]}"
        )


# --- recommendations ---------------------------------------------------------
def _rec_rows(since: str | None) -> list[dict]:
    df: pd.DataFrame = recs.to_dataframe(since=since)
    if df.empty:
        return []
    rows: list[dict] = []
    # Postgres integer columns reject float literals like "390.0". pandas promotes
    # a nullable int column (e.g. closing_line, NULL on some rows) to float64, so
    # coerce these back to int before sending.
    int_cols = ("game_id", "american_odds", "opening_line", "closing_line")
    for _, r in df.iterrows():
        row = {col: _clean(r.get(col)) for col in _REC_COLUMNS}
        for col in int_cols:
            if row.get(col) is not None:
                row[col] = int(row[col])
        # SQLite stores is_value as 0/1; Postgres column is boolean.
        if row.get("is_value") is not None:
            row["is_value"] = bool(int(row["is_value"]))
        else:
            # Old rows synced before the column existed -> treat as bets.
            row["is_value"] = True
        # reasoning_json (TEXT) -> reasoning (jsonb)
        raw = r.get("reasoning_json")
        if isinstance(raw, str) and raw:
            try:
                row["reasoning"] = _clean(json.loads(raw))
            except json.JSONDecodeError:
                row["reasoning"] = None
        else:
            row["reasoning"] = None
        rows.append(row)
    return rows


def push_recommendations(url: str, key: str, since: str | None = None) -> int:
    rows = _rec_rows(since)
    for start in range(0, len(rows), _BATCH):
        _post(url, key, "recommendations", rows[start:start + _BATCH],
              on_conflict="date,game_id,recommended_side")
    log.info("Synced %d recommendation(s) to Supabase.", len(rows))
    return len(rows)


# --- performance snapshot ----------------------------------------------------
def _segments_to_json(segments: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for title, df in segments.items():
        if df is None or df.empty:
            out[title] = []
            continue
        out[title] = [_clean(rec) for rec in df.to_dict(orient="records")]
    return out


def push_performance(url: str, key: str, since: str | None = None) -> int:
    report = perf.compute_performance(since=since)
    scope = "all" if not since else f"since:{since}"
    payload = [{
        "scope": scope,
        "overall": _clean(report.overall),
        "segments": _segments_to_json(report.segments),
    }]
    _post(url, key, "performance_snapshot", payload, on_conflict="scope")
    log.info("Synced performance snapshot (scope=%s) to Supabase.", scope)
    return 1


# --- entrypoint --------------------------------------------------------------
def push_all(since: str | None = None) -> SyncResult:
    """Push recommendations + the performance snapshot to Supabase."""
    url, key = _credentials()
    n_recs = push_recommendations(url, key, since=since)
    n_perf = push_performance(url, key, since=since)
    return SyncResult(recommendations=n_recs, performance_scopes=n_perf)


# --- pull (Supabase -> SQLite) -----------------------------------------------
def _get_all(url: str, key: str, table: str, *, select: str = "*", order: str = "id.asc") -> list[dict]:
    """Fetch every row of a table, paginating past the PostgREST row cap."""
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    out: list[dict] = []
    page, offset = 1000, 0
    while True:
        resp = requests.get(
            f"{url}/rest/v1/{table}?select={select}&order={order}&limit={page}&offset={offset}",
            headers=headers,
            timeout=_TIMEOUT,
        )
        if resp.status_code >= 300:
            raise RuntimeError(
                f"Supabase read from {table} failed ({resp.status_code}): {resp.text[:300]}"
            )
        batch = resp.json()
        out.extend(batch)
        if len(batch) < page:
            return out
        offset += page


def pull_recommendations() -> int:
    """Rebuild the local SQLite recommendations from Supabase.

    The hosted pipeline (GitHub Action) runs on an ephemeral machine, so Supabase
    is the source of truth: pull restores all prior bets (incl. results + CLV) so
    `results`/`today` can settle and re-price, then `sync` pushes back. Idempotent
    upsert on the natural key, so it's safe to run before every pipeline run.
    """
    url, key = _credentials()
    rows = _get_all(url, key, "recommendations")

    from mlb_value_bot.tracking.recommendations import connect, init_db

    init_db()
    with connect() as conn:
        for r in rows:
            reasoning = r.get("reasoning")
            reasoning_json = json.dumps(reasoning) if reasoning is not None else None
            # Supabase returns boolean true/false; SQLite needs 0/1. Default to
            # 1 (treat as bet) for rows synced before the column existed.
            is_value_raw = r.get("is_value")
            is_value = 1 if is_value_raw is None else (1 if bool(is_value_raw) else 0)
            conn.execute(
                """
                INSERT INTO recommendations
                  (date, game_id, home_team, away_team, recommended_side, model_prob,
                   market_prob_devigged, american_odds, decimal_odds, ev_pct, kelly_stake,
                   confidence, reasoning_json, opening_line, closing_line, clv_pct, result,
                   profit_loss, is_value, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(date, game_id, recommended_side) DO UPDATE SET
                  home_team=excluded.home_team, away_team=excluded.away_team,
                  model_prob=excluded.model_prob,
                  market_prob_devigged=excluded.market_prob_devigged,
                  american_odds=excluded.american_odds, decimal_odds=excluded.decimal_odds,
                  ev_pct=excluded.ev_pct, kelly_stake=excluded.kelly_stake,
                  confidence=excluded.confidence, reasoning_json=excluded.reasoning_json,
                  opening_line=excluded.opening_line, closing_line=excluded.closing_line,
                  clv_pct=excluded.clv_pct, result=excluded.result,
                  profit_loss=excluded.profit_loss, is_value=excluded.is_value,
                  updated_at=excluded.updated_at
                """,
                (
                    r["date"], r["game_id"], r["home_team"], r["away_team"], r["recommended_side"],
                    r["model_prob"], r["market_prob_devigged"], r["american_odds"], r["decimal_odds"],
                    r["ev_pct"], r["kelly_stake"], r["confidence"], reasoning_json,
                    r.get("opening_line"), r.get("closing_line"), r.get("clv_pct"),
                    r.get("result", "pending"), r.get("profit_loss"), is_value,
                    r.get("created_at"), r.get("updated_at"),
                ),
            )
    log.info("Pulled %d recommendation(s) from Supabase into local SQLite.", len(rows))
    return len(rows)
