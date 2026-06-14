"""Push/pull GriffBet's data to Supabase — NEW tables, same project.

Reuses BiffBet's Supabase plumbing (credentials, headers, batched upsert,
paginated read, JSON sanitation) by import, but writes to GriffBet's OWN tables
so it never collides with BiffBet on (date, game_id):

  * griffbet_recommendations    — mirror of griff_recommendations (richer schema)
  * referee_snapshot            — the cross-model referee report (scope keyed)

BiffBet's tables are untouched. Same SUPABASE_URL + service key (it bypasses RLS
for writes; the public site reads via the anon key under SELECT-only RLS).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from mlb_value_bot.sync.supabase_sync import (
    _BATCH,
    _clean,
    _credentials,
    _get_all,
    _post,
)
from mlb_value_bot.griffbet import tracking as gtrack
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.sync")

_GRIFF_REC_COLUMNS = (
    "date", "game_id", "home_team", "away_team", "recommended_side",
    "model_prob", "market_prob_devigged", "american_odds", "decimal_odds",
    "ev_pct", "kelly_stake", "confidence",
    "raw_model_prob", "blended_prob", "raw_pick_side", "raw_pick_open",
    "opening_line", "closing_line", "clv_pct",
    "sharp_close_book", "sharp_close_home_line", "sharp_close_away_line",
    "clv_raw_vs_sharp", "clv_blended_vs_sharp",
    "result", "profit_loss", "is_value", "created_at", "updated_at",
)
_INT_COLS = ("game_id", "american_odds", "raw_pick_open", "opening_line",
             "closing_line", "sharp_close_home_line", "sharp_close_away_line")


def _rec_rows(since: str | None) -> list[dict]:
    df: pd.DataFrame = gtrack.to_dataframe(since=since)
    if df.empty:
        return []
    rows: list[dict] = []
    for _, r in df.iterrows():
        row = {col: _clean(r.get(col)) for col in _GRIFF_REC_COLUMNS}
        for col in _INT_COLS:
            if row.get(col) is not None:
                row[col] = int(row[col])
        row["is_value"] = bool(int(row["is_value"])) if row.get("is_value") is not None else True
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
        _post(url, key, "griffbet_recommendations", rows[start:start + _BATCH],
              on_conflict="date,game_id")
    log.info("Synced %d GriffBet recommendation(s) to Supabase.", len(rows))
    return len(rows)


def push_referee(url: str, key: str, referee: dict) -> int:
    payload = [{
        "scope": "all",
        "data": _clean(referee),
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }]
    _post(url, key, "referee_snapshot", payload, on_conflict="scope")
    log.info("Synced referee snapshot to Supabase.")
    return 1


def push_all(referee: dict | None = None, since: str | None = None) -> dict:
    url, key = _credentials()
    n = push_recommendations(url, key, since=since)
    out = {"recommendations": n}
    if referee is not None:
        out["referee"] = push_referee(url, key, referee)
    return out


def pull_recommendations() -> int:
    """Rebuild local griff_recommendations from Supabase (cloud is source of
    truth on the ephemeral CI box). Idempotent upsert on (date, game_id)."""
    url, key = _credentials()
    rows = _get_all(url, key, "griffbet_recommendations")
    gtrack.init_db()
    with gtrack.connect() as conn:
        for r in rows:
            reasoning = r.get("reasoning")
            reasoning_json = json.dumps(reasoning) if reasoning is not None else None
            is_value = 1 if r.get("is_value") in (None, True, 1) else 0
            conn.execute(
                """INSERT INTO griff_recommendations
                     (date, game_id, home_team, away_team, recommended_side, model_prob,
                      market_prob_devigged, american_odds, decimal_odds, ev_pct, kelly_stake,
                      confidence, reasoning_json, raw_model_prob, blended_prob, raw_pick_side,
                      raw_pick_open, opening_line, closing_line, clv_pct, sharp_close_book,
                      sharp_close_home_line, sharp_close_away_line, clv_raw_vs_sharp,
                      clv_blended_vs_sharp, result, profit_loss, is_value, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(date, game_id) DO UPDATE SET
                     home_team=excluded.home_team, away_team=excluded.away_team,
                     recommended_side=excluded.recommended_side, model_prob=excluded.model_prob,
                     market_prob_devigged=excluded.market_prob_devigged,
                     american_odds=excluded.american_odds, decimal_odds=excluded.decimal_odds,
                     ev_pct=excluded.ev_pct, kelly_stake=excluded.kelly_stake,
                     confidence=excluded.confidence, reasoning_json=excluded.reasoning_json,
                     raw_model_prob=excluded.raw_model_prob, blended_prob=excluded.blended_prob,
                     raw_pick_side=excluded.raw_pick_side, raw_pick_open=excluded.raw_pick_open,
                     opening_line=excluded.opening_line, closing_line=excluded.closing_line,
                     clv_pct=excluded.clv_pct, sharp_close_book=excluded.sharp_close_book,
                     sharp_close_home_line=excluded.sharp_close_home_line,
                     sharp_close_away_line=excluded.sharp_close_away_line,
                     clv_raw_vs_sharp=excluded.clv_raw_vs_sharp,
                     clv_blended_vs_sharp=excluded.clv_blended_vs_sharp,
                     result=excluded.result, profit_loss=excluded.profit_loss,
                     is_value=excluded.is_value, updated_at=excluded.updated_at""",
                (r["date"], r["game_id"], r["home_team"], r["away_team"], r["recommended_side"],
                 r["model_prob"], r["market_prob_devigged"], r["american_odds"], r["decimal_odds"],
                 r["ev_pct"], r["kelly_stake"], r["confidence"], reasoning_json,
                 r.get("raw_model_prob"), r.get("blended_prob"), r.get("raw_pick_side"),
                 r.get("raw_pick_open"), r.get("opening_line"), r.get("closing_line"),
                 r.get("clv_pct"), r.get("sharp_close_book"), r.get("sharp_close_home_line"),
                 r.get("sharp_close_away_line"), r.get("clv_raw_vs_sharp"),
                 r.get("clv_blended_vs_sharp"), r.get("result", "pending"),
                 r.get("profit_loss"), is_value, r.get("created_at"), r.get("updated_at")),
            )
    log.info("Pulled %d GriffBet recommendation(s) from Supabase.", len(rows))
    return len(rows)
