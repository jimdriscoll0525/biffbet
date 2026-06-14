"""GriffBet persistence — its OWN SQLite store, separate from BiffBet's.

BiffBet's tracking/recommendations.py hardwires DB_PATH and stores only the
best-price open/close. GriffBet needs (a) a separate DB so it never collides
with BiffBet on the (date, game_id) key, and (b) a RICHER schema: the raw-model
and blended pick streams, the sharp closing line, and TWO CLV streams graded
against that sharp close, plus the best-available ("obtainable") close.

Schema lives in `_SCHEMA`. The committed-bet rule mirrors BiffBet: once a bet is
committed (is_value=1) we NEVER downgrade it -- later runs only refresh the
closing lines + CLV. CLV convention matches BiffBet: open/close as decimal,
CLV% = decimal(open)/decimal(close) - 1, positive = the taken price beat close.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from mlb_value_bot.analysis.ev_calculator import american_to_decimal
from mlb_value_bot.griffbet import GRIFF_DB_PATH
from mlb_value_bot.utils import ensure_dirs, get_logger

log = get_logger("griffbet.tracking")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS griff_recommendations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    date                  TEXT NOT NULL,
    game_id               INTEGER NOT NULL,
    home_team             TEXT NOT NULL,
    away_team             TEXT NOT NULL,
    recommended_side      TEXT NOT NULL,          -- blended (committed) pick side
    model_prob            REAL NOT NULL,          -- blended pick-side prob (EV basis)
    market_prob_devigged  REAL NOT NULL,
    american_odds         INTEGER NOT NULL,       -- blended bet price
    decimal_odds          REAL NOT NULL,
    ev_pct                REAL NOT NULL,
    kelly_stake           REAL NOT NULL,          -- AFTER discipline
    confidence            REAL NOT NULL,
    reasoning_json        TEXT,
    -- CLV split ------------------------------------------------------------
    raw_model_prob        REAL,                   -- raw (pre-blend) home prob
    blended_prob          REAL,                   -- blended home prob
    raw_pick_side         TEXT,                   -- side the raw model would bet
    raw_pick_open         INTEGER,                -- raw-pick price at commit
    -- Opening / best-available ("obtainable") close on the BLENDED side ----
    opening_line          INTEGER,                -- blended pick open price
    closing_line          INTEGER,                -- blended pick best close
    clv_pct               REAL,                   -- blended open vs best close (obtainable)
    -- Sharp close (Pinnacle-preferred) + the two sharp CLV streams ---------
    sharp_close_book      TEXT,
    sharp_close_home_line INTEGER,
    sharp_close_away_line INTEGER,
    clv_raw_vs_sharp      REAL,                   -- raw-pick open vs sharp close (raw side)
    clv_blended_vs_sharp  REAL,                   -- blended open vs sharp close (blended side)
    -- Grading --------------------------------------------------------------
    result                TEXT DEFAULT 'pending',
    profit_loss           REAL,
    is_value              INTEGER NOT NULL DEFAULT 1,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    UNIQUE(date, game_id)
);
"""

# GriffBet-owned feature store. Holds the Stage-4 griff_features block for ANY
# game (date, game_id), including historical games that live only in BiffBet's
# frozen store. The training extractor joins this in, so backfilled features
# reach the model WITHOUT ever writing to BiffBet's data.
_FEATURE_SCHEMA = """
CREATE TABLE IF NOT EXISTS griff_game_features (
    date          TEXT NOT NULL,
    game_id       INTEGER NOT NULL,
    features_json TEXT NOT NULL,
    source        TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (date, game_id)
);
"""


@dataclass
class GriffRecord:
    date: str
    game_id: int
    home_team: str
    away_team: str
    recommended_side: str
    model_prob: float
    market_prob_devigged: float
    american_odds: int
    decimal_odds: float
    ev_pct: float
    kelly_stake: float
    confidence: float
    reasoning: dict = field(default_factory=dict)
    raw_model_prob: float | None = None
    blended_prob: float | None = None
    raw_pick_side: str | None = None
    raw_pick_open: int | None = None
    # Sharp + best lines for this run (per side); CLV is derived on upsert.
    sharp_close_book: str | None = None
    sharp_close_home_line: int | None = None
    sharp_close_away_line: int | None = None
    best_home_line: int | None = None
    best_away_line: int | None = None
    is_value: bool = True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(GRIFF_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(_SCHEMA)
        conn.executescript(_FEATURE_SCHEMA)


def _clv(open_price: int | None, close_price: int | None) -> float | None:
    if open_price is None or close_price is None:
        return None
    try:
        return round((american_to_decimal(open_price) / american_to_decimal(close_price) - 1.0) * 100.0, 2)
    except (ValueError, ZeroDivisionError):
        return None


def _side_price(home_line: int | None, away_line: int | None, side: str | None) -> int | None:
    if side == "home":
        return home_line
    if side == "away":
        return away_line
    return None


def upsert_recommendation(rec: GriffRecord) -> int:
    """Insert or refresh a GriffBet recommendation, keyed on (date, game_id).

    First insert sets the opening lines (blended + raw). On a committed bet
    (is_value=1) later runs NEVER downgrade -- they only refresh the closing
    lines + all CLV streams (best-available close and sharp close), mirroring
    BiffBet. Analysis-only rows (is_value=0) refresh fully and may be promoted.
    """
    init_db()
    now = _now()
    sharp_close_side = _side_price(rec.sharp_close_home_line, rec.sharp_close_away_line, rec.recommended_side)
    sharp_close_raw = _side_price(rec.sharp_close_home_line, rec.sharp_close_away_line, rec.raw_pick_side)
    best_close_side = _side_price(rec.best_home_line, rec.best_away_line, rec.recommended_side)

    with connect() as conn:
        existing = conn.execute(
            "SELECT id, opening_line, raw_pick_open, is_value FROM griff_recommendations "
            "WHERE date=? AND game_id=?",
            (rec.date, rec.game_id),
        ).fetchone()

        if existing and bool(existing["is_value"]):
            # Committed bet: keep opens, refresh closes + CLV only.
            opening = existing["opening_line"]
            raw_open = existing["raw_pick_open"]
            clv = _clv(opening, best_close_side)
            clv_blended_sharp = _clv(opening, sharp_close_side)
            clv_raw_sharp = _clv(raw_open, sharp_close_raw)
            conn.execute(
                """UPDATE griff_recommendations SET
                     closing_line=?, clv_pct=?, sharp_close_book=?,
                     sharp_close_home_line=?, sharp_close_away_line=?,
                     clv_raw_vs_sharp=?, clv_blended_vs_sharp=?, updated_at=?
                   WHERE id=?""",
                (best_close_side, clv, rec.sharp_close_book,
                 rec.sharp_close_home_line, rec.sharp_close_away_line,
                 clv_raw_sharp, clv_blended_sharp, now, existing["id"]),
            )
            return int(existing["id"])

        opening = rec.american_odds
        raw_open = rec.raw_pick_open
        clv = _clv(opening, best_close_side)
        clv_blended_sharp = _clv(opening, sharp_close_side)
        clv_raw_sharp = _clv(raw_open, sharp_close_raw)

        if existing:
            # Analysis-only row: refresh all model fields; promote if now a bet.
            now_bet = bool(rec.is_value)
            open_keep = rec.american_odds if now_bet else (existing["opening_line"] or rec.american_odds)
            conn.execute(
                """UPDATE griff_recommendations SET
                     home_team=?, away_team=?, recommended_side=?, model_prob=?,
                     market_prob_devigged=?, american_odds=?, decimal_odds=?, ev_pct=?,
                     kelly_stake=?, confidence=?, reasoning_json=?,
                     raw_model_prob=?, blended_prob=?, raw_pick_side=?, raw_pick_open=?,
                     opening_line=?, closing_line=?, clv_pct=?, sharp_close_book=?,
                     sharp_close_home_line=?, sharp_close_away_line=?,
                     clv_raw_vs_sharp=?, clv_blended_vs_sharp=?, is_value=?, updated_at=?
                   WHERE id=?""",
                (rec.home_team, rec.away_team, rec.recommended_side, rec.model_prob,
                 rec.market_prob_devigged, rec.american_odds, rec.decimal_odds, rec.ev_pct,
                 rec.kelly_stake, rec.confidence, json.dumps(rec.reasoning),
                 rec.raw_model_prob, rec.blended_prob, rec.raw_pick_side, raw_open,
                 open_keep, best_close_side, clv, rec.sharp_close_book,
                 rec.sharp_close_home_line, rec.sharp_close_away_line,
                 clv_raw_sharp, clv_blended_sharp, 1 if now_bet else 0, now,
                 existing["id"]),
            )
            return int(existing["id"])

        cur = conn.execute(
            """INSERT INTO griff_recommendations (
                 date, game_id, home_team, away_team, recommended_side, model_prob,
                 market_prob_devigged, american_odds, decimal_odds, ev_pct, kelly_stake,
                 confidence, reasoning_json, raw_model_prob, blended_prob, raw_pick_side,
                 raw_pick_open, opening_line, closing_line, clv_pct, sharp_close_book,
                 sharp_close_home_line, sharp_close_away_line, clv_raw_vs_sharp,
                 clv_blended_vs_sharp, result, profit_loss, is_value, created_at, updated_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec.date, rec.game_id, rec.home_team, rec.away_team, rec.recommended_side,
             rec.model_prob, rec.market_prob_devigged, rec.american_odds, rec.decimal_odds,
             rec.ev_pct, rec.kelly_stake, rec.confidence, json.dumps(rec.reasoning),
             rec.raw_model_prob, rec.blended_prob, rec.raw_pick_side, raw_open,
             opening, best_close_side, clv, rec.sharp_close_book,
             rec.sharp_close_home_line, rec.sharp_close_away_line,
             clv_raw_sharp, clv_blended_sharp, "pending", None,
             1 if rec.is_value else 0, now, now),
        )
        return int(cur.lastrowid)


def refresh_closing_lines(
    game_date: str,
    game_id: int,
    *,
    best_home_line: int | None,
    best_away_line: int | None,
    sharp_close_book: str | None,
    sharp_close_home_line: int | None,
    sharp_close_away_line: int | None,
    raw_pick_side: str | None = None,
) -> bool:
    """Refresh closing lines + all CLV streams on a committed bet WITHOUT a full
    record -- used for games the pipeline sanity-SKIPPED this run, whose CLV
    would otherwise freeze. Keeps opens; recomputes best + sharp CLV. Returns
    True if a committed bet was updated.
    """
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT id, recommended_side, opening_line, raw_pick_side, raw_pick_open "
            "FROM griff_recommendations WHERE date=? AND game_id=? AND is_value=1",
            (game_date, game_id),
        ).fetchone()
        if row is None:
            return False
        raw_side = raw_pick_side or row["raw_pick_side"]
        best_close_side = _side_price(best_home_line, best_away_line, row["recommended_side"])
        sharp_close_side = _side_price(sharp_close_home_line, sharp_close_away_line, row["recommended_side"])
        sharp_close_raw = _side_price(sharp_close_home_line, sharp_close_away_line, raw_side)
        conn.execute(
            """UPDATE griff_recommendations SET
                 closing_line=?, clv_pct=?, sharp_close_book=?,
                 sharp_close_home_line=?, sharp_close_away_line=?,
                 clv_raw_vs_sharp=?, clv_blended_vs_sharp=?, updated_at=?
               WHERE id=?""",
            (best_close_side, _clv(row["opening_line"], best_close_side), sharp_close_book,
             sharp_close_home_line, sharp_close_away_line,
             _clv(row["raw_pick_open"], sharp_close_raw),
             _clv(row["opening_line"], sharp_close_side), _now(), row["id"]),
        )
        return True


def update_result(rec_id: int, result: str, profit_loss: float) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE griff_recommendations SET result=?, profit_loss=?, updated_at=? WHERE id=?",
            (result, profit_loss, _now(), rec_id),
        )


def get_open_for_date(game_date: str) -> list[sqlite3.Row]:
    init_db()
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM griff_recommendations WHERE date=? AND result='pending' AND is_value=1",
            (game_date,),
        ).fetchall()


def get_open_dates(before: str | None = None) -> list[str]:
    init_db()
    query = "SELECT DISTINCT date FROM griff_recommendations WHERE result='pending' AND is_value=1"
    params: tuple = ()
    if before:
        query += " AND date < ?"
        params = (before,)
    query += " ORDER BY date"
    with connect() as conn:
        return [r["date"] for r in conn.execute(query, params)]


def get_for_date(game_date: str) -> list[sqlite3.Row]:
    init_db()
    with connect() as conn:
        return conn.execute("SELECT * FROM griff_recommendations WHERE date=?", (game_date,)).fetchall()


def upsert_features(date: str, game_id: int, features: dict, source: str = "backfill") -> None:
    """Store a Stage-4 griff_features block for a game in the feature store."""
    init_db()
    with connect() as conn:
        conn.execute(
            """INSERT INTO griff_game_features (date, game_id, features_json, source, updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(date, game_id) DO UPDATE SET
                 features_json=excluded.features_json, source=excluded.source,
                 updated_at=excluded.updated_at""",
            (date, int(game_id), json.dumps(features), source, _now()),
        )


def get_all_features() -> dict[tuple, dict]:
    """{(date, game_id): features_dict} from the feature store (for training)."""
    init_db()
    out: dict[tuple, dict] = {}
    with connect() as conn:
        for r in conn.execute("SELECT date, game_id, features_json FROM griff_game_features"):
            try:
                out[(r["date"], int(r["game_id"]))] = json.loads(r["features_json"])
            except json.JSONDecodeError:
                continue
    return out


def all_feature_rows() -> list[dict]:
    """Raw feature-store rows for syncing to Supabase."""
    init_db()
    with connect() as conn:
        return [
            {"date": r["date"], "game_id": int(r["game_id"]),
             "features": json.loads(r["features_json"]),
             "source": r["source"], "updated_at": r["updated_at"]}
            for r in conn.execute("SELECT * FROM griff_game_features")
        ]


def upsert_feature_row(date: str, game_id: int, features: dict, source: str, updated_at: str) -> None:
    """Insert a feature-store row verbatim (used by the Supabase pull)."""
    init_db()
    with connect() as conn:
        conn.execute(
            """INSERT INTO griff_game_features (date, game_id, features_json, source, updated_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(date, game_id) DO UPDATE SET
                 features_json=excluded.features_json, source=excluded.source,
                 updated_at=excluded.updated_at""",
            (date, int(game_id), json.dumps(features), source, updated_at),
        )


def to_dataframe(since: str | None = None) -> pd.DataFrame:
    init_db()
    query = "SELECT * FROM griff_recommendations"
    params: tuple = ()
    if since:
        query += " WHERE date >= ?"
        params = (since,)
    query += " ORDER BY date, game_id"
    with connect() as conn:
        return pd.read_sql_query(query, conn, params=params)
