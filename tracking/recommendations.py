"""Persistence for bet recommendations (SQLite).

Schema (table `recommendations`) matches the spec exactly. Money/P&L are tracked
in BANKROLL-FRACTION units: `kelly_stake` is the fraction of bankroll staked,
and `profit_loss` is the realized return in the same units, so ROI is
bankroll-independent. (Flat-stake "to 1u" ROI is derived separately in
tracking/performance.py from the stored odds + result.)

CLV workflow: `opening_line` is captured the first time a game is recommended;
re-running `today` later updates `closing_line` to the latest price and
recomputes `clv_pct`. So run `today` once early and again near first pitch to
get a genuine open->close CLV reading. CLV at low sample sizes is a better
signal of real edge than win rate — hence it's first-class here.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from mlb_value_bot.analysis.ev_calculator import american_to_decimal
from mlb_value_bot.utils import DB_PATH, ensure_dirs, get_logger

log = get_logger("tracking.recommendations")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    date                  TEXT NOT NULL,          -- game date YYYY-MM-DD
    game_id               INTEGER NOT NULL,
    home_team             TEXT NOT NULL,
    away_team             TEXT NOT NULL,
    recommended_side      TEXT NOT NULL,          -- 'home' | 'away'
    model_prob            REAL NOT NULL,
    market_prob_devigged  REAL NOT NULL,
    american_odds         INTEGER NOT NULL,       -- price used for the EV calc (bet price)
    decimal_odds          REAL NOT NULL,
    ev_pct                REAL NOT NULL,
    kelly_stake           REAL NOT NULL,          -- fraction of bankroll
    confidence            REAL NOT NULL,
    reasoning_json        TEXT,
    opening_line          INTEGER,                -- American odds at first capture
    closing_line          INTEGER,                -- American odds near first pitch
    clv_pct               REAL,                   -- open->close CLV, %
    result                TEXT DEFAULT 'pending', -- pending|win|loss|push|void
    profit_loss           REAL,                   -- realized, bankroll-fraction units
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    UNIQUE(date, game_id, recommended_side)
);
"""


@dataclass
class RecommendationRecord:
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
    opening_line: int | None = None
    closing_line: int | None = None
    clv_pct: float | None = None
    result: str = "pending"
    profit_loss: float | None = None
    id: int | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(_SCHEMA)
    log.debug("DB initialized at %s", DB_PATH)


def _compute_clv(opening: int | None, closing: int | None) -> float | None:
    """CLV% = how much better your taken (opening) price is vs the close.

    Positive => you beat the closing line (got higher decimal odds than close).
    """
    if opening is None or closing is None:
        return None
    try:
        return round((american_to_decimal(opening) / american_to_decimal(closing) - 1.0) * 100.0, 2)
    except (ValueError, ZeroDivisionError):
        return None


def upsert_recommendation(rec: RecommendationRecord) -> int:
    """Insert a new recommendation, or update lines/CLV if it already exists.

    On a repeat sighting of the same (date, game, side) we keep the original
    bet price + opening_line, but refresh `closing_line` to the latest American
    odds and recompute CLV. Returns the row id.
    """
    init_db()
    now = _now()
    with connect() as conn:
        existing = conn.execute(
            "SELECT id, opening_line FROM recommendations WHERE date=? AND game_id=? AND recommended_side=?",
            (rec.date, rec.game_id, rec.recommended_side),
        ).fetchone()

        if existing:
            opening = existing["opening_line"]
            closing = rec.american_odds  # latest price seen becomes the closing line
            clv = _compute_clv(opening, closing)
            conn.execute(
                "UPDATE recommendations SET closing_line=?, clv_pct=?, updated_at=? WHERE id=?",
                (closing, clv, now, existing["id"]),
            )
            log.info("Updated closing line for game %s (%s): %s (CLV %.2f%%)",
                     rec.game_id, rec.recommended_side, closing, clv if clv is not None else 0.0)
            return int(existing["id"])

        opening = rec.opening_line if rec.opening_line is not None else rec.american_odds
        cur = conn.execute(
            """
            INSERT INTO recommendations (
                date, game_id, home_team, away_team, recommended_side,
                model_prob, market_prob_devigged, american_odds, decimal_odds,
                ev_pct, kelly_stake, confidence, reasoning_json,
                opening_line, closing_line, clv_pct, result, profit_loss,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.date, rec.game_id, rec.home_team, rec.away_team, rec.recommended_side,
                rec.model_prob, rec.market_prob_devigged, rec.american_odds, rec.decimal_odds,
                rec.ev_pct, rec.kelly_stake, rec.confidence, json.dumps(rec.reasoning),
                opening, rec.closing_line, rec.clv_pct, rec.result, rec.profit_loss,
                now, now,
            ),
        )
        log.info("Saved recommendation: %s %s @ %+d (EV %.1f%%)",
                 rec.recommended_side, rec.home_team if rec.recommended_side == "home" else rec.away_team,
                 rec.american_odds, rec.ev_pct * 100)
        return int(cur.lastrowid)


def update_result(rec_id: int, result: str, profit_loss: float) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE recommendations SET result=?, profit_loss=?, updated_at=? WHERE id=?",
            (result, profit_loss, _now(), rec_id),
        )


def get_open_for_date(game_date: str) -> list[sqlite3.Row]:
    """Pending recommendations for a given game date."""
    init_db()
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM recommendations WHERE date=? AND result='pending'",
            (game_date,),
        ).fetchall()


def get_for_date(game_date: str) -> list[sqlite3.Row]:
    init_db()
    with connect() as conn:
        return conn.execute("SELECT * FROM recommendations WHERE date=?", (game_date,)).fetchall()


def to_dataframe(since: str | None = None) -> pd.DataFrame:
    """All recommendations (optionally on/after `since`) as a DataFrame."""
    init_db()
    query = "SELECT * FROM recommendations"
    params: tuple = ()
    if since:
        query += " WHERE date >= ?"
        params = (since,)
    query += " ORDER BY date, game_id"
    with connect() as conn:
        return pd.read_sql_query(query, conn, params=params)
