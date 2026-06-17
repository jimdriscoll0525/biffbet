"""Persistence for TOTALS (over/under) picks -- a parallel table to the moneyline
`recommendations`, in the SAME SQLite DB but keyed independently on (date,
game_id), so a game can carry both a moneyline pick and a totals pick without
collision.

Totals-specific bookkeeping:
  * PAPER gate: every pick stores `paper` (1 while config.totals.paper_only).
    Real-money sizing stays off until CLV vs the totals close proves out.
  * CLV is a PROBABILITY move, not a price ratio (the totals LINE moves, so the
    moneyline's decimal-ratio CLV doesn't transfer). We freeze the de-vigged
    market P(pick side) at commit (`opening_devig_p_side`) and compare it to the
    de-vigged P(pick side) at the SHARP totals close: clv_pp = (close - open)*100,
    positive = the market moved toward our side (we beat the close).
  * Grading is line-aware: final total vs the line we BET, with integer-line
    pushes refunding the stake.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from mlb_value_bot.utils import DB_PATH, ensure_dirs, get_logger

log = get_logger("tracking.totals")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS totals_recommendations (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    date                     TEXT NOT NULL,
    game_id                  INTEGER NOT NULL,
    home_team                TEXT NOT NULL,
    away_team                TEXT NOT NULL,
    pick_side                TEXT NOT NULL,          -- 'over' | 'under'
    market_total             REAL,                   -- the line bet
    over_odds                INTEGER,
    under_odds               INTEGER,
    bet_odds                 INTEGER NOT NULL,       -- picked-side price (EV basis)
    decimal_odds             REAL NOT NULL,
    model_p_over             REAL,                   -- conditional model P(over)
    market_devig_over        REAL,                   -- de-vigged market P(over)
    blended_p_over           REAL,
    model_prob               REAL NOT NULL,          -- blended P(picked side)
    market_prob_devigged     REAL NOT NULL,          -- de-vig P(picked side)
    ev_pct                   REAL NOT NULL,
    kelly_stake              REAL NOT NULL,          -- fraction of bankroll (paper)
    confidence               REAL NOT NULL,
    tier                     TEXT,
    stability                TEXT,
    raw_model_total          REAL,
    expected_total           REAL,
    paper                    INTEGER NOT NULL DEFAULT 1,
    reasoning_json           TEXT,
    -- CLV (probability-pp move vs the sharp totals close) --------------------
    opening_line             REAL,                   -- line at commit
    opening_price            INTEGER,                -- picked-side price at commit
    opening_devig_p_side     REAL,                   -- de-vig P(side) at commit (CLV entry)
    closing_line             REAL,                   -- best-book line near close
    closing_price            INTEGER,
    sharp_close_book         TEXT,
    sharp_close_line         REAL,
    sharp_close_over         INTEGER,
    sharp_close_under        INTEGER,
    sharp_close_devig_p_side REAL,                   -- de-vig P(side) at sharp close
    clv_pp                   REAL,                   -- (close - entry) * 100 pp
    result                   TEXT DEFAULT 'pending', -- pending|win|loss|push|void
    profit_loss              REAL,                   -- paper, bankroll-fraction units
    is_value                 INTEGER NOT NULL DEFAULT 1,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    UNIQUE(date, game_id)
);
"""


@dataclass
class TotalsRecommendationRecord:
    date: str
    game_id: int
    home_team: str
    away_team: str
    pick_side: str
    bet_odds: int
    decimal_odds: float
    model_prob: float
    market_prob_devigged: float
    ev_pct: float
    kelly_stake: float
    confidence: float
    market_total: float | None = None
    over_odds: int | None = None
    under_odds: int | None = None
    model_p_over: float | None = None
    market_devig_over: float | None = None
    blended_p_over: float | None = None
    tier: str | None = None
    stability: str | None = None
    raw_model_total: float | None = None
    expected_total: float | None = None
    paper: bool = True
    opening_devig_p_side: float | None = None
    sharp_close: object | None = None      # SharpTotalsLine | None
    best_close_line: float | None = None
    best_close_price: int | None = None
    reasoning: dict = field(default_factory=dict)
    is_value: bool = True
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
    log.debug("Totals DB initialized at %s", DB_PATH)


def _clv_pp(opening: float | None, closing: float | None) -> float | None:
    """CLV in probability points: how far the sharp close moved TOWARD our side.

    Positive => the de-vigged P(our side) is higher at the close than when we
    bet => the market came to us => we beat the close. Both inputs are no-vig
    P(pick side), so this is line-move-robust (unlike a price ratio)."""
    if opening is None or closing is None:
        return None
    return round((closing - opening) * 100.0, 2)


def _sharp_close_fields(rec_pick_side: str, sharp_close) -> dict:
    """Extract the sharp-close columns (incl. de-vig P(pick side)) from a
    SharpTotalsLine, or all-None if absent."""
    if sharp_close is None:
        return {"book": None, "line": None, "over": None, "under": None, "devig_side": None}
    devig_over = sharp_close.devig_over
    devig_side = devig_over if rec_pick_side == "over" else (
        None if devig_over is None else 1.0 - devig_over
    )
    return {
        "book": sharp_close.book, "line": sharp_close.line,
        "over": sharp_close.over_price, "under": sharp_close.under_price,
        "devig_side": round(devig_side, 4) if devig_side is not None else None,
    }


def upsert_totals_recommendation(rec: TotalsRecommendationRecord) -> int:
    """Insert a totals pick, or update it if (date, game_id) already exists.

    Semantics mirror the moneyline upsert:
      * committed paper bet (is_value=1): NEVER downgrade -- keep the opening
        line/price/de-vig, only refresh the close (best line + sharp close) + CLV.
      * analysis -> bet promotion: this run's price becomes the opening reference.
      * analysis -> analysis: refresh all model fields with the latest run.
    """
    init_db()
    now = _now()
    sc = _sharp_close_fields(rec.pick_side, rec.sharp_close)
    with connect() as conn:
        existing = conn.execute(
            "SELECT id, opening_line, opening_price, opening_devig_p_side, is_value "
            "FROM totals_recommendations WHERE date=? AND game_id=?",
            (rec.date, rec.game_id),
        ).fetchone()

        if existing:
            was_bet = bool(existing["is_value"])
            now_bet = bool(rec.is_value)

            if was_bet:
                # Frozen commit: keep opening_*; refresh close + sharp close + CLV.
                open_devig = existing["opening_devig_p_side"]
                clv = _clv_pp(open_devig, sc["devig_side"])
                conn.execute(
                    "UPDATE totals_recommendations SET "
                    "closing_line=?, closing_price=?, sharp_close_book=?, sharp_close_line=?, "
                    "sharp_close_over=?, sharp_close_under=?, sharp_close_devig_p_side=?, "
                    "clv_pp=?, updated_at=? WHERE id=?",
                    (rec.best_close_line, rec.best_close_price, sc["book"], sc["line"],
                     sc["over"], sc["under"], sc["devig_side"], clv, now, existing["id"]),
                )
                log.info("Refreshed totals close for game %s (%s): sharp %s CLV %s pp",
                         rec.game_id, rec.pick_side, sc["line"], clv if clv is not None else "-")
                return int(existing["id"])

            # Was analysis: refresh everything. If now a bet, this run sets the
            # opening reference.
            open_line = rec.market_total if now_bet else (existing["opening_line"] if existing["opening_line"] is not None else rec.market_total)
            open_price = rec.bet_odds if now_bet else (existing["opening_price"] if existing["opening_price"] is not None else rec.bet_odds)
            open_devig = rec.opening_devig_p_side if now_bet else (
                existing["opening_devig_p_side"] if existing["opening_devig_p_side"] is not None else rec.opening_devig_p_side
            )
            clv = _clv_pp(open_devig, sc["devig_side"]) if now_bet else None
            conn.execute(
                """
                UPDATE totals_recommendations SET
                    home_team=?, away_team=?, pick_side=?, market_total=?, over_odds=?, under_odds=?,
                    bet_odds=?, decimal_odds=?, model_p_over=?, market_devig_over=?, blended_p_over=?,
                    model_prob=?, market_prob_devigged=?, ev_pct=?, kelly_stake=?, confidence=?,
                    tier=?, stability=?, raw_model_total=?, expected_total=?, paper=?, reasoning_json=?,
                    opening_line=?, opening_price=?, opening_devig_p_side=?, closing_line=?, closing_price=?,
                    sharp_close_book=?, sharp_close_line=?, sharp_close_over=?, sharp_close_under=?,
                    sharp_close_devig_p_side=?, clv_pp=?, is_value=?, updated_at=?
                WHERE id=?
                """,
                (rec.home_team, rec.away_team, rec.pick_side, rec.market_total, rec.over_odds, rec.under_odds,
                 rec.bet_odds, rec.decimal_odds, rec.model_p_over, rec.market_devig_over, rec.blended_p_over,
                 rec.model_prob, rec.market_prob_devigged, rec.ev_pct, rec.kelly_stake, rec.confidence,
                 rec.tier, rec.stability, rec.raw_model_total, rec.expected_total, 1 if rec.paper else 0,
                 json.dumps(rec.reasoning), open_line, open_price, open_devig, rec.best_close_line, rec.best_close_price,
                 sc["book"], sc["line"], sc["over"], sc["under"], sc["devig_side"], clv,
                 1 if now_bet else 0, now, existing["id"]),
            )
            if now_bet:
                log.info("Promoted totals game %s (%s) to a paper bet @ line %s (EV %.1f%%)",
                         rec.game_id, rec.pick_side, rec.market_total, rec.ev_pct * 100)
            return int(existing["id"])

        # New row.
        open_line = rec.market_total
        open_price = rec.bet_odds
        open_devig = rec.opening_devig_p_side
        clv = _clv_pp(open_devig, sc["devig_side"]) if rec.is_value else None
        cur = conn.execute(
            """
            INSERT INTO totals_recommendations (
                date, game_id, home_team, away_team, pick_side, market_total, over_odds, under_odds,
                bet_odds, decimal_odds, model_p_over, market_devig_over, blended_p_over, model_prob,
                market_prob_devigged, ev_pct, kelly_stake, confidence, tier, stability, raw_model_total,
                expected_total, paper, reasoning_json, opening_line, opening_price, opening_devig_p_side,
                closing_line, closing_price, sharp_close_book, sharp_close_line, sharp_close_over,
                sharp_close_under, sharp_close_devig_p_side, clv_pp, result, profit_loss, is_value,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (rec.date, rec.game_id, rec.home_team, rec.away_team, rec.pick_side, rec.market_total,
             rec.over_odds, rec.under_odds, rec.bet_odds, rec.decimal_odds, rec.model_p_over,
             rec.market_devig_over, rec.blended_p_over, rec.model_prob, rec.market_prob_devigged,
             rec.ev_pct, rec.kelly_stake, rec.confidence, rec.tier, rec.stability, rec.raw_model_total,
             rec.expected_total, 1 if rec.paper else 0, json.dumps(rec.reasoning), open_line, open_price,
             open_devig, rec.best_close_line, rec.best_close_price, sc["book"], sc["line"], sc["over"],
             sc["under"], sc["devig_side"], clv, "pending", None, 1 if rec.is_value else 0, now, now),
        )
        log.info("Saved totals %s: %s %.1f @ %+d (EV %.1f%%)%s",
                 "bet" if rec.is_value else "analysis", rec.pick_side, rec.market_total or 0.0,
                 rec.bet_odds, rec.ev_pct * 100, " [PAPER]" if rec.paper else "")
        return int(cur.lastrowid)


def update_result(rec_id: int, result: str, profit_loss: float) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE totals_recommendations SET result=?, profit_loss=?, updated_at=? WHERE id=?",
            (result, profit_loss, _now(), rec_id),
        )


def refresh_totals_close(game_date: str, game_id: int, analysis) -> bool:
    """Update the sharp close + best-book close + CLV on a committed paper totals
    bet whose game the pipeline did NOT save this run (a sanity skip). Reads the
    DB row's pick_side + opening de-vig and applies the latest sharp close from
    the TotalsAnalysis. Returns True if a bet was updated."""
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT id, pick_side, opening_devig_p_side FROM totals_recommendations "
            "WHERE date=? AND game_id=? AND is_value=1",
            (game_date, game_id),
        ).fetchone()
        if row is None:
            return False
        sc = _sharp_close_fields(row["pick_side"], getattr(analysis, "sharp_close", None))
        if sc["devig_side"] is None:
            return False
        intel = getattr(analysis, "intel", None)
        best_line = getattr(intel, "bet_line", None) if intel is not None else None
        best_price = None
        if intel is not None:
            best_price = intel.best_over_price if row["pick_side"] == "over" else intel.best_under_price
        clv = _clv_pp(row["opening_devig_p_side"], sc["devig_side"])
        conn.execute(
            "UPDATE totals_recommendations SET closing_line=?, closing_price=?, sharp_close_book=?, "
            "sharp_close_line=?, sharp_close_over=?, sharp_close_under=?, sharp_close_devig_p_side=?, "
            "clv_pp=?, updated_at=? WHERE id=?",
            (best_line, best_price, sc["book"], sc["line"], sc["over"], sc["under"],
             sc["devig_side"], clv, _now(), row["id"]),
        )
        log.info("Refreshed totals close for skipped game %s (%s): CLV %s pp",
                 game_id, row["pick_side"], clv if clv is not None else "-")
        return True


def get_open_for_date(game_date: str) -> list[sqlite3.Row]:
    """Pending totals BETS (is_value=1) for a date -- the grading worklist."""
    init_db()
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM totals_recommendations WHERE date=? AND result='pending' AND is_value=1",
            (game_date,),
        ).fetchall()


def get_open_dates(before: str | None = None) -> list[str]:
    """Distinct dates with pending totals bets, ascending (grading backfill)."""
    init_db()
    query = "SELECT DISTINCT date FROM totals_recommendations WHERE result='pending' AND is_value=1"
    params: tuple = ()
    if before:
        query += " AND date < ?"
        params = (before,)
    query += " ORDER BY date"
    with connect() as conn:
        return [row["date"] for row in conn.execute(query, params)]


def to_dataframe(since: str | None = None) -> pd.DataFrame:
    init_db()
    query = "SELECT * FROM totals_recommendations"
    params: tuple = ()
    if since:
        query += " WHERE date >= ?"
        params = (since,)
    query += " ORDER BY date, game_id"
    with connect() as conn:
        return pd.read_sql_query(query, conn, params=params)
