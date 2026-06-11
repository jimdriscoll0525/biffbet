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
    is_value              INTEGER NOT NULL DEFAULT 1,  -- 1 = actual bet (>= EV threshold);
                                                       -- 0 = analyzed but didn't clear threshold,
                                                       -- kept so the site can show full slates.
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    UNIQUE(date, game_id, recommended_side)
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns / indexes introduced after the original schema. SQLite can
    only ADD COLUMN via ALTER (no DROP/RENAME), so each migration is an
    idempotent ADD; constraint-shaped migrations use CREATE UNIQUE INDEX IF
    NOT EXISTS, which acts as a unique constraint for ON CONFLICT purposes.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(recommendations)")}
    if "is_value" not in cols:
        # Existing rows predate this column and were ALL +EV (only +EV picks were
        # saved by the old save_value_bets); default to 1 backfills them correctly.
        conn.execute("ALTER TABLE recommendations ADD COLUMN is_value INTEGER NOT NULL DEFAULT 1")
        log.info("Migrated recommendations: added is_value column (existing rows -> 1).")

    # 2026-05-28: dedupe + tighten unique key to (date, game_id). One row per
    # game per date. The original schema's UNIQUE(date, game_id, side) let the
    # sync push create duplicate rows when the engine's best side flipped
    # between runs. We dedupe by keeping the most authoritative row per game
    # (bets before analyses; most-recently updated within a tier), then add
    # the new unique index. The OLD UNIQUE constraint stays in the CREATE TABLE
    # definition but is now a strict superset of the new one (never violated).
    indexes = {row["name"] for row in conn.execute("PRAGMA index_list(recommendations)")}
    if "recommendations_date_game_uidx" not in indexes:
        conn.execute(
            """
            DELETE FROM recommendations
            WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY date, game_id
                        ORDER BY is_value DESC, updated_at DESC, id DESC
                    ) AS rn FROM recommendations
                ) WHERE rn > 1
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS recommendations_date_game_uidx "
            "ON recommendations(date, game_id)"
        )
        log.info("Migrated recommendations: deduped + added (date, game_id) unique index.")


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
        _migrate(conn)
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
    """Insert a new recommendation, or update it if (date, game_id) already exists.

    Semantics depend on is_value (the actual bet vs. an analysis breadcrumb):

    * Existing is_value=1 (already a committed bet): we NEVER downgrade. Even if
      the new analysis no longer clears EV, the bet stands. We only refresh
      closing_line + CLV.
    * Existing is_value=0, new is_value=1: an analysis-only row has been
      promoted to a real bet on this run. Reset opening_line to the bet price,
      flip is_value, and refresh model fields (this is the first "real" snapshot
      of this bet).
    * Existing is_value=0, new is_value=0: still just an analysis. Refresh ALL
      model fields with the latest run; opening_line/closing_line track the
      latest prices.

    We match by (date, game_id) — at most one row per game per date — so if the
    favored side flips between runs while still a non-bet analysis, we update
    the side in place rather than creating a duplicate row.
    """
    init_db()
    now = _now()
    with connect() as conn:
        existing = conn.execute(
            """
            SELECT id, opening_line, is_value FROM recommendations
            WHERE date=? AND game_id=?
            """,
            (rec.date, rec.game_id),
        ).fetchone()

        if existing:
            was_bet = bool(existing["is_value"])
            now_bet = bool(rec.is_value)

            if was_bet:
                # Already a committed bet -> keep opening_line, only update close + CLV.
                opening = existing["opening_line"]
                closing = rec.american_odds
                clv = _compute_clv(opening, closing)
                conn.execute(
                    "UPDATE recommendations SET closing_line=?, clv_pct=?, updated_at=? WHERE id=?",
                    (closing, clv, now, existing["id"]),
                )
                log.info("Refreshed closing line for game %s (%s): %s (CLV %.2f%%)",
                         rec.game_id, rec.recommended_side, closing,
                         clv if clv is not None else 0.0)
                return int(existing["id"])

            # Was a non-bet analysis: refresh everything. If it's now a bet, this
            # run is the first "real" snapshot -> the current price becomes
            # opening_line and is_value flips to 1.
            opening = rec.american_odds if now_bet else (existing["opening_line"] or rec.american_odds)
            closing = rec.american_odds
            clv = _compute_clv(opening, closing) if now_bet else None
            conn.execute(
                """
                UPDATE recommendations SET
                    home_team=?, away_team=?, recommended_side=?,
                    model_prob=?, market_prob_devigged=?,
                    american_odds=?, decimal_odds=?, ev_pct=?, kelly_stake=?,
                    confidence=?, reasoning_json=?,
                    opening_line=?, closing_line=?, clv_pct=?, is_value=?, updated_at=?
                WHERE id=?
                """,
                (
                    rec.home_team, rec.away_team, rec.recommended_side,
                    rec.model_prob, rec.market_prob_devigged,
                    rec.american_odds, rec.decimal_odds, rec.ev_pct, rec.kelly_stake,
                    rec.confidence, json.dumps(rec.reasoning),
                    opening, closing, clv, 1 if now_bet else 0, now,
                    existing["id"],
                ),
            )
            if now_bet:
                log.info("Promoted game %s (%s) to a bet @ %+d (EV %.1f%%)",
                         rec.game_id, rec.recommended_side, rec.american_odds, rec.ev_pct * 100)
            return int(existing["id"])

        opening = rec.opening_line if rec.opening_line is not None else rec.american_odds
        cur = conn.execute(
            """
            INSERT INTO recommendations (
                date, game_id, home_team, away_team, recommended_side,
                model_prob, market_prob_devigged, american_odds, decimal_odds,
                ev_pct, kelly_stake, confidence, reasoning_json,
                opening_line, closing_line, clv_pct, result, profit_loss, is_value,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.date, rec.game_id, rec.home_team, rec.away_team, rec.recommended_side,
                rec.model_prob, rec.market_prob_devigged, rec.american_odds, rec.decimal_odds,
                rec.ev_pct, rec.kelly_stake, rec.confidence, json.dumps(rec.reasoning),
                opening, rec.closing_line, rec.clv_pct, rec.result, rec.profit_loss,
                1 if rec.is_value else 0,
                now, now,
            ),
        )
        tag = "bet" if rec.is_value else "analysis"
        log.info("Saved %s: %s %s @ %+d (EV %.1f%%)",
                 tag, rec.recommended_side,
                 rec.home_team if rec.recommended_side == "home" else rec.away_team,
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
    """Pending BET recommendations for a given game date (is_value=1 only).

    Non-value analysis rows aren't bets; they don't get graded and don't count
    toward W/L or P&L.
    """
    init_db()
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM recommendations WHERE date=? AND result='pending' AND is_value=1",
            (game_date,),
        ).fetchall()


def refresh_closing_line(game_date: str, game_id: int, side_odds: dict[str, int]) -> bool:
    """Update closing_line/clv_pct on a committed bet from current prices.

    Mirrors the committed-bet branch of `upsert_recommendation` (opening line
    kept, only close + CLV move) for games the pipeline did NOT save this run
    -- i.e. sanity-skipped games, whose committed bets otherwise freeze their
    CLV at the last non-skipped run. `side_odds` maps "home"/"away" to the
    bet-book American price. Returns True if a bet was updated.
    """
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT id, recommended_side, opening_line FROM recommendations "
            "WHERE date=? AND game_id=? AND is_value=1",
            (game_date, game_id),
        ).fetchone()
        if row is None:
            return False
        closing = side_odds.get(row["recommended_side"])
        if closing is None:
            return False
        clv = _compute_clv(row["opening_line"], int(closing))
        conn.execute(
            "UPDATE recommendations SET closing_line=?, clv_pct=?, updated_at=? WHERE id=?",
            (int(closing), clv, _now(), row["id"]),
        )
        log.info(
            "Refreshed closing line for skipped game %s (%s): %+d (CLV %.2f%%)",
            game_id, row["recommended_side"], int(closing), clv if clv is not None else 0.0,
        )
        return True


def get_committed_bet(game_date: str, game_id: int) -> sqlite3.Row | None:
    """The committed bet row (is_value=1) for a game, or None."""
    init_db()
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM recommendations WHERE date=? AND game_id=? AND is_value=1",
            (game_date, game_id),
        ).fetchone()


def add_scratch_alerts(game_date: str, game_id: int, alerts: list[dict]) -> int:
    """Merge starter-change alerts into a committed bet's reasoning_json.

    Committed bets deliberately freeze their reasoning at commit time (the
    upsert never rewrites it), so this is the ONE sanctioned mutation: an
    append-only `scratch_alerts` list recording that the probable starter the
    bet was priced on has changed. Deduped on (side, was, now) so the every-30m
    pipeline doesn't stack copies. Returns the number of alerts actually added.
    """
    if not alerts:
        return 0
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT id, reasoning_json FROM recommendations WHERE date=? AND game_id=? AND is_value=1",
            (game_date, game_id),
        ).fetchone()
        if row is None:
            return 0
        try:
            reasoning = json.loads(row["reasoning_json"] or "{}")
        except json.JSONDecodeError:
            reasoning = {}
        existing = reasoning.get("scratch_alerts") or []
        seen = {(a.get("side"), a.get("was"), a.get("now")) for a in existing}
        added = 0
        for alert in alerts:
            key = (alert.get("side"), alert.get("was"), alert.get("now"))
            if key in seen:
                continue
            seen.add(key)
            existing.append({**alert, "detected_at": _now()})
            added += 1
        if not added:
            return 0
        reasoning["scratch_alerts"] = existing
        conn.execute(
            "UPDATE recommendations SET reasoning_json=?, updated_at=? WHERE id=?",
            (json.dumps(reasoning), _now(), row["id"]),
        )
        return added


def get_open_dates(before: str | None = None) -> list[str]:
    """Distinct game dates that still have pending bets (is_value=1), ascending.

    `before` (YYYY-MM-DD, exclusive) restricts to past dates so an in-progress
    slate isn't swept; None returns every open date. Used by the grading
    backfill: a bet can outlive the old "grade yesterday" default (a failed
    run, a game still in progress when graded, rows created before grading
    existed), so `results` sweeps these instead of assuming yesterday.
    """
    init_db()
    query = "SELECT DISTINCT date FROM recommendations WHERE result='pending' AND is_value=1"
    params: tuple = ()
    if before:
        query += " AND date < ?"
        params = (before,)
    query += " ORDER BY date"
    with connect() as conn:
        return [row["date"] for row in conn.execute(query, params)]


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
