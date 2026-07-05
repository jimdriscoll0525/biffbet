"""Persistence for football picks — football's OWN SQLite DB (storage/
football.db, GriffBet-style isolation), one row per (league, date, game_id,
market).

Bookkeeping mirrors the totals store's proven semantics:
  * PAPER gate: every row stores `paper` (1 while betting.paper_only).
  * CLV is a PROBABILITY move in pp vs the SHARP close (lines move, so price
    ratios don't transfer): freeze the de-vigged P(pick side) at commit,
    compare to the de-vigged P(pick side) at the sharp close.
  * Committed bets are FROZEN: later runs only refresh the close + CLV.
    Analysis rows refresh fully; an analysis->bet promotion sets the opening.
  * `line` is stored from the PICKED side's perspective (away pick at home
    -3.5 stores +3.5; an under stores the total) — grading and display both
    read naturally.

Every aggregate downstream (football_performance) filters is_value=1 AND
model_tag AND league AND market — the GriffBet record-bug rule.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from mlb_value_bot.football import FOOTBALL_DB_PATH
from mlb_value_bot.utils import ensure_dirs, get_logger

log = get_logger("football.tracking.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS football_recommendations (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    league                   TEXT NOT NULL,           -- 'nfl' | 'cfb'
    date                     TEXT NOT NULL,           -- kickoff date (UTC date)
    week                     INTEGER,
    game_id                  TEXT NOT NULL,
    market                   TEXT NOT NULL,           -- 'spread' | 'total' | 'moneyline'
    home_team                TEXT NOT NULL,
    away_team                TEXT NOT NULL,
    pick_side                TEXT NOT NULL,           -- home|away|over|under
    line                     REAL,                    -- picked-side line
    bet_odds                 INTEGER NOT NULL,
    decimal_odds             REAL NOT NULL,
    model_prob               REAL NOT NULL,           -- blended P(pick side)
    market_prob_devigged     REAL NOT NULL,
    p_push                   REAL,
    ev_pct                   REAL NOT NULL,           -- raw EV
    adjusted_ev_pct          REAL,
    flat_stake               REAL NOT NULL,           -- fraction of bankroll (paper)
    confidence               REAL NOT NULL,
    tier                     TEXT,
    stability                TEXT,
    edge_score               REAL,
    archetype                TEXT,
    projected_margin         REAL,
    projected_total          REAL,
    paper                    INTEGER NOT NULL DEFAULT 1,
    model_tag                TEXT NOT NULL DEFAULT 'matchup_v1',
    reasoning_json           TEXT,
    -- CLV (probability-pp vs the sharp close) --------------------------------
    opening_line             REAL,
    opening_price            INTEGER,
    opening_devig_p_side     REAL,
    closing_line             REAL,
    closing_price            INTEGER,
    sharp_close_line         REAL,
    sharp_close_devig_p_side REAL,
    clv_pp                   REAL,
    result                   TEXT DEFAULT 'pending',  -- pending|win|loss|push|void
    home_score               INTEGER,
    away_score               INTEGER,
    profit_loss              REAL,                    -- bankroll-fraction units
    is_value                 INTEGER NOT NULL DEFAULT 0,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    UNIQUE(league, date, game_id, market)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(FOOTBALL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(_SCHEMA)


def _clv_pp(opening: float | None, closing: float | None) -> float | None:
    """Positive = the sharp close moved toward our side (we beat the close)."""
    if opening is None or closing is None:
        return None
    return round((closing - opening) * 100.0, 2)


def _picked_side_fields(pick) -> dict:
    """Line/de-vig fields oriented to the PICKED side (side A = home/over)."""
    r = pick.reasoning.get("market", {})
    side_a = pick.side in ("home", "over")
    if pick.market == "total":
        line = pick.line                      # the total is side-agnostic
    else:
        # pick.line arrives as the HOME line; store the picked side's number.
        line = pick.line if side_a else -pick.line
    devig_a = r.get("devig_p_a")
    sharp_a = r.get("sharp_devig_p_a")
    devig_side = devig_a if side_a else (1.0 - devig_a if devig_a is not None else None)
    sharp_side = sharp_a if side_a else (1.0 - sharp_a if sharp_a is not None else None)
    sharp_line = r.get("sharp_line")
    if sharp_line is not None and pick.market == "spread" and not side_a:
        sharp_line = -sharp_line
    return {"line": line, "devig_side": devig_side,
            "sharp_line": sharp_line, "sharp_devig_side": sharp_side}


def upsert_pick(analysis, pick, config: dict) -> int:
    """Insert or update one market row. See module docstring for semantics."""
    init_db()
    now = _now()
    f = _picked_side_fields(pick)
    from mlb_value_bot.analysis.ev_calculator import american_to_decimal

    decimal_odds = american_to_decimal(pick.american_odds)
    model_tag = config.get("model_tag", "matchup_v1")
    paper = 1 if config.get("betting", {}).get("paper_only", True) else 0
    m = pick.reasoning.get("matchup", {})
    proj = pick.reasoning.get("projection", {})

    with connect() as conn:
        existing = conn.execute(
            "SELECT id, is_value, opening_line, opening_price, opening_devig_p_side "
            "FROM football_recommendations WHERE league=? AND date=? AND game_id=? AND market=?",
            (analysis.league, analysis.date, analysis.game_id, pick.market),
        ).fetchone()

        if existing and bool(existing["is_value"]):
            # Frozen committed bet: refresh close + CLV only.
            clv = _clv_pp(existing["opening_devig_p_side"], f["sharp_devig_side"])
            conn.execute(
                "UPDATE football_recommendations SET closing_line=?, closing_price=?, "
                "sharp_close_line=?, sharp_close_devig_p_side=?, clv_pp=?, updated_at=? "
                "WHERE id=?",
                (f["line"], pick.american_odds, f["sharp_line"],
                 f["sharp_devig_side"], clv, now, existing["id"]),
            )
            return int(existing["id"])

        common = (
            analysis.week, analysis.home, analysis.away, pick.side, f["line"],
            pick.american_odds, decimal_odds, pick.model_prob, pick.market_prob,
            pick.p_push, pick.raw_ev, pick.adjusted_ev, pick.stake_pct,
            pick.confidence, pick.tier, pick.stability_label,
            m.get("home_edge"), m.get("archetype"),
            proj.get("margin"), proj.get("total"), paper, model_tag,
            json.dumps(pick.reasoning),
        )

        if existing:
            # Analysis row: full refresh; promotion sets the opening reference.
            now_bet = 1 if pick.is_value else 0
            open_line = f["line"] if pick.is_value else existing["opening_line"]
            open_price = pick.american_odds if pick.is_value else existing["opening_price"]
            open_devig = f["devig_side"] if pick.is_value else existing["opening_devig_p_side"]
            clv = _clv_pp(open_devig, f["sharp_devig_side"]) if pick.is_value else None
            conn.execute(
                """
                UPDATE football_recommendations SET
                    week=?, home_team=?, away_team=?, pick_side=?, line=?, bet_odds=?,
                    decimal_odds=?, model_prob=?, market_prob_devigged=?, p_push=?, ev_pct=?,
                    adjusted_ev_pct=?, flat_stake=?, confidence=?, tier=?, stability=?,
                    edge_score=?, archetype=?, projected_margin=?, projected_total=?,
                    paper=?, model_tag=?, reasoning_json=?,
                    opening_line=?, opening_price=?, opening_devig_p_side=?,
                    closing_line=?, closing_price=?, sharp_close_line=?,
                    sharp_close_devig_p_side=?, clv_pp=?, is_value=?, updated_at=?
                WHERE id=?
                """,
                common + (open_line, open_price, open_devig, f["line"], pick.american_odds,
                          f["sharp_line"], f["sharp_devig_side"], clv, now_bet, now,
                          existing["id"]),
            )
            if pick.is_value:
                log.info("Promoted %s %s %s to a paper bet (%s %s)", analysis.league,
                         analysis.game_id, pick.market, pick.side, f["line"])
            return int(existing["id"])

        clv = _clv_pp(f["devig_side"], f["sharp_devig_side"]) if pick.is_value else None
        cur = conn.execute(
            """
            INSERT INTO football_recommendations (
                league, date, game_id, market, week, home_team, away_team, pick_side, line,
                bet_odds, decimal_odds, model_prob, market_prob_devigged, p_push, ev_pct,
                adjusted_ev_pct, flat_stake, confidence, tier, stability, edge_score,
                archetype, projected_margin, projected_total, paper, model_tag,
                reasoning_json, opening_line, opening_price, opening_devig_p_side,
                closing_line, closing_price, sharp_close_line, sharp_close_devig_p_side,
                clv_pp, result, is_value, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (analysis.league, analysis.date, analysis.game_id, pick.market)
            + (analysis.week, analysis.home, analysis.away, pick.side, f["line"],
               pick.american_odds, decimal_odds, pick.model_prob, pick.market_prob,
               pick.p_push, pick.raw_ev, pick.adjusted_ev, pick.stake_pct,
               pick.confidence, pick.tier, pick.stability_label,
               m.get("home_edge"), m.get("archetype"),
               proj.get("margin"), proj.get("total"), paper, model_tag,
               json.dumps(pick.reasoning),
               f["line"] if pick.is_value else None,
               pick.american_odds if pick.is_value else None,
               f["devig_side"] if pick.is_value else None,
               f["line"], pick.american_odds, f["sharp_line"], f["sharp_devig_side"],
               clv, "pending", 1 if pick.is_value else 0, now, now),
        )
        return int(cur.lastrowid)


def save_slate(analyses: list, config: dict) -> tuple[int, int]:
    """Persist every priced market of every analyzed game. Returns
    (total rows, committed paper bets)."""
    total = n_value = 0
    for a in analyses:
        for p in a.picks:
            upsert_pick(a, p, config)
            total += 1
            n_value += 1 if p.is_value else 0
    return total, n_value


def update_result(rec_id: int, result: str, profit_loss: float | None,
                  home_score: int | None = None, away_score: int | None = None) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE football_recommendations SET result=?, profit_loss=?, home_score=?, "
            "away_score=?, updated_at=? WHERE id=?",
            (result, profit_loss, home_score, away_score, _now(), rec_id),
        )


def get_open_bets(league: str | None = None, before: str | None = None) -> list[sqlite3.Row]:
    """Pending committed bets — the grading worklist."""
    init_db()
    q = "SELECT * FROM football_recommendations WHERE result='pending' AND is_value=1"
    params: list = []
    if league:
        q += " AND league=?"
        params.append(league)
    if before:
        q += " AND date < ?"
        params.append(before)
    q += " ORDER BY date, game_id"
    with connect() as conn:
        return conn.execute(q, params).fetchall()


def to_dataframe(since: str | None = None) -> pd.DataFrame:
    init_db()
    q = "SELECT * FROM football_recommendations"
    params: tuple = ()
    if since:
        q += " WHERE date >= ?"
        params = (since,)
    q += " ORDER BY date, game_id, market"
    with connect() as conn:
        return pd.read_sql_query(q, conn, params=params)
