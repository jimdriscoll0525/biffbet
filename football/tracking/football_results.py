"""Grading for football picks — the tracking/results.py twin.

Scores: NFL from nflverse schedules (final scores land there within the
feed's game-day refresh); CFB from CFBD /games (completed flag + points).
Matched by game_id.

Settlement (PURE in grade_pick, so tests hit it directly):
  * spread: picked team's margin + picked line > 0 win, == 0 PUSH, < 0 loss.
    Exactly on the number is ALWAYS a push (stake returned, never a loss) —
    NFL/CFB key numbers (3, 7) make this the rule that matters most.
  * total: over wins above the line, under below, exactly on it = push.
  * canceled games (CFBD completed=false past the void window, or an NFL game
    with no score long past its date) -> void, stake returned.

grade_all_open() is the self-healing backfill sweep (same contract as MLB's):
every past date with pending committed bets is retried on every run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime, timedelta

import pandas as pd

from mlb_value_bot.football.tracking import football_store as store
from mlb_value_bot.utils import get_logger

log = get_logger("football.tracking.results")



@dataclass
class GradingSummary:
    league: str
    graded: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    voids: int = 0
    pending: int = 0
    profit_loss: float = 0.0
    dates: list[str] = field(default_factory=list)


def grade_pick(side: str, line: float | None, home_score: int, away_score: int) -> str:
    """PURE settlement for one pick against a final score."""
    margin = home_score - away_score
    total = home_score + away_score
    if side in ("home", "away"):
        if line is None:
            return "void"
        picked_margin = margin if side == "home" else -margin
        val = picked_margin + line
        if val > 0:
            return "win"
        if val == 0:
            return "push"
        return "loss"
    if side in ("over", "under"):
        if line is None:
            return "void"
        if total == line:
            return "push"
        if side == "over":
            return "win" if total > line else "loss"
        return "win" if total < line else "loss"
    return "void"


def profit_for(result: str, stake: float, decimal_odds: float) -> float:
    if result == "win":
        return stake * (decimal_odds - 1.0)
    if result == "loss":
        return -stake
    return 0.0    # push / void: stake returned


def _nfl_finals(season: int, config: dict) -> dict[str, tuple[int, int]]:
    from mlb_value_bot.football.data import nfl_client

    sched = nfl_client.schedules(season, config)
    finals: dict[str, tuple[int, int]] = {}
    if sched.empty or "game_id" not in sched.columns:
        return finals
    for _, r in sched.iterrows():
        hs, as_ = r.get("home_score"), r.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            finals[str(r["game_id"])] = (int(hs), int(as_))
    return finals


def _cfb_finals(season: int, config: dict) -> dict[str, tuple[int, int]]:
    from mlb_value_bot.football.data.cfbd_client import CfbdClient

    client = CfbdClient(config)
    if not client.configured:
        return {}
    finals: dict[str, tuple[int, int]] = {}
    for season_type in ("regular", "postseason"):
        games = client.games(season, season_type=season_type)
        if games.empty:
            continue
        id_col = "id" if "id" in games.columns else None
        hs_col = next((c for c in ("home_points", "homePoints") if c in games.columns), None)
        as_col = next((c for c in ("away_points", "awayPoints") if c in games.columns), None)
        if not id_col or not hs_col or not as_col:
            continue
        for _, r in games.iterrows():
            if pd.notna(r[hs_col]) and pd.notna(r[as_col]):
                finals[str(r[id_col])] = (int(r[hs_col]), int(r[as_col]))
    return finals


def grade_open(config: dict, before: str | None = None) -> list[GradingSummary]:
    """Grade every pending committed bet dated before `before` (default all
    past). Finals are fetched once per league+season."""
    from mlb_value_bot.football import season_for_date

    before = before or _date.today().isoformat()
    rows = store.get_open_bets(before=before)
    if not rows:
        return []

    finals_cache: dict[tuple[str, int], dict] = {}
    summaries: dict[str, GradingSummary] = {}
    for row in rows:
        league = row["league"]
        season = season_for_date(row["date"])
        key = (league, season)
        if key not in finals_cache:
            finals_cache[key] = (_nfl_finals if league == "nfl" else _cfb_finals)(season, config)
        finals = finals_cache[key]
        s = summaries.setdefault(league, GradingSummary(league=league))
        if row["date"] not in s.dates:
            s.dates.append(row["date"])

        void_after = int(config.get("grading", {}).get("void_after_days", 10))
        score = finals.get(str(row["game_id"]))
        if score is None:
            age = (datetime.now() - datetime.fromisoformat(row["date"])).days
            if age > void_after:
                store.update_result(row["id"], "void", 0.0)
                s.voids += 1
                log.info("Voided %s %s %s (%d days without a final)",
                         league, row["game_id"], row["market"], age)
            else:
                s.pending += 1
            continue

        hs, as_ = score
        result = grade_pick(row["pick_side"], row["line"], hs, as_)
        pl = profit_for(result, float(row["flat_stake"] or 0.0), float(row["decimal_odds"]))
        store.update_result(row["id"], result, pl, hs, as_)
        s.graded += 1
        s.profit_loss += pl
        if result == "win":
            s.wins += 1
        elif result == "loss":
            s.losses += 1
        elif result == "push":
            s.pushes += 1
        else:
            s.voids += 1
    return list(summaries.values())
