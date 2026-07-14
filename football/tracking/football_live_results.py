"""Grading for the live in-game totals tool (live_total_v1).

The biffbet site writes a football_live_recommendations row on every "Update
Over/Under" press that produced a lean; this module settles those rows
post-game in the daily sweep:

  * final scores from nflverse schedules, matched by (nflverse home/away
    abbr, date) — the live tables key on the ESPN event id, which nflverse
    doesn't know, so team+date is the join (with a +/-1 day tolerance for
    late kickoffs crossing the date line),
  * result vs the rec's own live_line (on the number = push, as everywhere),
  * closing_live_line = the game's LAST snapshot that had a line entered,
  * clv_pts = signed points the line moved in the lean's favor after entry
    (over: closing - entry; under: entry - closing). Deliberately a separate
    metric from the pregame model's clv_pp (probability points vs the sharp
    close) — self-captured manual closes are weaker evidence; never aggregate
    the two.

Self-healing like football_results.grade_open: every pending row is retried
each run; no final after live.grading.void_after_days -> void.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from datetime import datetime, timedelta, timezone

import pandas as pd

from mlb_value_bot.football.tracking.football_results import grade_pick
from mlb_value_bot.utils import get_logger

log = get_logger("football.tracking.live_results")

# ESPN abbreviations that differ from nflverse's (identity otherwise).
_ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}


def espn_to_nflverse(abbr: str) -> str:
    return _ESPN_TO_NFLVERSE.get(str(abbr).upper(), str(abbr).upper())


@dataclass
class LiveGradingSummary:
    graded: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    voids: int = 0
    pending: int = 0
    dates: list[str] = field(default_factory=list)


def _nfl_finals_by_teams(season: int, config: dict) -> dict[tuple[str, str, str], tuple[int, int]]:
    """(home, away, gameday) -> (home_score, away_score), nflverse abbrs."""
    from mlb_value_bot.football.data import nfl_client

    sched = nfl_client.schedules(season, config)
    finals: dict[tuple[str, str, str], tuple[int, int]] = {}
    if sched.empty:
        return finals
    date_col = "gameday" if "gameday" in sched.columns else "date"
    for _, r in sched.iterrows():
        hs, as_ = r.get("home_score"), r.get("away_score")
        if pd.notna(hs) and pd.notna(as_):
            key = (str(r["home_team"]), str(r["away_team"]), str(r[date_col])[:10])
            finals[key] = (int(hs), int(as_))
    return finals


def _find_final(finals: dict, home: str, away: str, date_iso: str) -> tuple[int, int] | None:
    """Exact-date lookup with +/-1 day tolerance (late kickoffs / timezones)."""
    d = _date.fromisoformat(date_iso[:10])
    for delta in (0, -1, 1):
        hit = finals.get((home, away, (d + timedelta(days=delta)).isoformat()))
        if hit is not None:
            return hit
    return None


def _closing_lines(snapshots: list[dict]) -> dict[str, float]:
    """game_key -> live_line of the game's last snapshot with a line entered."""
    latest: dict[str, tuple[str, float]] = {}
    for s in snapshots:
        line = s.get("live_line")
        if line is None:
            continue
        key = str(s.get("game_key"))
        ts = str(s.get("captured_at") or "")
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, float(line))
    return {k: v[1] for k, v in latest.items()}


def grade_live(config: dict, before: str | None = None) -> LiveGradingSummary:
    """Settle every pending live recommendation dated before `before`
    (default: today). Reads/patches Supabase via sync_football."""
    from mlb_value_bot.football import season_for_date
    from mlb_value_bot.football.sync_football import (
        fetch_live_table, patch_live_recommendation)

    before = before or _date.today().isoformat()
    void_after = int(config.get("live", {}).get("grading", {})
                     .get("void_after_days", 10))
    summary = LiveGradingSummary()

    rows = [r for r in fetch_live_table("football_live_recommendations")
            if r.get("result") == "pending" and str(r.get("date")) < before]
    if not rows:
        return summary
    closing = _closing_lines(fetch_live_table("football_live_snapshots"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    finals_cache: dict[int, dict] = {}
    for r in rows:
        date_iso = str(r["date"])
        if date_iso not in summary.dates:
            summary.dates.append(date_iso)
        season = season_for_date(date_iso)
        if season not in finals_cache:
            finals_cache[season] = _nfl_finals_by_teams(season, config)
        home = espn_to_nflverse(r["home_team"])
        away = espn_to_nflverse(r["away_team"])
        score = _find_final(finals_cache[season], home, away, date_iso)

        if score is None:
            age = (datetime.now() - datetime.fromisoformat(date_iso)).days
            if age > void_after:
                patch_live_recommendation(r["id"], {"result": "void", "graded_at": now})
                summary.voids += 1
                log.info("Voided live rec %s %s@%s (%d days without a final)",
                         r["id"], away, home, age)
            else:
                summary.pending += 1
            continue

        hs, as_ = score
        entry = float(r["live_line"])
        result = grade_pick(r["lean"], entry, hs, as_)
        close = closing.get(str(r["game_key"]))
        clv_pts = None
        if close is not None:
            clv_pts = (close - entry) if r["lean"] == "over" else (entry - close)
        patch_live_recommendation(r["id"], {
            "result": result, "final_total": hs + as_,
            "closing_live_line": close, "clv_pts": clv_pts, "graded_at": now,
        })
        summary.graded += 1
        if result == "win":
            summary.wins += 1
        elif result == "loss":
            summary.losses += 1
        elif result == "push":
            summary.pushes += 1
        else:
            summary.voids += 1
    return summary
