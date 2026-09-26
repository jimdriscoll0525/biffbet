"""Settle pending rsi_shadow_picks against final scores.

  * MLB moneyline: finals from the MLB Stats API per date (same rules as
    tracking/results.py -- postponed / cancelled / suspended = void, a tie
    stays pending);
  * MLB totals: final runs vs the row's `line` (exact = push);
  * football: finals via football_results (_nfl_finals / _cfb_finals) and
    the PURE grade_pick; a game with no final past grading.void_after_days
    is voided.

flat_pl is 1u flat: win = decimal_odds - 1, loss = -1, push / void = 0.
Rows are patched by id; one bad row never stops the sweep.
"""
from __future__ import annotations

from datetime import date as _date
from datetime import datetime, timezone

from mlb_value_bot.utils import get_logger

log = get_logger("rsi.shadow_grade")

_SPORTS_FOR_ENGINE = {
    "mlb": ("mlb", "mlb_totals"),
    "mlb_totals": ("mlb_totals",),
    "football": ("nfl", "cfb"),
}
_VOID_STATES = {"Postponed", "Cancelled", "Canceled", "Suspended"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def flat_pl(result: str, decimal_odds: float | None) -> float | None:
    if result == "win":
        return round(float(decimal_odds) - 1.0, 4) if decimal_odds is not None else None
    if result == "loss":
        return -1.0
    if result in ("push", "void"):
        return 0.0
    return None


def _grade_ml(row: dict, game) -> str | None:
    """win/loss/void or None (still pending) for a moneyline shadow row."""
    if game is None:
        return None
    if not game.is_final:
        return "void" if game.status in _VOID_STATES else None
    winner = game.winner
    if winner is None:
        return None
    picked = game.home_team if row.get("pick_side") == "home" else game.away_team
    return "win" if winner == picked else "loss"


def _grade_total(row: dict, game) -> str | None:
    if game is None:
        return None
    if not game.is_final:
        return "void" if game.status in _VOID_STATES else None
    if game.home_score is None or game.away_score is None:
        return None
    line = row.get("line") if row.get("line") is not None else row.get("opening_line")
    if line is None:
        return None
    total = game.home_score + game.away_score
    line = float(line)
    if total == line:
        return "push"
    went_over = total > line
    side = row.get("pick_side")
    return "win" if (side == "over") == went_over else "loss"


def _patch(row: dict, result: str, counts: dict) -> None:
    from mlb_value_bot.rsi import supa

    supa.patch_rows("rsi_shadow_picks", {"id": row["id"]},
                    {"result": result, "flat_pl": flat_pl(result, row.get("decimal_odds")),
                     "updated_at": _now()})
    counts[result] = counts.get(result, 0) + 1
    counts["graded"] += 1


def _grade_mlb_rows(rows: list[dict], counts: dict, mlb_client=None) -> None:
    from mlb_value_bot.data.mlb_client import MLBClient

    mlb = mlb_client or MLBClient()
    by_date: dict[str, list[dict]] = {}
    for r in rows:
        by_date.setdefault(str(r["date"])[:10], []).append(r)
    for game_date, day_rows in sorted(by_date.items()):
        try:
            finals = {int(g.game_id): g for g in mlb.get_results(game_date)}
        except Exception as exc:  # noqa: BLE001
            log.warning("shadow grade: MLB finals for %s unavailable (%s)", game_date, exc)
            counts["pending"] += len(day_rows)
            continue
        for row in day_rows:
            try:
                game = finals.get(int(row["game_id"]))
            except (TypeError, ValueError):
                game = None
            result = (_grade_total(row, game) if row.get("market") == "total"
                      else _grade_ml(row, game))
            if result is None:
                counts["pending"] += 1
                continue
            try:
                _patch(row, result, counts)
            except Exception as exc:  # noqa: BLE001
                log.warning("shadow grade: patch failed for row %s: %s", row.get("id"), exc)
                counts["errors"] += 1


def _grade_football_rows(rows: list[dict], counts: dict, config: dict | None = None) -> None:
    from mlb_value_bot.football import load_football_config, season_for_date
    from mlb_value_bot.football.tracking import football_results as fr

    config = config or load_football_config()
    void_after = int((config.get("grading") or {}).get("void_after_days", 10))
    finals_cache: dict[tuple[str, int], dict] = {}
    for row in rows:
        league = str(row.get("league") or row.get("sport"))
        game_date = str(row["date"])[:10]
        season = season_for_date(game_date)
        key = (league, season)
        if key not in finals_cache:
            try:
                finals_cache[key] = (fr._nfl_finals if league == "nfl" else fr._cfb_finals)(season, config)
            except Exception as exc:  # noqa: BLE001
                log.warning("shadow grade: %s finals for %d unavailable (%s)", league, season, exc)
                finals_cache[key] = {}
        score = finals_cache[key].get(str(row["game_id"]))
        try:
            if score is None:
                age = (datetime.now() - datetime.fromisoformat(game_date)).days
                if age > void_after:
                    _patch(row, "void", counts)
                else:
                    counts["pending"] += 1
                continue
            hs, as_ = score
            line = row.get("line")
            result = fr.grade_pick(str(row.get("pick_side")), None if line is None else float(line),
                                   int(hs), int(as_))
            _patch(row, result, counts)
        except Exception as exc:  # noqa: BLE001
            log.warning("shadow grade: row %s failed: %s", row.get("id"), exc)
            counts["errors"] += 1


def grade_shadow(engine: str, before: str | None = None, mlb_client=None,
                 football_config: dict | None = None) -> dict:
    """Grade every pending shadow pick for `engine` ('mlb' covers both the
    moneyline and totals sports; 'football' covers nfl + cfb) dated before
    `before` (default: today). Returns {graded, win, loss, push, void,
    pending, errors, rows}."""
    from mlb_value_bot.rsi import supa

    sports = _SPORTS_FOR_ENGINE.get(engine)
    if not sports:
        raise ValueError(f"grade_shadow: unknown engine {engine!r}")
    before = before or _date.today().isoformat()
    counts = {"engine": engine, "before": before, "rows": 0, "graded": 0, "pending": 0,
              "errors": 0, "win": 0, "loss": 0, "push": 0, "void": 0}
    rows = supa.get_rows("rsi_shadow_picks",
                         {"sport": ("in", list(sports)), "result": "pending", "date": ("lt", before)},
                         order="date.asc,id.asc")
    counts["rows"] = len(rows)
    if not rows:
        return counts
    if engine == "football":
        _grade_football_rows(rows, counts, football_config)
    else:
        _grade_mlb_rows(rows, counts, mlb_client)
    log.info("Shadow grade %s (< %s): %d row(s), %d settled (%dW-%dL-%dP, %d void), %d pending",
             engine, before, counts["rows"], counts["graded"], counts["win"], counts["loss"],
             counts["push"], counts["void"], counts["pending"])
    return counts
