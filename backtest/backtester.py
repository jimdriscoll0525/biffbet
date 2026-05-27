"""Re-run the model over historical games.

Historical odds from The Odds API require a PAID plan, so this is built around a
CSV fallback (and the scaffolding makes swapping in a real historical-odds
source later a one-function change). Point it at a CSV of past lines and it will,
for each game: pull that day's schedule + probable pitchers, run the exact same
model the live `today` command uses, grade against the real final score, and
report ROI / hit rate / CLV.

Expected CSV columns (header row, names case-insensitive):
    date, home_team, away_team, home_odds, away_odds
Optional:
    home_closing, away_closing   # to measure CLV in the backtest

KNOWN LIMITATION (documented in README): the FanGraphs season-stat path uses
full-season numbers, which leaks future info into a historical backtest
(look-ahead bias). Statcast pulls are correctly windowed to <= the game date.
Treat backtest ROI as optimistic until an as-of-date stat source is wired in;
CLV (if closing lines are supplied) is unaffected by this bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlb_value_bot.analysis.ev_calculator import american_to_decimal
from mlb_value_bot.analysis.team_metrics import TeamMetricsProvider
from mlb_value_bot.data.mlb_client import MLBClient
from mlb_value_bot.data.odds_client import GameOdds, SidePrice
from mlb_value_bot.pipeline import evaluate_game
from mlb_value_bot.constants import normalize_team
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("backtest.backtester")


@dataclass
class BacktestResult:
    bets: pd.DataFrame
    summary: dict


def _read_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "home_team", "away_team", "home_odds", "away_odds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return df


def run_backtest(
    start: str,
    end: str,
    csv_path: str,
    config: dict | None = None,
    mlb_client: MLBClient | None = None,
) -> BacktestResult:
    """Run the model over CSV-provided historical odds in [start, end]."""
    config = config or load_config()
    mlb = mlb_client or MLBClient(config)
    threshold = float(config["ev"]["threshold"])

    df = _read_csv(csv_path)
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    log.info("Backtest %s..%s: %d candidate games from CSV", start, end, len(df))

    providers: dict[int, TeamMetricsProvider] = {}
    schedule_cache: dict[str, list] = {}
    results_cache: dict[str, dict] = {}

    rows: list[dict] = []
    for _, r in df.iterrows():
        game_date = str(r["date"])[:10]
        home = normalize_team(str(r["home_team"]))
        away = normalize_team(str(r["away_team"]))
        season = int(game_date[:4])

        if game_date not in schedule_cache:
            try:
                schedule_cache[game_date] = mlb.get_schedule(game_date)
            except Exception as exc:
                log.warning("schedule fetch failed for %s: %s", game_date, exc)
                schedule_cache[game_date] = []
        sched = next(
            (s for s in schedule_cache[game_date]
             if {s.home_team, s.away_team} == {home, away}),
            None,
        )
        if sched is None:
            log.debug("No scheduled game matched %s @ %s on %s", away, home, game_date)
            continue

        if season not in providers:
            providers[season] = TeamMetricsProvider(season=season, config=config, mlb_client=mlb)

        game_odds = GameOdds(
            event_id="", commence_time="", home_team=sched.home_team, away_team=sched.away_team,
            home=SidePrice(sched.home_team, int(r["home_odds"]), "csv"),
            away=SidePrice(sched.away_team, int(r["away_odds"]), "csv"),
        )
        from datetime import date as _date
        analysis = evaluate_game(
            sched, game_odds, providers[season], season, _date.fromisoformat(game_date), config
        )
        be = analysis.best_eval
        if be is None or not analysis.is_value(threshold):
            continue

        # Grade against the real final score.
        if game_date not in results_cache:
            try:
                results_cache[game_date] = {g.game_id: g for g in mlb.get_results(game_date)}
            except Exception:
                results_cache[game_date] = {}
        result_game = results_cache[game_date].get(sched.game_id)
        if result_game is None or result_game.winner is None:
            continue

        bet_team = sched.home_team if analysis.best_side == "home" else sched.away_team
        won = result_game.winner == bet_team
        dec = american_to_decimal(be.american_odds)
        kelly_pl = be.kelly_stake * (dec - 1.0) if won else -be.kelly_stake
        flat_pl = (dec - 1.0) if won else -1.0

        # Optional CLV from supplied closing lines.
        clv = np.nan
        close_col = "home_closing" if analysis.best_side == "home" else "away_closing"
        if close_col in df.columns and pd.notna(r.get(close_col)):
            try:
                clv = (dec / american_to_decimal(int(r[close_col])) - 1.0) * 100.0
            except (ValueError, ZeroDivisionError):
                clv = np.nan

        rows.append({
            "date": game_date,
            "matchup": f"{away} @ {home}",
            "side": analysis.best_side,
            "team": bet_team,
            "american_odds": be.american_odds,
            "model_prob": round(be.model_prob, 4),
            "market_prob_devigged": round(be.market_prob_devigged, 4),
            "ev_pct": round(be.ev_pct * 100, 2),
            "confidence": analysis.confidence,
            "kelly_stake": round(be.kelly_stake, 4),
            "result": "win" if won else "loss",
            "kelly_pl": kelly_pl,
            "flat_pl": flat_pl,
            "clv_pct": clv,
        })

    bets = pd.DataFrame(rows)
    summary = _summarize(bets)
    return BacktestResult(bets=bets, summary=summary)


def _summarize(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"bets": 0}
    n = len(bets)
    wins = int((bets["result"] == "win").sum())
    staked = float(bets["kelly_stake"].sum())
    kelly_pl = float(bets["kelly_pl"].sum())
    flat_pl = float(bets["flat_pl"].sum())
    clv = bets["clv_pct"].dropna()
    return {
        "bets": n,
        "wins": wins,
        "losses": n - wins,
        "hit_rate": wins / n,
        "avg_ev_pct": float(bets["ev_pct"].mean()),
        "kelly_pl_units": kelly_pl,
        "kelly_roi": (kelly_pl / staked) if staked > 0 else float("nan"),
        "flat_pl_units": flat_pl,
        "flat_roi": flat_pl / n,
        "avg_clv_pct": float(clv.mean()) if len(clv) else float("nan"),
        "clv_tracked": int(len(clv)),
    }
