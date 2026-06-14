"""Team rotation-average starting-pitching strength (GriffBet only).

BiffBet's TeamMetricsProvider exposes a team's *bullpen* FIP (relievers, GS/G <
0.5) but no rotation-average starter rate. GriffBet's starter-neutralized base
rate needs exactly the inverse: a team's season rotation-average run-prevention
rate (starters, GS/G >= 0.5), on the same ~4.00 xFIP scale the starter component
already uses.

This is ADDITIVE -- it reads the same shared `get_season_pitching` leaderboard
BiffBet already pulls (model-agnostic data) and never touches team_metrics.py.
It mirrors `TeamMetricsProvider.bullpen_fip()` with the starter predicate.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from mlb_value_bot.analysis.team_metrics import _match_fg_team
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.team_extras")


def _rotation_rate_column(df: pd.DataFrame) -> str | None:
    """Pick the rate column to aggregate, preferring xFIP (the starter
    component's primary stat), then SIERA, then FIP. Returns None if the frame
    has none of them."""
    for col in ("xFIP", "SIERA", "FIP"):
        if col in df.columns:
            return col
    return None


@lru_cache(maxsize=4)
def team_rotation_rates(season: int) -> dict[str, float]:
    """IP-weighted rotation-average rate per team (starters := GS/G >= 0.5).

    Same source + shape as `TeamMetricsProvider.bullpen_fip()`, predicate
    flipped to starters. Returns {canonical_team -> rate}; empty when the
    leaderboard is unavailable (FanGraphs blocked, no CSV) -- callers must
    degrade gracefully (GriffBet falls back to the un-neutralized base rate).
    Cached per season so the leaderboard is touched at most once.
    """
    # Reuse BiffBet's shared, cached leaderboard producer.
    from mlb_value_bot.analysis.pitcher_metrics import get_season_pitching

    df = get_season_pitching(season)
    out: dict[str, float] = {}
    if df is None or df.empty:
        return out
    rate_col = _rotation_rate_column(df)
    needed = {"Team", "IP", "GS", "G"}
    if rate_col is None or not needed.issubset(df.columns):
        return out

    starters = df[(df["G"] > 0) & (df["GS"] / df["G"] >= 0.5)].copy()
    starters = starters.dropna(subset=[rate_col, "IP"])
    for team_raw, grp in starters.groupby("Team"):
        team = _match_fg_team(str(team_raw))
        ip_total = grp["IP"].sum()
        if team and ip_total > 0:
            out[team] = float((grp[rate_col] * grp["IP"]).sum() / ip_total)
    if out:
        log.debug("rotation rates (%s) for %d teams via %s", season, len(out), rate_col)
    return out


def rotation_winpct_increment(team: str, season: int, config: dict) -> float | None:
    """Win% increment (vs a league-average rotation) attributable to a team's
    season rotation-average starting pitching.

    Uses the SAME conversion the starter component uses:
        increment = (league_avg_rate - team_rotation_rate) * pitcher_run_to_winpct
    so a rotation a full run/9 better than average ≈ +6.5 win% points. This is
    the amount the starter-neutralized base rate STRIPS out of the team's
    regressed win% before log5, so today's starter delta (also measured vs
    league average) is no longer partially redundant with the season-rotation
    quality already baked into the win-loss record.

    Returns None when the rotation rate is unavailable -> caller leaves that
    team's base rate un-neutralized (graceful degradation).
    """
    rates = team_rotation_rates(season)
    rate = rates.get(team)
    if rate is None:
        return None
    m = config["model"]
    lg = config["league"]
    prefer = m.get("pitcher_stat", "xfip")
    lg_avg = float(lg.get("avg_xfip", 4.0) if prefer == "xfip" else lg.get("avg_siera", 4.0))
    run_to_wp = float(m.get("pitcher_run_to_winpct", 0.065))
    return (lg_avg - rate) * run_to_wp
