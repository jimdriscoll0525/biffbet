"""NFL luck / regression-candidate detector — PURE (M4 of the ATS module).

Season-to-date (NO recency weighting — luck is about the season line, not
form) team indicators built from stats that are mostly random and regress:

  fumble_luck   — share of ALL fumbles in the team's games it recovered (~0.5)
  rz_luck       — red-zone TD rate (per RZ play) minus league mean, own minus
                  allowed
  to_luck       — actual turnover margin minus expected (INTs counted as-is,
                  every fumble worth 0.5 recoveries)
  win_luck      — actual win% minus EPA-implied win% (net EPA/play margin ->
                  normal cover prob, sigma = the engine's nfl_margin_sigma)

A composite z-ish score past ±`flag_threshold` marks a team `fade` (wins
outrunning EPA) or `buy`. Flags surface in reasoning and nudge the margin by
±`luck_adjust_pts` (small, clamped with everything else; set 0 to disable —
the M6 backtest decides whether it earns its keep).
"""
from __future__ import annotations

import math

import pandas as pd


def team_luck(ws: pd.DataFrame, schedules: pd.DataFrame, as_of_week: int,
              config: dict) -> pd.DataFrame:
    cfg = config.get("nfl_regression", {})
    min_games = int(cfg.get("min_games", 4))
    sigma = float(config.get("projections", {}).get("nfl_margin_sigma", 13.2))
    if ws is None or ws.empty:
        return pd.DataFrame()
    d = ws[ws["week"] < as_of_week]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("team")
    f = pd.DataFrame({
        "games": g["off_games"].sum(),
        "off_fum": g["off_fumbles"].sum(), "off_fum_lost": g["off_fumbles_lost"].sum(),
        "def_fum": g["def_fumbles"].sum(), "def_fum_lost": g["def_fumbles_lost"].sum(),
        "off_int": g["off_int"].sum(), "def_int": g["def_int"].sum(),
        "off_rz_td": g["off_rz_td"].sum(), "off_rz_plays": g["off_rz_plays"].sum(),
        "def_rz_td": g["def_rz_td"].sum(), "def_rz_plays": g["def_rz_plays"].sum(),
        "net_epa_total": g["off_all_ngt_epa"].sum() + g["off_all_gt_epa"].sum()
                         - g["def_all_ngt_epa"].sum() - g["def_all_gt_epa"].sum(),
    })
    fum_all = f["off_fum"] + f["def_fum"]
    recovered = (f["off_fum"] - f["off_fum_lost"]) + f["def_fum_lost"]
    f["fumble_luck"] = (recovered / fum_all.where(fum_all > 0)) - 0.5

    rz_rate_off = f["off_rz_td"] / f["off_rz_plays"].where(f["off_rz_plays"] > 0)
    rz_rate_def = f["def_rz_td"] / f["def_rz_plays"].where(f["def_rz_plays"] > 0)
    league_rz = (f["off_rz_td"].sum() / max(f["off_rz_plays"].sum(), 1))
    f["rz_luck"] = (rz_rate_off - league_rz).fillna(0) - (rz_rate_def - league_rz).fillna(0)

    to_actual = (f["def_int"] + f["def_fum_lost"]) - (f["off_int"] + f["off_fum_lost"])
    to_expected = (f["def_int"] + 0.5 * f["def_fum"]) - (f["off_int"] + 0.5 * f["off_fum"])
    f["to_luck"] = (to_actual - to_expected) / f["games"].clip(lower=1)

    # Wins vs EPA-implied wins (from schedules finals).
    f["wins"] = 0.0
    if schedules is not None and not schedules.empty:
        s = schedules[(schedules["week"] < as_of_week)
                      & schedules["home_score"].notna()]
        for _, r in s.iterrows():
            hw = 1.0 if r["home_score"] > r["away_score"] else (0.5 if r["home_score"] == r["away_score"] else 0.0)
            if r["home_team"] in f.index:
                f.loc[r["home_team"], "wins"] += hw
            if r["away_team"] in f.index:
                f.loc[r["away_team"], "wins"] += 1.0 - hw
    epa_margin_pg = f["net_epa_total"] / f["games"].clip(lower=1)
    f["epa_win_pct"] = epa_margin_pg.apply(
        lambda m: 0.5 * (1.0 + math.erf((m / sigma) / math.sqrt(2))) if m == m else float("nan"))
    f["win_luck"] = f["wins"] / f["games"].clip(lower=1) - f["epa_win_pct"]

    # Composite (each component roughly on its own natural scale; weights
    # config-tunable, M6 decides if the nudge earns points at all).
    w = cfg.get("weights", {"win_luck": 1.0, "to_luck": 0.5,
                            "fumble_luck": 0.5, "rz_luck": 0.5})
    f["luck_score"] = (w.get("win_luck", 1.0) * f["win_luck"].fillna(0)
                       + w.get("to_luck", 0.5) * f["to_luck"].fillna(0)
                       + w.get("fumble_luck", 0.5) * f["fumble_luck"].fillna(0)
                       + w.get("rz_luck", 0.5) * f["rz_luck"].fillna(0))
    thresh = float(cfg.get("flag_threshold", 0.18))
    f["flag"] = "none"
    eligible = f["games"] >= min_games
    f.loc[eligible & (f["luck_score"] > thresh), "flag"] = "fade"
    f.loc[eligible & (f["luck_score"] < -thresh), "flag"] = "buy"
    return f


def luck_adjustment(home: str, away: str, luck: pd.DataFrame,
                    config: dict) -> tuple[float, list[str]]:
    """Home-positive margin nudge from fade/buy flags + notes."""
    pts = float(config.get("nfl_regression", {}).get("luck_adjust_pts", 1.0))
    if luck is None or luck.empty or pts == 0:
        return 0.0, []
    total, notes = 0.0, []
    for team, sign in ((home, 1.0), (away, -1.0)):
        if team not in luck.index:
            continue
        flag = luck.loc[team, "flag"]
        if flag == "fade":
            total -= sign * pts
            notes.append(f"regression: {team} wins outrun EPA "
                         f"(luck {luck.loc[team, 'luck_score']:+.2f}) -> fade ({-sign * pts:+.1f})")
        elif flag == "buy":
            total += sign * pts
            notes.append(f"regression: {team} EPA outruns record "
                         f"(luck {luck.loc[team, 'luck_score']:+.2f}) -> buy ({sign * pts:+.1f})")
    return round(total, 2), notes
