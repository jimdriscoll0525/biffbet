"""Team drive-stat priors for the live totals model — PURE aggregation.

The site's live in-game over/under tool (live_total_v1, biffbet repo) projects
expected remaining points as `remaining possessions x adjusted points-per-
drive`. Those priors are computed HERE from nflverse drive-shaped pbp and
synced daily to Supabase `football_team_drive_stats`; the site only reads.

Drive points come from the posteam score delta across the drive (captures
PAT/2pt truth without mapping fixed_drive_result strings). Kneel-only drives
and end-of-half clock-kills are excluded so the priors reflect competitive
possessions — the live model applies its own game-script layer on top.

NFL only for v1: CFB has no drive feed wired; callers get an empty frame.
"""
from __future__ import annotations

import pandas as pd

from mlb_value_bot.utils import get_logger

log = get_logger("football.analysis.drive_stats")

_STAT_COLS = [
    "games", "ppd_off", "ppd_def_allowed", "drives_pg", "plays_per_drive",
    "sec_per_play", "drive_sec_avg", "explosive_play_rate",
    "pts_per_min_trailing",
]


def _top_seconds(value) -> float:
    """'MM:SS' drive time of possession -> seconds (NaN-safe)."""
    if isinstance(value, str) and ":" in value:
        try:
            mm, ss = value.split(":", 1)
            return int(mm) * 60 + int(ss)
        except ValueError:
            return float("nan")
    return float("nan")


def nfl_drive_stats(pbp: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Per-team drive priors from drive-trimmed pbp. Index = team abbr,
    columns = the football_team_drive_stats stat columns."""
    if pbp.empty:
        return pd.DataFrame(columns=_STAT_COLS)
    cfg = config.get("live", {}).get("drive_stats", {})
    exp_pass = float(cfg.get("explosive_pass_yards", 20))
    exp_rush = float(cfg.get("explosive_rush_yards", 10))
    min_games = int(cfg.get("min_games", 1))

    df = pbp[pbp["posteam"].notna() & (pbp["posteam"] != "")].copy()
    if df.empty or "fixed_drive" not in df.columns:
        return pd.DataFrame(columns=_STAT_COLS)
    for col in ("pass", "rush", "qb_kneel", "yards_gained", "posteam_score",
                "defteam_score", "posteam_score_post", "game_seconds_remaining",
                "drive_time_of_possession"):
        if col not in df.columns:
            df[col] = pd.NA
    df["is_play"] = ((df["pass"] == 1) | (df["rush"] == 1)).astype(int)
    df["is_kneel"] = (df["qb_kneel"] == 1).astype(int)
    df["is_snap"] = ((df["is_play"] == 1) | (df["is_kneel"] == 1)).astype(int)

    # --- drive frame: one row per (game, fixed_drive) ------------------------
    gb = df.groupby(["game_id", "fixed_drive"], sort=False)
    drives = pd.DataFrame({
        "posteam": gb["posteam"].first(),
        "defteam": gb["defteam"].first(),
        # Score delta across the drive = actual points incl. PAT/2pt.
        "points": gb["posteam_score_post"].max() - gb["posteam_score"].min(),
        "plays": gb["is_play"].sum(),
        "kneels": gb["is_kneel"].sum(),
        "snaps": gb["is_snap"].sum(),
        "top_sec": gb["drive_time_of_possession"].first().map(_top_seconds),
        # Trailing at the drive's first snap (comeback-scoring input).
        "start_trailing": gb["posteam_score"].first() < gb["defteam_score"].first(),
    })
    # TOP missing on some feeds -> clock delta across the drive's snaps.
    gsr = gb["game_seconds_remaining"]
    clock_delta = (gsr.max() - gsr.min()).clip(lower=0.0)
    drives["top_sec"] = drives["top_sec"].fillna(clock_delta)
    drives["points"] = drives["points"].clip(lower=0.0)

    # Non-competitive possessions poison a PPD prior: pure kneel-out drives and
    # end-of-half clock-kills (the live model handles those situations itself).
    kneel_only = (drives["kneels"] > 0) & (drives["kneels"] == drives["snaps"])
    clock_kill = (drives["plays"] < 2) & (drives["top_sec"] < 20)
    kept = drives[~kneel_only & ~clock_kill]
    if kept.empty:
        return pd.DataFrame(columns=_STAT_COLS)

    # --- offense ---------------------------------------------------------------
    off = kept.groupby("posteam")
    out = pd.DataFrame(index=off.size().index)
    games = df.groupby("posteam")["game_id"].nunique().reindex(out.index)
    n_drives = off.size().astype(float)
    total_plays = off["plays"].sum().astype(float)
    total_sec = off["top_sec"].sum()
    out["games"] = games
    out["ppd_off"] = off["points"].sum() / n_drives
    out["drives_pg"] = n_drives / games
    out["plays_per_drive"] = total_plays / n_drives
    out["sec_per_play"] = total_sec / total_plays.replace(0.0, float("nan"))
    out["drive_sec_avg"] = off["top_sec"].mean()

    # --- defense (same drives, grouped by who was defending) -------------------
    def_ = kept.groupby("defteam")
    out["ppd_def_allowed"] = (def_["points"].sum()
                              / def_.size().astype(float)).reindex(out.index)

    # --- explosive play rate (play-level) ---------------------------------------
    plays = df[df["is_play"] == 1]
    explosive = (((plays["pass"] == 1) & (plays["yards_gained"] >= exp_pass))
                 | ((plays["rush"] == 1) & (plays["yards_gained"] >= exp_rush)))
    per_team = plays.groupby("posteam").size().astype(float)
    out["explosive_play_rate"] = (plays[explosive].groupby("posteam").size()
                                  .reindex(out.index, fill_value=0)
                                  / per_team.reindex(out.index))

    # --- comeback scoring rate: points per trailing possession minute ----------
    trailing = kept[kept["start_trailing"]].groupby("posteam")
    trail_min = trailing["top_sec"].sum() / 60.0
    out["pts_per_min_trailing"] = (trailing["points"].sum()
                                   / trail_min.replace(0.0, float("nan"))
                                   ).reindex(out.index)

    out = out[out["games"] >= min_games][["games"] + _STAT_COLS[1:]]
    out.index.name = "team"
    return out
