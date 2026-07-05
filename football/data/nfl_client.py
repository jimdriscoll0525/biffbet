"""nflverse data via nflreadpy (free, no key). NFL's only stat/schedule source.

Responsibilities (I/O ONLY — aggregation into unit stats is analysis-layer):
  * play-by-play (the EPA source of truth), trimmed to the columns the
    matchup model reads before converting Polars -> pandas (pbp is ~370 cols),
  * schedules: kickoff datetime, roof/surface, and FINAL SCORES (grading),
  * snap counts: the OL-continuity proxy input,
  * FTN charting: blitz / pass-rusher counts for the pressure PROXY
    (real pressure data isn't free in-season; nflverse injuries died after
    2024, which is why continuity comes from snap deltas, not reports).

Everything routes through cache.cached_dataframe, so a nflverse outage
degrades to stale/empty frames instead of killing the slate.
"""
from __future__ import annotations

import pandas as pd

from mlb_value_bot.data.cache import cached_dataframe
from mlb_value_bot.utils import get_logger

log = get_logger("football.data.nfl")

# The pbp columns the matchup model actually reads. Trimming BEFORE to_pandas
# keeps the cached parquet and memory footprint sane.
_PBP_COLS = [
    "game_id", "season", "week", "posteam", "defteam", "season_type",
    "pass", "rush", "qb_dropback", "pass_attempt", "rush_attempt",
    "sack", "qb_hit", "interception", "fumble_lost",
    "pass_touchdown", "rush_touchdown", "touchdown",
    "yards_gained", "epa", "success", "qtr", "yardline_100",
]


def _ttl_hours(config: dict, key: str, default: float) -> float:
    return float(config.get("nfl_data", {}).get(key, default)) * 3600.0


def _load(loader_name: str, season: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Call a nflreadpy loader inside the producer (import deferred so tests
    never need nflreadpy installed)."""
    import nflreadpy

    df = getattr(nflreadpy, loader_name)(season)
    if columns:
        have = [c for c in columns if c in df.columns]
        missing = sorted(set(columns) - set(have))
        if missing:
            log.warning("nflverse %s missing columns %s", loader_name, missing)
        df = df.select(have)
    return df.to_pandas()


def pbp(season: int, config: dict, force_refresh: bool = False) -> pd.DataFrame:
    """Regular-season play-by-play, trimmed to the model's columns."""
    df = cached_dataframe(
        f"nfl_pbp_{season}",
        lambda: _load("load_pbp", season, _PBP_COLS),
        ttl_seconds=_ttl_hours(config, "pbp_ttl_hours", 12),
        force_refresh=force_refresh,
    )
    if not df.empty and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    return df


def schedules(season: int, config: dict, force_refresh: bool = False) -> pd.DataFrame:
    """Season schedule: kickoff, roof, surface, and final scores when played."""
    return cached_dataframe(
        f"nfl_schedules_{season}",
        lambda: _load("load_schedules", season),
        ttl_seconds=_ttl_hours(config, "schedules_ttl_hours", 3),
        force_refresh=force_refresh,
    )


def snap_counts(season: int, config: dict, force_refresh: bool = False) -> pd.DataFrame:
    return cached_dataframe(
        f"nfl_snap_counts_{season}",
        lambda: _load("load_snap_counts", season),
        ttl_seconds=_ttl_hours(config, "snap_counts_ttl_hours", 12),
        force_refresh=force_refresh,
    )


def ftn_charting(season: int, config: dict, force_refresh: bool = False) -> pd.DataFrame:
    return cached_dataframe(
        f"nfl_ftn_{season}",
        lambda: _load("load_ftn_charting", season),
        ttl_seconds=_ttl_hours(config, "ftn_ttl_hours", 12),
        force_refresh=force_refresh,
    )
