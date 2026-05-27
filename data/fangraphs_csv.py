"""Load FanGraphs leaderboard CSVs exported manually from a browser.

WHY THIS EXISTS: FanGraphs leaderboards sit behind a Cloudflare JavaScript
challenge ("Just a moment..."), which returns HTTP 403 to every plain HTTP
client (pybaseball, requests, curl) regardless of IP. A real browser solves the
challenge invisibly, so the reliable path is: export the leaderboard CSV in your
browser and drop it here. The model prefers these CSVs over live scraping.

Expected files (in `storage/fangraphs/`, configurable in config.yaml):
  * pitching_{season}.csv  — FanGraphs Pitching leaderboard, Min IP = 0/1.
        Needed columns: Name, Team, IP, G, GS, FIP and at least one of xFIP/SIERA.
        Nice to have:    K-BB%, ERA, WHIP, Stuff+.
        (Bullpen FIP is derived from relievers in THIS file: GS/G < 0.5.)
  * batting_{season}.csv   — FanGraphs Team Batting leaderboard.
        Needed columns: Team, wRC+.

The season-less names (`pitching.csv` / `batting.csv`) are accepted as a
fallback so you can just drop the current export without renaming. Headers are
taken as-is from FanGraphs; percentage strings like "18.0%" are handled.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from mlb_value_bot.utils import PACKAGE_DIR, get_logger, load_config

log = get_logger("data.fangraphs_csv")

# Numeric columns we clean (strip %/commas -> float) if present.
_PITCHING_NUMERIC = ["IP", "G", "GS", "TBF", "ERA", "FIP", "xFIP", "SIERA", "K-BB%", "WHIP", "Stuff+"]
_BATTING_NUMERIC = ["wRC+", "wOBA", "R", "PA", "OBP", "SLG", "OPS"]

# Header aliases -> canonical (what downstream code reads).
_ALIASES = {
    "playername": "Name",
    "player": "Name",
    "tm": "Team",
    "k-bb": "K-BB%",
    "wrc+": "wRC+",
    "siera%": "SIERA",
}


def _cfg() -> dict:
    return load_config().get("fangraphs_csv", {}) or {}


def csv_dir() -> Path:
    """Directory where exported CSVs live (relative to the package root)."""
    d = _cfg().get("dir", "storage/fangraphs")
    path = Path(d)
    return path if path.is_absolute() else (PACKAGE_DIR / path)


def ensure_dir() -> Path:
    d = csv_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _enabled() -> bool:
    return bool(_cfg().get("enabled", True))


def _resolve(pattern: str, season: int) -> Path | None:
    """Find the CSV for a season: `pitching_2026.csv` then bare `pitching.csv`."""
    d = csv_dir()
    if not d.exists():
        return None
    candidates = [pattern.format(season=season), pattern.replace("_{season}", "").format(season=season)]
    for name in candidates:
        p = d / name
        if p.exists():
            return p
    return None


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip())
    remap = {c: _ALIASES[c.lower()] for c in df.columns if c.lower() in _ALIASES}
    return df.rename(columns=remap) if remap else df


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = (
                df[c].astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[c] = pd.to_numeric(s, errors="coerce")
    return df


def _read(path: Path) -> pd.DataFrame:
    # FanGraphs exports are UTF-8 (sometimes with a BOM); handle both.
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        log.warning("failed reading %s: %s", path.name, exc)
        return pd.DataFrame()


@lru_cache(maxsize=8)
def load_pitching_csv(season: int) -> pd.DataFrame | None:
    """Return a normalized pitching leaderboard DataFrame, or None if absent."""
    if not _enabled():
        return None
    path = _resolve(_cfg().get("pitching_filename", "pitching_{season}.csv"), season)
    if path is None:
        return None
    df = _rename_aliases(_read(path))
    if df.empty:
        return None
    df = _coerce_numeric(df, _PITCHING_NUMERIC)

    have_rate = any(c in df.columns for c in ("xFIP", "SIERA"))
    if "Name" not in df.columns or not have_rate:
        log.warning(
            "%s loaded but missing required columns (need Name + xFIP or SIERA); "
            "found: %s", path.name, list(df.columns)[:12],
        )
        return None
    log.info("Loaded FanGraphs pitching CSV %s (%d rows)", path.name, len(df))
    return df


@lru_cache(maxsize=8)
def load_batting_csv(season: int) -> pd.DataFrame | None:
    """Return a normalized team-batting DataFrame, or None if absent."""
    if not _enabled():
        return None
    path = _resolve(_cfg().get("batting_filename", "batting_{season}.csv"), season)
    if path is None:
        return None
    df = _rename_aliases(_read(path))
    if df.empty:
        return None
    df = _coerce_numeric(df, _BATTING_NUMERIC)
    if "Team" not in df.columns or "wRC+" not in df.columns:
        log.warning(
            "%s loaded but missing required columns (need Team + wRC+); found: %s",
            path.name, list(df.columns)[:12],
        )
        return None
    log.info("Loaded FanGraphs batting CSV %s (%d rows)", path.name, len(df))
    return df


def status(season: int) -> dict:
    """Diagnostic snapshot for the CLI `data-status` command."""
    pit_path = _resolve(_cfg().get("pitching_filename", "pitching_{season}.csv"), season)
    bat_path = _resolve(_cfg().get("batting_filename", "batting_{season}.csv"), season)
    pit = load_pitching_csv(season)
    bat = load_batting_csv(season)

    def _cols(df, wanted):
        return [c for c in wanted if c in (df.columns if df is not None else [])]

    return {
        "enabled": _enabled(),
        "dir": str(csv_dir()),
        "pitching": {
            "file": pit_path.name if pit_path else None,
            "loaded": pit is not None,
            "rows": int(len(pit)) if pit is not None else 0,
            "key_cols": _cols(pit, ["Name", "Team", "IP", "G", "GS", "FIP", "xFIP", "SIERA", "K-BB%", "Stuff+"]),
        },
        "batting": {
            "file": bat_path.name if bat_path else None,
            "loaded": bat is not None,
            "rows": int(len(bat)) if bat is not None else 0,
            "key_cols": _cols(bat, ["Team", "wRC+", "wOBA"]),
        },
    }
