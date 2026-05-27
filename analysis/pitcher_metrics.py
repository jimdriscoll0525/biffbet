"""Pull and normalize starting-pitcher metrics from pybaseball.

Two sources are combined:
  * FanGraphs season stats (`pitching_stats`) -> xFIP, SIERA, K-BB%, IP, etc.
    These are the rate stats the win-probability model consumes.
  * Statcast (`statcast_pitcher`) -> Whiff%, CSW%, HardHit%, xwOBA-on-contact,
    plus a recent (last-30-day / last-5-start) rolling form snapshot. These feed
    the confidence score and the recent-form model component.

IMPORTANT (per spec): we only surface metrics pybaseball actually returns. If a
metric isn't present (e.g. Stuff+ on some installs), it's left as None and noted
rather than fabricated. Every pull is cached to parquet with a 24h TTL.

All functions degrade gracefully: if pybaseball is missing or a pull fails, the
returned profile simply has lower data completeness, which the model/confidence
layer accounts for.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from mlb_value_bot.data.cache import cached_dataframe
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("analysis.pitcher_metrics")

try:  # pybaseball is optional at import time; the bot still runs without it.
    import pybaseball  # type: ignore

    _HAVE_PYBASEBALL = True
except Exception as _exc:  # pragma: no cover - environment dependent
    pybaseball = None  # type: ignore
    _HAVE_PYBASEBALL = False
    log.warning("pybaseball unavailable (%s); pitcher metrics will be empty", _exc)

# Statcast `description` buckets used to derive plate-discipline rates.
_WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
_SWING_DESCRIPTIONS = _WHIFF_DESCRIPTIONS | {
    "foul",
    "foul_tip",
    "hit_into_play",
    "foul_bunt",
}
_CSW_DESCRIPTIONS = _WHIFF_DESCRIPTIONS | {"called_strike"}


@dataclass
class PitcherProfile:
    """Normalized snapshot of one starting pitcher."""

    player_id: int | None
    name: str | None

    # FanGraphs season rate stats
    ip: float | None = None
    games_started: float | None = None
    xfip: float | None = None
    siera: float | None = None
    fip: float | None = None
    era: float | None = None
    k_bb_pct: float | None = None
    whip: float | None = None
    stuff_plus: float | None = None  # only if the install/leaderboard exposes it

    # Statcast (season-to-date)
    whiff_pct: float | None = None
    csw_pct: float | None = None
    hardhit_pct: float | None = None
    xwoba_con: float | None = None  # xwOBA on contact (proxy for contact quality)
    xwoba_against: float | None = None  # full PA-weighted xwOBA against
    statcast_rate: float | None = None  # runs/9 derived from xwoba_against (xFIP fallback)
    ip_is_estimated: bool = False       # True if IP was inferred from pitch count

    # Recent form
    recent_csw_pct: float | None = None
    recent_xwoba_con: float | None = None
    recent_starts: int = 0

    has_season_stats: bool = False
    has_statcast: bool = False

    def primary_rate(self, prefer: str = "xfip") -> float | None:
        """Preferred run-prevention rate stat with a sensible fallback chain.

        Returns xFIP (or SIERA) if available, else the other, then the
        Statcast-derived `statcast_rate` (which keeps the starter component alive
        when FanGraphs is unavailable), then FIP/ERA.
        """
        base = ["xfip", "siera"] if prefer == "xfip" else ["siera", "xfip"]
        order = base + ["statcast_rate", "fip", "era"]
        for attr in order:
            val = getattr(self, attr)
            if val is not None:
                return float(val)
        return None

    def primary_rate_source(self, prefer: str = "xfip") -> str | None:
        """Which stat `primary_rate` resolved to (for transparent labeling)."""
        base = ["xfip", "siera"] if prefer == "xfip" else ["siera", "xfip"]
        for attr in base + ["statcast_rate", "fip", "era"]:
            if getattr(self, attr) is not None:
                return attr
        return None

    @property
    def data_completeness(self) -> float:
        """Fraction in [0,1] of the key fields we successfully populated."""
        checks = [
            self.primary_rate() is not None,
            self.k_bb_pct is not None,
            self.csw_pct is not None,
            self.recent_starts > 0,
        ]
        return sum(checks) / len(checks)


# --- FanGraphs season stats --------------------------------------------------
def _normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation for robust name matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return "".join(ch for ch in ascii_only.lower() if ch.isalnum() or ch == " ").strip()


@lru_cache(maxsize=4)
def get_season_pitching(season: int) -> pd.DataFrame:
    """FanGraphs season pitching leaderboard (qual=0 -> everyone), cached 24h.

    Memoized per-process so a FanGraphs outage (currently a recurring 403 from
    Cloudflare blocking) is attempted at most once per season per run, not once
    per pitcher. cached_dataframe returns an empty frame on failure, so callers
    just see "no FanGraphs data" and degrade to the Statcast-derived rating.
    """
    # Prefer a manually exported FanGraphs CSV (Cloudflare blocks live scraping).
    from mlb_value_bot.data.fangraphs_csv import load_pitching_csv

    csv = load_pitching_csv(season)
    if csv is not None and not csv.empty:
        return csv

    if not _HAVE_PYBASEBALL:
        return pd.DataFrame()

    def _producer() -> pd.DataFrame:
        # qual=0 ensures we get pitchers with few innings too (early season).
        df = pybaseball.pitching_stats(season, season, qual=0)  # type: ignore[union-attr]
        return df

    return cached_dataframe(f"fg_pitching_{season}", _producer)


def _season_row_for(df: pd.DataFrame, name: str | None, player_id: int | None) -> pd.Series | None:
    if df.empty or not name:
        return None
    if "Name" not in df.columns:
        return None
    target = _normalize_name(name)
    norm = df["Name"].astype(str).map(_normalize_name)
    matches = df[norm == target]
    if matches.empty:
        return None
    if len(matches) > 1:
        # Prefer the one with the most innings pitched (the real starter).
        ip_col = "IP" if "IP" in matches.columns else None
        if ip_col:
            matches = matches.sort_values(ip_col, ascending=False)
    return matches.iloc[0]


def _coerce(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# --- Statcast ----------------------------------------------------------------
def get_pitcher_statcast(player_id: int, start: str, end: str) -> pd.DataFrame:
    """Cached statcast pull for one pitcher over [start, end] (YYYY-MM-DD)."""
    if not _HAVE_PYBASEBALL or not player_id:
        return pd.DataFrame()

    def _producer() -> pd.DataFrame:
        return pybaseball.statcast_pitcher(start, end, int(player_id))  # type: ignore[union-attr]

    return cached_dataframe(f"statcast_pitcher_{player_id}_{start}_{end}", _producer)


def _rate(numer_mask: pd.Series, denom_mask: pd.Series) -> float | None:
    denom = int(denom_mask.sum())
    if denom == 0:
        return None
    return float(numer_mask.sum()) / denom


def _statcast_rates(df: pd.DataFrame) -> dict[str, float | None]:
    """Compute Whiff%, CSW%, HardHit%, xwOBA-on-contact from pitch-level data."""
    if df.empty or "description" not in df.columns:
        return {"whiff_pct": None, "csw_pct": None, "hardhit_pct": None, "xwoba_con": None}

    desc = df["description"].astype(str)
    total_pitches = len(df)

    csw = desc.isin(_CSW_DESCRIPTIONS)
    whiff = desc.isin(_WHIFF_DESCRIPTIONS)
    swing = desc.isin(_SWING_DESCRIPTIONS)

    csw_pct = float(csw.sum()) / total_pitches if total_pitches else None
    whiff_pct = _rate(whiff, swing)

    # Batted-ball based metrics.
    bip = desc == "hit_into_play"
    hardhit_pct = None
    xwoba_con = None
    if "launch_speed" in df.columns:
        ev = pd.to_numeric(df.loc[bip, "launch_speed"], errors="coerce").dropna()
        if len(ev) > 0:
            hardhit_pct = float((ev >= 95.0).sum()) / len(ev)
    if "estimated_woba_using_speedangle" in df.columns:
        xw = pd.to_numeric(
            df.loc[bip, "estimated_woba_using_speedangle"], errors="coerce"
        ).dropna()
        if len(xw) > 0:
            xwoba_con = float(xw.mean())

    return {
        "whiff_pct": whiff_pct,
        "csw_pct": csw_pct,
        "hardhit_pct": hardhit_pct,
        "xwoba_con": xwoba_con,
    }


def _xwoba_against(df: pd.DataFrame) -> tuple[float | None, int]:
    """Full PA-weighted xwOBA against, plus the PA count, from pitch-level data.

    Standard xwOBA construction: per plate appearance use the estimated wOBA on
    balls in play (`estimated_woba_using_speedangle`) and the actual `woba_value`
    for non-contact outcomes (K=0, BB/HBP weighted). Numerator / sum(woba_denom).
    Returns (xwoba_against, plate_appearances). This is independent of FanGraphs.
    """
    if df.empty or "woba_denom" not in df.columns or "woba_value" not in df.columns:
        return None, 0
    denom = pd.to_numeric(df["woba_denom"], errors="coerce")
    wv = pd.to_numeric(df["woba_value"], errors="coerce")
    est = pd.to_numeric(df.get("estimated_woba_using_speedangle"), errors="coerce") \
        if "estimated_woba_using_speedangle" in df.columns else pd.Series([float("nan")] * len(df))
    num = est.where(est.notna(), wv).fillna(0.0)

    mask = denom.fillna(0) > 0
    pa = int(denom[mask].sum())
    total_denom = float(denom[mask].sum())
    if total_denom <= 0:
        return None, 0
    return float(num[mask].sum() / total_denom), pa


def _recent_form(df: pd.DataFrame, last_n_starts: int = 5) -> dict[str, float | int | None]:
    """CSW% / xwOBA-on-contact over the pitcher's most recent N start dates."""
    if df.empty or "game_date" not in df.columns:
        return {"recent_csw_pct": None, "recent_xwoba_con": None, "recent_starts": 0}
    dates = sorted(pd.to_datetime(df["game_date"], errors="coerce").dropna().dt.date.unique())
    recent_dates = set(dates[-last_n_starts:])
    recent = df[pd.to_datetime(df["game_date"], errors="coerce").dt.date.isin(recent_dates)]
    rates = _statcast_rates(recent)
    return {
        "recent_csw_pct": rates["csw_pct"],
        "recent_xwoba_con": rates["xwoba_con"],
        "recent_starts": len(recent_dates),
    }


# --- Profile builder ---------------------------------------------------------
def build_pitcher_profile(
    player_id: int | None,
    name: str | None,
    season: int,
    as_of: date | None = None,
    recent_window_days: int = 30,
) -> PitcherProfile:
    """Assemble a PitcherProfile from FanGraphs season + Statcast data."""
    profile = PitcherProfile(player_id=player_id, name=name)
    as_of = as_of or date.today()

    # 1) FanGraphs season rate stats (matched by name).
    season_df = get_season_pitching(season)
    row = _season_row_for(season_df, name, player_id)
    if row is not None:
        profile.has_season_stats = True
        profile.ip = _coerce(row.get("IP"))
        profile.games_started = _coerce(row.get("GS"))
        profile.xfip = _coerce(row.get("xFIP"))
        profile.siera = _coerce(row.get("SIERA"))
        profile.fip = _coerce(row.get("FIP"))
        profile.era = _coerce(row.get("ERA"))
        profile.whip = _coerce(row.get("WHIP"))
        # K-BB% may be a fraction (0.18) or a percent (18.0) depending on source.
        kbb = _coerce(row.get("K-BB%"))
        if kbb is not None:
            profile.k_bb_pct = kbb / 100.0 if kbb > 1.5 else kbb
        # Stuff+ is only present on some FanGraphs pulls; include if found.
        if "Stuff+" in season_df.columns:
            profile.stuff_plus = _coerce(row.get("Stuff+"))

    # 2) Statcast season-to-date + recent form.
    if player_id:
        season_start = date(season, 3, 1)
        sc_full = get_pitcher_statcast(player_id, season_start.isoformat(), as_of.isoformat())
        if not sc_full.empty:
            profile.has_statcast = True
            rates = _statcast_rates(sc_full)
            profile.whiff_pct = rates["whiff_pct"]
            profile.csw_pct = rates["csw_pct"]
            profile.hardhit_pct = rates["hardhit_pct"]
            profile.xwoba_con = rates["xwoba_con"]

            # Full xwOBA-against -> a runs/9 rate on the same scale as xFIP, used
            # as the starter-component input when FanGraphs is unavailable.
            # The observed xwOBA is regressed toward league average by batters
            # faced (PA) so small samples don't read as ace/replacement level.
            xwoba, pa = _xwoba_against(sc_full)
            if xwoba is not None:
                cfg = load_config()
                lg = cfg["league"]
                avg_xw = float(lg["avg_xwoba"])
                k = float(cfg["model"].get("statcast_regression_pa", 200))
                reg_xwoba = (xwoba * pa + avg_xw * k) / (pa + k) if pa > 0 else avg_xw
                rate = float(lg["avg_xfip"]) + (reg_xwoba - avg_xw) * float(lg["xwoba_to_run9"])
                profile.xwoba_against = round(xwoba, 4)        # observed (for transparency)
                profile.statcast_rate = round(min(max(rate, 2.0), 7.0), 3)  # regressed -> runs/9

            # If FanGraphs gave us no IP, estimate it from pitch count
            # (~15 pitches/inning) so sample-size confidence stays meaningful.
            if profile.ip is None:
                profile.ip = round(len(sc_full) / 15.0, 1)
                profile.ip_is_estimated = True

            recent_start = (as_of - timedelta(days=recent_window_days)).isoformat()
            sc_recent = get_pitcher_statcast(player_id, recent_start, as_of.isoformat())
            form = _recent_form(sc_recent)
            profile.recent_csw_pct = form["recent_csw_pct"]
            profile.recent_xwoba_con = form["recent_xwoba_con"]
            profile.recent_starts = int(form["recent_starts"] or 0)

    log.debug(
        "Pitcher %s: season=%s statcast=%s completeness=%.2f",
        name, profile.has_season_stats, profile.has_statcast, profile.data_completeness,
    )
    return profile
