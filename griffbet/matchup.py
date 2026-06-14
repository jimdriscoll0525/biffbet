"""Full per-batter pitch-type matchup model (the review's "biggest upgrade").

For a game with CONFIRMED lineups, model each side's expected offense as the
interaction of the opposing starter's pitch ARSENAL with each lineup batter's
performance BY PITCH TYPE:

    home_offense = mean over home batters of  Σ_pt  away_arsenal[pt] · batter_xwoba[pt]
    away_offense = mean over away batters of  Σ_pt  home_arsenal[pt] · batter_xwoba[pt]
    matchup_net  = home_offense - away_offense          (+ favors home)

Data: Statcast pitch-level (free, via pybaseball) -- the starter's pitch-type mix
and each batter's xwOBA by pitch type, both AS OF the game date (no leakage).
Only computed when BOTH lineups are confirmed (so we have the 9 batter ids and
it's look-ahead-free); otherwise the feature is 0 and accumulates forward.

Heavy by design (≈2 pitchers + up to 18 batters of Statcast per game), so it's
config-gated and cached per (player, as-of-date). The core math
(`compute_matchup`) is a pure function, unit-tested without network.
"""
from __future__ import annotations

from datetime import date as date_cls

from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.matchup")

# Per-process caches keyed by (player_id, as_of_iso).
_ARSENAL_CACHE: dict[tuple, dict] = {}
_BATTER_CACHE: dict[tuple, dict] = {}


# --- pure matchup math (unit-tested) -----------------------------------------
def _lineup_offense(arsenal: dict, batters: list[dict]) -> float | None:
    """Mean over batters of Σ_pt arsenal_freq[pt] · batter_xwoba[pt]. None when
    there's nothing to combine (missing arsenal or no batter data)."""
    if not arsenal or not batters:
        return None
    vals = []
    for b in batters:
        if not b:
            continue
        # Only pitch types the batter has faced contribute; renormalize the
        # arsenal weight over the overlap so missing splits don't bias low.
        overlap = {pt: w for pt, w in arsenal.items() if pt in b}
        wsum = sum(overlap.values())
        if wsum <= 0:
            continue
        vals.append(sum(w * b[pt] for pt, w in overlap.items()) / wsum)
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_matchup(home_arsenal: dict, away_arsenal: dict,
                    home_batters: list[dict], away_batters: list[dict]) -> float:
    """matchup_net = home_offense - away_offense (+ favors home). 0 when either
    side can't be computed (degrade-safe)."""
    home_off = _lineup_offense(away_arsenal, home_batters)   # home bats vs away starter
    away_off = _lineup_offense(home_arsenal, away_batters)   # away bats vs home starter
    if home_off is None or away_off is None:
        return 0.0
    return round(home_off - away_off, 4)


# --- Statcast fetch (cached, degrade-safe) -----------------------------------
def _season_start(season: int) -> str:
    return f"{season}-03-01"


def pitcher_arsenal(pitcher_id, season: int, as_of: date_cls) -> dict:
    """{pitch_type: frequency} for a pitcher, as-of date. {} on any failure."""
    if not pitcher_id:
        return {}
    key = (int(pitcher_id), as_of.isoformat())
    if key in _ARSENAL_CACHE:
        return _ARSENAL_CACHE[key]
    out: dict = {}
    try:
        from pybaseball import statcast_pitcher
        df = statcast_pitcher(_season_start(season), as_of.isoformat(), int(pitcher_id))
        if df is not None and not df.empty and "pitch_type" in df.columns:
            counts = df["pitch_type"].dropna().value_counts()
            total = float(counts.sum())
            if total > 0:
                out = {pt: float(c) / total for pt, c in counts.items()}
    except Exception as exc:  # noqa: BLE001
        log.debug("arsenal fetch failed for %s (%s)", pitcher_id, exc)
    _ARSENAL_CACHE[key] = out
    return out


def batter_xwoba_by_pitch(batter_id, season: int, as_of: date_cls) -> dict:
    """{pitch_type: mean xwOBA} for a batter, as-of date. {} on any failure."""
    if not batter_id:
        return {}
    key = (int(batter_id), as_of.isoformat())
    if key in _BATTER_CACHE:
        return _BATTER_CACHE[key]
    out: dict = {}
    try:
        from pybaseball import statcast_batter
        df = statcast_batter(_season_start(season), as_of.isoformat(), int(batter_id))
        if df is not None and not df.empty and "pitch_type" in df.columns:
            col = "estimated_woba_using_speedangle" if "estimated_woba_using_speedangle" in df.columns else "woba_value"
            if col in df.columns:
                grp = df.dropna(subset=["pitch_type", col]).groupby("pitch_type")[col].mean()
                out = {pt: float(v) for pt, v in grp.items()}
    except Exception as exc:  # noqa: BLE001
        log.debug("batter splits fetch failed for %s (%s)", batter_id, exc)
    _BATTER_CACHE[key] = out
    return out


def matchup_feature(home_pp, away_pp, home_lu, away_lu, season: int,
                    as_of: date_cls, config: dict) -> float:
    """Live matchup_net. Computes only when BOTH lineups are confirmed (so we
    have batter ids and it's look-ahead-free); else 0. Config-gated."""
    if not config.get("griff_features", {}).get("matchup", {}).get("enabled", True):
        return 0.0
    if not (home_lu is not None and getattr(home_lu, "is_confirmed", False)
            and away_lu is not None and getattr(away_lu, "is_confirmed", False)):
        return 0.0
    home_ids = list(getattr(home_lu, "batting_order_ids", []) or [])
    away_ids = list(getattr(away_lu, "batting_order_ids", []) or [])
    if not home_ids or not away_ids:
        return 0.0
    home_arsenal = pitcher_arsenal(getattr(home_pp, "player_id", None), season, as_of)
    away_arsenal = pitcher_arsenal(getattr(away_pp, "player_id", None), season, as_of)
    home_batters = [batter_xwoba_by_pitch(b, season, as_of) for b in home_ids]
    away_batters = [batter_xwoba_by_pitch(b, season, as_of) for b in away_ids]
    return compute_matchup(home_arsenal, away_arsenal, home_batters, away_batters)
