"""Stage 4 — extra FREE training features for GriffBet's residual model.

Three groups, all derivable from data GriffBet already pulls (plus one free,
no-key weather feed):

  * pitcher pitch-quality nets (whiff/CSW/hard-hit) -- a pragmatic, free stand-in
    for the review's "pitch-type matchup" using Statcast metrics already on
    PitcherProfile. (NOT the full pitcher x batter x pitch-type model -- that's a
    much larger build flagged for later.)
  * lineup state -- confirmed-vs-projected flag + net key-bats missing, richer
    than the clamped lineup component.
  * weather / run environment -- temperature + wind from Open-Meteo (free, no
    key), keyed by ballpark coordinates. Degrades to neutral (0) on any failure.

These accumulate FORWARD: computed live each run and stored in
reasoning["griff_features"], so they enter the training set as games are graded.
Historical rows that predate them read 0 (the extractor's default), so they have
no effect until enough forward data accrues -- the honest "accumulate live" path.
"""
from __future__ import annotations

from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.features")

# Shared schema: the residual model's FEATURES list appends these, and both the
# live pipeline and the training extractor read exactly these keys.
GRIFF_FEATURE_KEYS = [
    "whiff_net", "csw_net", "hardhit_net",       # pitcher pitch-quality (+ favors home)
    "lineup_confirmed", "keybats_net",            # lineup state
    "temp", "wind",                               # weather / run environment
]


def _net(home_val, away_val) -> float:
    """home - away, or 0 if either side is missing."""
    if home_val is None or away_val is None:
        return 0.0
    return float(home_val) - float(away_val)


def pitcher_quality_features(home_pp, away_pp) -> dict:
    """Pitch-quality nets, signed so + favors HOME. Whiff/CSW: higher = better,
    so home - away. Hard-hit: lower = better, so away - home."""
    return {
        "whiff_net": _net(getattr(home_pp, "whiff_pct", None), getattr(away_pp, "whiff_pct", None)),
        "csw_net": _net(getattr(home_pp, "csw_pct", None), getattr(away_pp, "csw_pct", None)),
        "hardhit_net": _net(getattr(away_pp, "hardhit_pct", None), getattr(home_pp, "hardhit_pct", None)),
    }


def lineup_features(home_lu, away_lu) -> dict:
    """Confirmed flag (1 iff both lineups posted) + net key-bats missing
    (away_missing - home_missing, + favors home). 0 when not both confirmed."""
    both_confirmed = (
        home_lu is not None and getattr(home_lu, "is_confirmed", False)
        and away_lu is not None and getattr(away_lu, "is_confirmed", False)
    )
    if not both_confirmed:
        return {"lineup_confirmed": 0.0, "keybats_net": 0.0}
    h_missing = float(getattr(home_lu, "missing_count", 0) or 0)
    a_missing = float(getattr(away_lu, "missing_count", 0) or 0)
    return {"lineup_confirmed": 1.0, "keybats_net": a_missing - h_missing}


# --- weather (Open-Meteo, free, no key) --------------------------------------
# Ballpark coordinates keyed by canonical home-team name.
_PARK_COORDS = {
    "Arizona Diamondbacks": (33.4453, -112.0667), "Atlanta Braves": (33.8907, -84.4677),
    "Baltimore Orioles": (39.2839, -76.6217), "Boston Red Sox": (42.3467, -71.0972),
    "Chicago Cubs": (41.9484, -87.6553), "Chicago White Sox": (41.8299, -87.6338),
    "Cincinnati Reds": (39.0975, -84.5066), "Cleveland Guardians": (41.4962, -81.6852),
    "Colorado Rockies": (39.7559, -104.9942), "Detroit Tigers": (42.3390, -83.0485),
    "Houston Astros": (29.7572, -95.3556), "Kansas City Royals": (39.0517, -94.4803),
    "Los Angeles Angels": (33.8003, -117.8827), "Los Angeles Dodgers": (34.0739, -118.2400),
    "Miami Marlins": (25.7780, -80.2197), "Milwaukee Brewers": (43.0280, -87.9712),
    "Minnesota Twins": (44.9817, -93.2776), "New York Mets": (40.7571, -73.8458),
    "New York Yankees": (40.8296, -73.9262), "Athletics": (38.7590, -121.2700),
    "Philadelphia Phillies": (39.9061, -75.1665), "Pittsburgh Pirates": (40.4469, -80.0057),
    "San Diego Padres": (32.7073, -117.1566), "San Francisco Giants": (37.7786, -122.3893),
    "Seattle Mariners": (47.5914, -122.3325), "St. Louis Cardinals": (38.6226, -90.1928),
    "Tampa Bay Rays": (27.7683, -82.6534), "Texas Rangers": (32.7473, -97.0847),
    "Toronto Blue Jays": (43.6414, -79.3894), "Washington Nationals": (38.8730, -77.0074),
}

_WEATHER_CACHE: dict[tuple, dict] = {}


def weather_features(home_team: str, game_date: str, config: dict) -> dict:
    """Temperature (centered at 20C) + wind speed at the ballpark, from
    Open-Meteo. Neutral (0/0) when disabled, coordinates unknown, or the call
    fails -- never raises. Cached per (date, team)."""
    cfg = config.get("griff_features", {}).get("weather", {})
    if not cfg.get("enabled", True):
        return {"temp": 0.0, "wind": 0.0}
    coords = _PARK_COORDS.get(home_team)
    if coords is None:
        return {"temp": 0.0, "wind": 0.0}
    key = (game_date, home_team)
    if key in _WEATHER_CACHE:
        return _WEATHER_CACHE[key]
    out = {"temp": 0.0, "wind": 0.0}
    try:
        import requests
        lat, lon = coords
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current": "temperature_2m,wind_speed_10m"},
            timeout=float(cfg.get("timeout", 8)),
        )
        if resp.status_code < 300:
            cur = resp.json().get("current", {})
            temp_c = cur.get("temperature_2m")
            wind = cur.get("wind_speed_10m")
            if temp_c is not None:
                out["temp"] = round(float(temp_c) - 20.0, 2)   # centered: 0 ≈ 20C neutral
            if wind is not None:
                out["wind"] = round(float(wind) / 10.0, 3)      # scale ~O(1)
    except Exception as exc:  # noqa: BLE001
        log.debug("weather lookup failed for %s (%s)", home_team, exc)
    _WEATHER_CACHE[key] = out
    return out


def extra_features(home_pp, away_pp, home_lu, away_lu, home_team: str,
                   game_date: str, config: dict) -> dict:
    """All Stage-4 features for one game, keyed by GRIFF_FEATURE_KEYS."""
    feat = {}
    feat.update(pitcher_quality_features(home_pp, away_pp))
    feat.update(lineup_features(home_lu, away_lu))
    feat.update(weather_features(home_team, game_date, config))
    return {k: feat.get(k, 0.0) for k in GRIFF_FEATURE_KEYS}
