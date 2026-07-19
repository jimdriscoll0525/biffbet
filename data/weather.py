"""Weather / run-environment for the totals model (Open-Meteo, free, no key).

Produces a run-environment MULTIPLIER from temperature + wind projected onto the
ballpark's home->center-field axis (blowing out boosts scoring, in suppresses) +
roof handling. The dominant totals factor and a hard requirement for v1.

Graceful degradation (hard rule): a missing feed, unknown park, or fixed/closed
roof returns multiplier 1.0 with available=False or a roof flag -- never a
fabricated tilt. The totals pipeline drops confidence and flags "weather
unavailable" so we never bet a total blind to weather.

NOTE: park orientations are approximate (home->CF compass bearing) and
config-overridable. Retractable roofs are treated OPEN by default (free data
can't detect a closed retractable) and flagged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from mlb_value_bot.utils import get_logger

log = get_logger("data.weather")

# (latitude, longitude) per canonical home-team name.
PARK_COORDS = {
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

# Approximate home-plate -> center-field compass bearing (degrees from N, CW).
# Wind blowing TOWARD this bearing = "out" (boost); FROM it = "in" (suppress).
PARK_ORIENTATION = {
    "Arizona Diamondbacks": 0, "Atlanta Braves": 25, "Baltimore Orioles": 30,
    "Boston Red Sox": 45, "Chicago Cubs": 30, "Chicago White Sox": 30,
    "Cincinnati Reds": 60, "Cleveland Guardians": 0, "Colorado Rockies": 0,
    "Detroit Tigers": 30, "Houston Astros": 0, "Kansas City Royals": 0,
    "Los Angeles Angels": 30, "Los Angeles Dodgers": 30, "Miami Marlins": 40,
    "Milwaukee Brewers": 0, "Minnesota Twins": 90, "New York Mets": 25,
    "New York Yankees": 80, "Athletics": 60, "Philadelphia Phillies": 20,
    "Pittsburgh Pirates": 60, "San Diego Padres": 0, "San Francisco Giants": 95,
    "Seattle Mariners": 70, "St. Louis Cardinals": 60, "Tampa Bay Rays": 45,
    "Texas Rangers": 75, "Toronto Blue Jays": 0, "Washington Nationals": 30,
}

# Roof handling. Fixed = always closed/climate-controlled (weather-neutral).
# Retractable = open by default + flagged (we can't detect a closed retractable
# from free data). Everything else is open-air.
FIXED_ROOF = {"Tampa Bay Rays"}
RETRACTABLE_ROOF = {
    "Toronto Blue Jays", "Houston Astros", "Milwaukee Brewers", "Arizona Diamondbacks",
    "Miami Marlins", "Texas Rangers", "Seattle Mariners",
}

_WEATHER_CACHE: dict[tuple, dict] = {}


@dataclass
class WeatherEnv:
    multiplier: float          # run-environment multiplier (1.0 = neutral)
    available: bool            # False -> bet blind to weather; flag + drop confidence
    temp_c: float | None
    wind_kmh: float | None
    wind_out_component: float | None  # + = blowing out to CF, - = in
    roof: str                  # "open" | "retractable_assumed_open" | "fixed_closed"
    note: str


class _OpenMeteoTransient(Exception):
    """Server-side flake (5xx / 429) worth retrying."""


def _get_open_meteo(lat: float, lon: float, timeout: float):
    """One GET, wrapped in the same tenacity pattern as the other API clients.

    2026-07-19: six of sixteen parks failed the single un-retried fetch in one
    run, holding an otherwise-qualified totals pick to analysis-only. Retrying
    transient failures (connection/timeout/5xx/429) recovers most of these.
    """
    import requests
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.5, max=4),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, _OpenMeteoTransient)
        ),
    )
    def _get():
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "current": "temperature_2m,wind_speed_10m,wind_direction_10m"},
            timeout=timeout,
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _OpenMeteoTransient(f"open-meteo HTTP {resp.status_code}")
        return resp

    return _get()


def _fetch_open_meteo(lat: float, lon: float, timeout: float) -> dict | None:
    try:
        resp = _get_open_meteo(lat, lon, timeout)
        if resp.status_code < 300:
            return resp.json().get("current", {})
        log.warning("open-meteo fetch failed (HTTP %s)", resp.status_code)
    except Exception as exc:  # noqa: BLE001
        # WARNING (not debug): an exhausted retry here holds a totals pick to
        # analysis-only, so it must be visible in the pipeline logs.
        log.warning("open-meteo fetch failed after retries (%s)", exc)
    return None


def _wind_out_component(wind_kmh: float, wind_from_deg: float, out_bearing: float) -> float:
    """Signed wind speed along the home->CF axis. + = blowing out, - = in.

    Open-Meteo wind_direction is the direction the wind blows FROM
    (meteorological), so it blows TOWARD (from + 180). Project that onto the
    out-to-CF bearing via cosine of the angle between them."""
    wind_to = (wind_from_deg + 180.0) % 360.0
    align = math.cos(math.radians(wind_to - out_bearing))   # +1 out, -1 in, 0 cross
    return wind_kmh * align


def weather_env(home_team: str, game_date: str, config: dict) -> WeatherEnv:
    """Run-environment multiplier for a game's ballpark. Degrade-safe."""
    cfg = config.get("totals", {}).get("weather", {})
    if not cfg.get("enabled", True):
        return WeatherEnv(1.0, False, None, None, None, "open", "weather disabled")

    # Fixed dome -> weather-neutral, but this is KNOWN (not a missing feed), so
    # it's available (we correctly model "no weather effect").
    if home_team in FIXED_ROOF:
        return WeatherEnv(1.0, True, None, None, None, "fixed_closed", "fixed roof (climate-controlled)")

    coords = PARK_COORDS.get(home_team)
    if coords is None:
        return WeatherEnv(1.0, False, None, None, None, "open", "unknown ballpark coordinates")

    key = (game_date, home_team)
    if key in _WEATHER_CACHE:
        d = _WEATHER_CACHE[key]
        return WeatherEnv(**d)

    cur = _fetch_open_meteo(*coords, float(cfg.get("timeout", 8)))
    if not cur or cur.get("temperature_2m") is None:
        env = WeatherEnv(1.0, False, None, None, None, "open", "weather feed unavailable")
        _WEATHER_CACHE[key] = env.__dict__
        return env

    temp_c = float(cur["temperature_2m"])
    wind_kmh = float(cur.get("wind_speed_10m") or 0.0)
    wind_from = float(cur.get("wind_direction_10m") or 0.0)
    out_bearing = float(PARK_ORIENTATION.get(home_team, 45))
    out_comp = _wind_out_component(wind_kmh, wind_from, out_bearing)

    temp_coef = float(cfg.get("temp_coef_per_c", 0.006))
    temp_ref = float(cfg.get("temp_ref_c", 20.0))
    wind_out = float(cfg.get("wind_out_coef", 0.010))
    wind_in = float(cfg.get("wind_in_coef", 0.010))
    max_tilt = float(cfg.get("max_weather_tilt", 0.08))

    temp_effect = (temp_c - temp_ref) * temp_coef
    wind_effect = out_comp * (wind_out if out_comp >= 0 else wind_in)
    mult = max(1.0 - max_tilt, min(1.0 + max_tilt, 1.0 + temp_effect + wind_effect))

    roof = "retractable_assumed_open" if home_team in RETRACTABLE_ROOF else "open"
    note = (f"{temp_c:.0f}C, wind {wind_kmh:.0f}km/h "
            f"{'out' if out_comp >= 0 else 'in'} {abs(out_comp):.0f}"
            + ("; retractable roof assumed OPEN" if roof == "retractable_assumed_open" else ""))
    env = WeatherEnv(round(mult, 4), True, round(temp_c, 1), round(wind_kmh, 1),
                     round(out_comp, 1), roof, note)
    _WEATHER_CACHE[key] = env.__dict__
    return env
