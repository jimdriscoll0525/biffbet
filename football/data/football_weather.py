"""Game-time weather for football totals (Open-Meteo, free, no key).

Football weather is SUPPRESS-ONLY by design (Jim's spec + the MLB over-bias
lesson): sustained wind kills passing efficiency and kicking, extreme cold
kills scoring, and nothing in the v1 model moves a total UP on weather. The
multiplier is therefore <= 1.0 by construction, clamped at 1 - max_tilt.

Unlike the MLB module this reads the HOURLY forecast at the kickoff hour
(football slates are known days out; current conditions on Tuesday say nothing
about Sunday), in imperial units to match the config thresholds.

Graceful degradation (hard rule): indoor game -> 1.0 with available=True (a
KNOWN no-effect state); missing feed / unknown coordinates -> 1.0 with
available=False, and the pipeline holds outdoor totals to analysis-only
(weather.require_for_bet) — we never bet an outdoor total blind to weather.
"""
from __future__ import annotations

from dataclasses import dataclass

from mlb_value_bot.utils import get_logger

log = get_logger("football.data.weather")

_WEATHER_CACHE: dict[tuple, dict] = {}


@dataclass
class FootballWeather:
    multiplier: float          # total multiplier, <= 1.0 (1.0 = no suppression)
    available: bool            # False -> outdoor total held to analysis-only
    indoor: bool
    temp_f: float | None
    wind_mph: float | None
    note: str


def _fetch_hourly(lat: float, lon: float, date_iso: str, timeout: float) -> dict | None:
    try:
        import requests
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "start_date": date_iso, "end_date": date_iso, "timezone": "UTC",
            },
            timeout=timeout,
        )
        if resp.status_code < 300:
            return resp.json().get("hourly", {})
    except Exception as exc:  # noqa: BLE001
        log.debug("open-meteo fetch failed (%s)", exc)
    return None


def suppression_multiplier(temp_f: float | None, wind_mph: float | None, cfg: dict) -> float:
    """PURE: the suppress-only multiplier from temperature + wind. Testable."""
    max_tilt = float(cfg.get("max_tilt", 0.06))
    wind_start = float(cfg.get("wind_mph_start", 12.0))
    wind_coef = float(cfg.get("wind_coef_per_mph", 0.006))
    cold_start = float(cfg.get("cold_f_start", 25.0))
    cold_coef = float(cfg.get("cold_coef_per_f", 0.004))

    suppression = 0.0
    if wind_mph is not None and wind_mph > wind_start:
        suppression += (wind_mph - wind_start) * wind_coef
    if temp_f is not None and temp_f < cold_start:
        suppression += (cold_start - temp_f) * cold_coef
    return max(1.0 - max_tilt, 1.0 - suppression)


def game_weather(lat: float | None, lon: float | None, kickoff_utc: str | None,
                 indoor: bool, config: dict) -> FootballWeather:
    """Weather for one game. kickoff_utc is ISO8601; only the date+hour are used."""
    cfg = config.get("weather", {})
    if not cfg.get("enabled", True):
        return FootballWeather(1.0, False, indoor, None, None, "weather disabled")

    # Indoor is a KNOWN no-effect state, not a missing feed.
    if indoor:
        return FootballWeather(1.0, True, True, None, None, "indoor (dome/closed roof)")

    if lat is None or lon is None or not kickoff_utc:
        return FootballWeather(1.0, False, False, None, None, "unknown stadium coordinates/kickoff")

    date_iso, hour = kickoff_utc[:10], 18
    if len(kickoff_utc) >= 13:
        try:
            hour = int(kickoff_utc[11:13])
        except ValueError:
            pass

    key = (round(lat, 3), round(lon, 3), date_iso, hour)
    if key in _WEATHER_CACHE:
        return FootballWeather(**_WEATHER_CACHE[key])

    hourly = _fetch_hourly(lat, lon, date_iso, float(cfg.get("timeout", 8)))
    temps = (hourly or {}).get("temperature_2m") or []
    winds = (hourly or {}).get("wind_speed_10m") or []
    if len(temps) <= hour or len(winds) <= hour or temps[hour] is None:
        env = FootballWeather(1.0, False, False, None, None, "weather feed unavailable")
        _WEATHER_CACHE[key] = env.__dict__
        return env

    temp_f = float(temps[hour])
    wind_mph = float(winds[hour] or 0.0)
    mult = suppression_multiplier(temp_f, wind_mph, cfg)
    note = f"{temp_f:.0f}F, wind {wind_mph:.0f}mph at kickoff"
    if mult < 1.0:
        note += f" -> total x{mult:.3f}"
    env = FootballWeather(round(mult, 4), True, False, round(temp_f, 1), round(wind_mph, 1), note)
    _WEATHER_CACHE[key] = env.__dict__
    return env
