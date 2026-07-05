"""Football stadium table: coordinates + indoor detection for the weather layer.

NFL: a static 32-team dict (nflverse abbr -> lat/lon), like the MLB park table.
Roof truth comes from the nflverse SCHEDULE's per-game `roof` column when
present ('dome'/'closed' = indoor; 'open'/'outdoors' = outdoor) — a retractable
roof reported closed for a given game IS closed, which is better information
than a static building list. The static dome set below is only the fallback
when a schedule row lacks a roof value.

CFB: coordinates come from CFBD's /venues frame (cached weekly); the lookup is
a pure function over that frame so tests can feed fixtures. Unknown venue ->
None -> weather unavailable -> outdoor totals held to analysis-only downstream.
"""
from __future__ import annotations

import pandas as pd

# nflverse team abbr -> (lat, lon) of the home stadium.
NFL_STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4010), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LV": (36.0909, -115.1833), "LAC": (33.9535, -118.3392),
    "LA": (33.9535, -118.3392), "MIA": (25.9580, -80.2389), "MIN": (44.9738, -93.2577),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SF": (37.4032, -121.9698), "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9077, -76.8645),
}

# Fallback ONLY (schedule `roof` wins): fixed domes / default-closed buildings.
NFL_INDOOR_FALLBACK = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LV", "LAC", "LA", "MIN", "NO"}

# Schedule roof values meaning the game is played indoors.
_INDOOR_ROOF_VALUES = {"dome", "closed"}


def nfl_coords(team_abbr: str) -> tuple[float, float] | None:
    return NFL_STADIUM_COORDS.get(team_abbr)


def is_indoor(roof_value: str | None, home_abbr: str | None = None) -> bool:
    """Indoor game? Trust the per-game roof value; fall back to the dome set."""
    if roof_value:
        return str(roof_value).strip().lower() in _INDOOR_ROOF_VALUES
    return home_abbr in NFL_INDOOR_FALLBACK if home_abbr else False


def cfb_venue(venues: pd.DataFrame, venue_id: int | None) -> dict | None:
    """(lat, lon, dome) for a CFBD venue id, from the cached /venues frame.

    Pure lookup — returns None when the frame is empty/missing the venue, and
    the weather layer degrades from there.
    """
    if venues is None or venues.empty or venue_id is None:
        return None
    id_col = "id" if "id" in venues.columns else None
    if id_col is None:
        return None
    row = venues[venues[id_col] == venue_id]
    if row.empty:
        return None
    r = row.iloc[0]
    lat = r.get("location.latitude", r.get("latitude"))
    lon = r.get("location.longitude", r.get("longitude"))
    if pd.isna(lat) or pd.isna(lon):
        return None
    dome = bool(r.get("dome")) if "dome" in row.columns and pd.notna(r.get("dome")) else False
    return {"lat": float(lat), "lon": float(lon), "dome": dome}
