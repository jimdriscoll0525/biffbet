"""Static reference data: canonical MLB team names and name normalization.

The Odds API and the MLB Stats API mostly agree on full team names, but not
always (e.g. the Athletics' relocation, "St. Louis" punctuation, abbreviations).
We normalize everything to a single canonical name so odds rows can be matched
to scheduled games reliably.
"""
from __future__ import annotations

# Canonical names == MLB Stats API `team.name` values.
CANONICAL_TEAMS: tuple[str, ...] = (
    "Arizona Diamondbacks",
    "Atlanta Braves",
    "Baltimore Orioles",
    "Boston Red Sox",
    "Chicago Cubs",
    "Chicago White Sox",
    "Cincinnati Reds",
    "Cleveland Guardians",
    "Colorado Rockies",
    "Detroit Tigers",
    "Houston Astros",
    "Kansas City Royals",
    "Los Angeles Angels",
    "Los Angeles Dodgers",
    "Miami Marlins",
    "Milwaukee Brewers",
    "Minnesota Twins",
    "New York Mets",
    "New York Yankees",
    "Athletics",
    "Philadelphia Phillies",
    "Pittsburgh Pirates",
    "San Diego Padres",
    "San Francisco Giants",
    "Seattle Mariners",
    "St. Louis Cardinals",
    "Tampa Bay Rays",
    "Texas Rangers",
    "Toronto Blue Jays",
    "Washington Nationals",
)

# Aliases (lowercased) -> canonical name. Covers abbreviations, old names,
# and punctuation/spacing variants seen across the two APIs.
_ALIASES: dict[str, str] = {
    "oakland athletics": "Athletics",
    "oakland a's": "Athletics",
    "las vegas athletics": "Athletics",
    "sacramento athletics": "Athletics",
    "ath": "Athletics",
    "oak": "Athletics",
    "st louis cardinals": "St. Louis Cardinals",
    "saint louis cardinals": "St. Louis Cardinals",
    "stl": "St. Louis Cardinals",
    "la dodgers": "Los Angeles Dodgers",
    "la angels": "Los Angeles Angels",
    "los angeles angels of anaheim": "Los Angeles Angels",
    "anaheim angels": "Los Angeles Angels",
    "ny yankees": "New York Yankees",
    "ny mets": "New York Mets",
    "cleveland indians": "Cleveland Guardians",  # historical
    "florida marlins": "Miami Marlins",          # historical
    "tampa bay devil rays": "Tampa Bay Rays",    # historical
    "wsh": "Washington Nationals",
    "was": "Washington Nationals",
}

# Build lowercase canonical lookup once.
_CANONICAL_LOOKUP: dict[str, str] = {name.lower(): name for name in CANONICAL_TEAMS}


def normalize_team(name: str | None) -> str | None:
    """Map any reasonable team string to its canonical MLB name.

    Returns None if the input is None/empty. Falls back to a best-effort
    substring match (e.g. "Yankees" -> "New York Yankees") before giving up
    and returning the original (title-cased) string so callers can still log it.
    """
    if not name:
        return None
    key = " ".join(name.strip().lower().split())  # collapse whitespace
    if key in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[key]
    if key in _ALIASES:
        return _ALIASES[key]

    # Substring / nickname fallback: match on the last word (the nickname).
    nickname = key.split()[-1] if key.split() else key
    matches = [c for c in CANONICAL_TEAMS if c.lower().endswith(nickname)]
    if len(matches) == 1:
        return matches[0]

    # Couldn't confidently map it; return a tidied version for logging/matching.
    return name.strip()
