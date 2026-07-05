"""Football team-name normalization (the football twin of constants.normalize_team).

Three naming universes must join cleanly:
  * The Odds API: full display names ("Kansas City Chiefs", "Alabama Crimson Tide")
  * nflverse: team abbreviations ("KC", "LA", "WAS")
  * CFBD: school names ("Alabama", "Ohio State", "Miami (OH)")

Canonical forms used everywhere downstream (store, reasoning, site):
  * NFL: the nflverse abbreviation.
  * CFB: the CFBD school name.

The NFL map is a static 32-row dict. The CFB matcher is a PURE function over a
school/mascot frame (supplied by cfbd_client at runtime, or a fixture in tests)
plus a small manual-override dict for the known ambiguous names.
"""
from __future__ import annotations

import re

# --- NFL: Odds API full name -> nflverse abbreviation -------------------------
NFL_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
NFL_ABBR_TO_NAME: dict[str, str] = {v: k for k, v in NFL_NAME_TO_ABBR.items()}
# nflverse historical/alternate abbreviations that may appear in older frames.
_NFL_ABBR_ALIASES: dict[str, str] = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS"}


def normalize_nfl(name: str | None) -> str | None:
    """Odds-API full name OR any nflverse-ish abbreviation -> canonical abbr."""
    if not name:
        return None
    name = name.strip()
    if name in NFL_NAME_TO_ABBR:
        return NFL_NAME_TO_ABBR[name]
    up = name.upper()
    if up in NFL_ABBR_TO_NAME:
        return up
    if up in _NFL_ABBR_ALIASES:
        return _NFL_ABBR_ALIASES[up]
    return None


# --- CFB: Odds API display name -> CFBD school --------------------------------
# Manual overrides for names the generic matcher gets wrong or that collide.
# Key = Odds API name, value = CFBD school. Extend as mismatches surface (the
# pipeline logs every unmatched slate name).
CFB_NAME_OVERRIDES: dict[str, str] = {
    "Miami Hurricanes": "Miami",
    "Miami (OH) RedHawks": "Miami (OH)",
    "Ole Miss Rebels": "Ole Miss",
    "USC Trojans": "USC",
    "UCF Knights": "UCF",
    "SMU Mustangs": "SMU",
    "TCU Horned Frogs": "TCU",
    "BYU Cougars": "BYU",
    "UTSA Roadrunners": "UT San Antonio",
    "UTEP Miners": "UTEP",
    "UNLV Rebels": "UNLV",
    "UMass Minutemen": "UMass",
    "UConn Huskies": "Connecticut",
    "Appalachian State Mountaineers": "App State",
    "Southern Miss Golden Eagles": "Southern Miss",
    "Louisiana Ragin' Cajuns": "Louisiana",
    "Hawaii Rainbow Warriors": "Hawai'i",
    "San Jose State Spartans": "San José State",
    "Army Black Knights": "Army",
    "Navy Midshipmen": "Navy",
}


def _clean(s: str) -> str:
    """Lowercase, strip punctuation/diacritic-ish noise for fuzzy-ish equality."""
    s = s.lower().replace("'", "'")
    s = re.sub(r"[^a-z0-9() ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build_cfb_matcher(school_mascots: list[tuple[str, str | None]]):
    """Return a normalize(name)->school function over CFBD's (school, mascot)
    pairs. PURE — takes data in, so tests feed it fixtures.

    Match order: manual override -> exact "school mascot" -> exact school ->
    unique school-prefix of the display name. None when nothing matches
    (an FCS opponent, typically) — callers skip and log those games.
    """
    by_full: dict[str, str] = {}
    by_school: dict[str, str] = {}
    for school, mascot in school_mascots:
        by_school[_clean(school)] = school
        if mascot:
            by_full[_clean(f"{school} {mascot}")] = school

    schools_cleaned = sorted(by_school.items(), key=lambda kv: -len(kv[0]))

    def normalize_cfb(name: str | None) -> str | None:
        if not name:
            return None
        if name in CFB_NAME_OVERRIDES:
            return CFB_NAME_OVERRIDES[name]
        c = _clean(name)
        if c in by_full:
            return by_full[c]
        if c in by_school:
            return by_school[c]
        # Longest school name that prefixes the display name ("Ohio State
        # Buckeyes" -> "ohio state"). Longest-first prevents "Ohio" stealing it.
        for cleaned, school in schools_cleaned:
            if c.startswith(cleaned + " ") or c == cleaned:
                return school
        return None

    return normalize_cfb
