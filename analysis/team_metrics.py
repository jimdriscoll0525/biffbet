"""Team-level inputs: offense, bullpen, win%, and park factors.

Sources:
  * Team win% — MLB Stats API standings (authoritative, real-time).
  * Offense — FanGraphs team batting (`team_batting`) wRC+ (100 = league avg).
  * Bullpen — derived from the FanGraphs pitcher leaderboard by aggregating
    *relievers only* (GS/G < 0.5), IP-weighted, into a team relief FIP. pybaseball
    has no clean "team bullpen" endpoint, so we build it from data it DOES return
    rather than inventing a number. (Documented limitation: this is regular-staff
    FIP for relievers, not a leverage-weighted bullpen rating.)
  * Park factors — hardcoded, tunable table in config.yaml.

Everything degrades gracefully to league-average defaults if a pull fails.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlb_value_bot.analysis.pitcher_metrics import _HAVE_PYBASEBALL
from mlb_value_bot.data.cache import cached_dataframe
from mlb_value_bot.data.mlb_client import MLBClient
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("analysis.team_metrics")

if _HAVE_PYBASEBALL:
    import pybaseball  # type: ignore

LEAGUE_AVG_WRC = 100.0


@dataclass
class TeamProfile:
    """Normalized team inputs for the win-probability model."""

    team: str
    wins: float = 0.0
    losses: float = 0.0
    games: float = 0.0
    raw_winpct: float = 0.5
    offense_wrc_plus: float | None = None  # 100 = league average
    bullpen_fip: float | None = None
    park_factor: float = 100.0             # for THIS team's home park

    @property
    def has_record(self) -> bool:
        return self.games > 0


class TeamMetricsProvider:
    """Loads and caches team-level metrics for a season."""

    def __init__(self, season: int, config: dict | None = None, mlb_client: MLBClient | None = None) -> None:
        self.season = season
        self.config = config or load_config()
        self.mlb = mlb_client or MLBClient(self.config)
        self._park_factors: dict[str, int] = self.config.get("park_factors", {})
        self._default_park: int = self.config.get("default_park_factor", 100)
        self._standings: dict[str, dict[str, float]] | None = None
        self._offense: dict[str, float] | None = None
        self._bullpen: dict[str, float] | None = None

    # -- Standings (win%) -----------------------------------------------------
    def standings(self) -> dict[str, dict[str, float]]:
        if self._standings is None:
            try:
                self._standings = self.mlb.get_standings(self.season)
            except Exception as exc:
                log.warning("standings fetch failed (%s); using empty", exc)
                self._standings = {}
        return self._standings

    # -- Offense (wRC+) -------------------------------------------------------
    def _team_batting(self) -> pd.DataFrame:
        # Prefer a manually exported FanGraphs CSV (Cloudflare blocks scraping).
        from mlb_value_bot.data.fangraphs_csv import load_batting_csv

        csv = load_batting_csv(self.season)
        if csv is not None and not csv.empty:
            return csv

        if not _HAVE_PYBASEBALL:
            return pd.DataFrame()

        def _producer() -> pd.DataFrame:
            return pybaseball.team_batting(self.season, self.season)  # type: ignore[union-attr]

        return cached_dataframe(f"fg_team_batting_{self.season}", _producer)

    def offense_ratings(self) -> dict[str, float]:
        """Map canonical team -> offense index (100 = average). Empty if unavailable.

        Prefers FanGraphs wRC+; falls back to an automatable OPS+ proxy from the
        MLB Stats API when FanGraphs is blocked (the usual case).
        """
        if self._offense is not None:
            return self._offense
        df = self._team_batting()
        out: dict[str, float] = {}
        if not df.empty and "Team" in df.columns and "wRC+" in df.columns:
            for _, row in df.iterrows():
                team = _match_fg_team(str(row["Team"]))
                val = row.get("wRC+")
                if team and pd.notna(val):
                    out[team] = float(val)
        if not out:
            out = self._offense_proxy()
        self._offense = out
        return out

    def _offense_proxy(self) -> dict[str, float]:
        """OPS+ (100 = league avg) from MLB Stats API, when FanGraphs wRC+ is
        unavailable. OPS+ = 100 * (OBP/lgOBP + SLG/lgSLG - 1) — same 100-centered
        scale the park/offense component expects.
        """
        hitting = self.mlb.get_team_hitting(self.season)
        if not hitting:
            return {}
        obps = [v["obp"] for v in hitting.values() if v.get("obp")]
        slgs = [v["slg"] for v in hitting.values() if v.get("slg")]
        if not obps or not slgs:
            return {}
        lg_obp = sum(obps) / len(obps)
        lg_slg = sum(slgs) / len(slgs)
        if lg_obp <= 0 or lg_slg <= 0:
            return {}
        out = {
            team: 100.0 * (v["obp"] / lg_obp + v["slg"] / lg_slg - 1.0)
            for team, v in hitting.items()
            if v.get("obp") and v.get("slg")
        }
        if out:
            log.info("Offense: MLB Stats API OPS+ proxy for %d teams (FanGraphs wRC+ unavailable)", len(out))
        return out

    # -- Bullpen FIP (derived from reliever leaderboard) ----------------------
    def _reliever_leaderboard(self) -> pd.DataFrame:
        # Same source as the starter leaderboard (CSV-preferred, then scrape),
        # so a single pitching export feeds both starter rates and bullpen FIP.
        from mlb_value_bot.analysis.pitcher_metrics import get_season_pitching

        return get_season_pitching(self.season)

    def bullpen_fip(self) -> dict[str, float]:
        """IP-weighted relief FIP per team (relievers := GS/G < 0.5)."""
        if self._bullpen is not None:
            return self._bullpen
        df = self._reliever_leaderboard()
        out: dict[str, float] = {}
        needed = {"Team", "FIP", "IP", "GS", "G"}
        if not df.empty and needed.issubset(df.columns):
            relievers = df[(df["G"] > 0) & (df["GS"] / df["G"] < 0.5)].copy()
            relievers = relievers.dropna(subset=["FIP", "IP"])
            for team_raw, grp in relievers.groupby("Team"):
                team = _match_fg_team(str(team_raw))
                ip_total = grp["IP"].sum()
                if team and ip_total > 0:
                    out[team] = float((grp["FIP"] * grp["IP"]).sum() / ip_total)
        if not out:
            # FanGraphs blocked -> automatable proxy: IP-weighted reliever ERA
            # from the MLB Stats API (same ~4.10 scale as relief FIP).
            out = self.mlb.get_team_bullpen_era(self.season)
            if out:
                log.info("Bullpen: MLB Stats API reliever-ERA proxy for %d teams (FanGraphs FIP unavailable)", len(out))
        self._bullpen = out
        return out

    # -- Park factor ----------------------------------------------------------
    def park_factor(self, home_team: str) -> float:
        return float(self._park_factors.get(home_team, self._default_park))

    # -- Assembly -------------------------------------------------------------
    def build_team_profile(self, team: str, is_home: bool) -> TeamProfile:
        rec = self.standings().get(team, {})
        profile = TeamProfile(
            team=team,
            wins=rec.get("wins", 0.0),
            losses=rec.get("losses", 0.0),
            games=rec.get("games", 0.0),
            raw_winpct=rec.get("winpct", 0.5),
            offense_wrc_plus=self.offense_ratings().get(team),
            bullpen_fip=self.bullpen_fip().get(team),
            park_factor=self.park_factor(team) if is_home else 100.0,
        )
        return profile


# FanGraphs uses team abbreviations (e.g. "NYY", "LAD") in some tables and full
# names in others. Map both to canonical names.
_FG_ABBR = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CWS": "Chicago White Sox", "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers", "HOU": "Houston Astros",
    "KCR": "Kansas City Royals", "KC": "Kansas City Royals", "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets", "NYY": "New York Yankees",
    "OAK": "Athletics", "ATH": "Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SD": "San Diego Padres",
    "SFG": "San Francisco Giants", "SF": "San Francisco Giants", "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays", "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
    "WSH": "Washington Nationals",
}


def _match_fg_team(raw: str) -> str | None:
    """Resolve a FanGraphs team string (abbr or full name) to canonical.

    Returns None if it can't be resolved to one of the 30 canonical teams, so
    callers never key metrics under a garbage name.
    """
    from mlb_value_bot.constants import CANONICAL_TEAMS, normalize_team

    raw = raw.strip()
    if raw.upper() in _FG_ABBR:
        return _FG_ABBR[raw.upper()]
    canon = normalize_team(raw)
    return canon if canon in set(CANONICAL_TEAMS) else None
