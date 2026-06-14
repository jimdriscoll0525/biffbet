"""Team rotation-average starting-pitching strength (GriffBet only).

GriffBet's starter-neutralized base rate needs a team's season rotation-average
run-prevention rate (starters, GS/G >= 0.5). Two sources, in order:

  1. FanGraphs pitching leaderboard (xFIP-preferred) -- the same shared CSV
     BiffBet uses, when present.
  2. MLB Stats API starter ERA -- an AUTOMATABLE proxy that works everywhere
     with no CSV (FanGraphs is Cloudflare-blocked to scrapers). This mirrors
     BiffBet's existing reliever-ERA bullpen proxy (get_team_bullpen_era) with
     the predicate flipped to starters.

The neutralization increment is centered on the EMPIRICAL league-mean rotation
rate, not a fixed xFIP constant, so the xFIP path and the ERA-proxy path are
both scale-correct: a team is neutralized by how much better/worse its rotation
is than the league-average rotation, in whatever metric we actually have.

All ADDITIVE -- never touches team_metrics.py or mlb_client.py. (It does call
MLBClient's lower-level _get/_parse_ip by import, which reuses BiffBet's HTTP
retry/config without modifying it.)
"""
from __future__ import annotations

import pandas as pd

from mlb_value_bot.analysis.team_metrics import _match_fg_team
from mlb_value_bot.constants import normalize_team
from mlb_value_bot.data.mlb_client import MLBClient
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.team_extras")

# Manual per-season memo (config isn't hashable, so no lru_cache). Stores the
# rates dict and which source produced it.
_RATES_CACHE: dict[int, dict[str, float]] = {}
_SOURCE_CACHE: dict[int, str | None] = {}


def _rotation_rate_column(df: pd.DataFrame) -> str | None:
    for col in ("xFIP", "SIERA", "FIP"):
        if col in df.columns:
            return col
    return None


def _from_fangraphs(season: int) -> tuple[dict[str, float], str | None]:
    """IP-weighted rotation rate per team from the FanGraphs leaderboard
    (starters := GS/G >= 0.5). Empty when the leaderboard is unavailable."""
    from mlb_value_bot.analysis.pitcher_metrics import get_season_pitching

    df = get_season_pitching(season)
    out: dict[str, float] = {}
    if df is None or df.empty:
        return out, None
    rate_col = _rotation_rate_column(df)
    if rate_col is None or not {"Team", "IP", "GS", "G"}.issubset(df.columns):
        return out, None
    starters = df[(df["G"] > 0) & (df["GS"] / df["G"] >= 0.5)].copy()
    starters = starters.dropna(subset=[rate_col, "IP"])
    for team_raw, grp in starters.groupby("Team"):
        team = _match_fg_team(str(team_raw))
        ip_total = grp["IP"].sum()
        if team and ip_total > 0:
            out[team] = float((grp[rate_col] * grp["IP"]).sum() / ip_total)
    return out, (f"fangraphs_{rate_col.lower()}" if out else None)


def _from_mlb_starter_era(season: int, config: dict | None) -> dict[str, float]:
    """IP-weighted STARTER ERA per team from the MLB Stats API (starters :=
    GS/G >= 0.5). Automatable stand-in for FanGraphs rotation xFIP -- the exact
    inverse of MLBClient.get_team_bullpen_era. {} on failure."""
    mlb = MLBClient(config) if config is not None else MLBClient()
    try:
        data = mlb._get(  # noqa: SLF001 -- reuse BiffBet's HTTP layer, no mutation
            "/v1/stats",
            {"stats": "season", "group": "pitching", "season": int(season),
             "sportId": 1, "gameType": "R", "playerPool": "all", "limit": 3000},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("rotation starter-ERA fetch failed (%s)", exc)
        return {}
    ip_sum: dict[str, float] = {}
    era_ip: dict[str, float] = {}
    for grp in data.get("stats", []):
        for sp in grp.get("splits", []):
            team = normalize_team(sp.get("team", {}).get("name"))
            st = sp.get("stat", {})
            g = st.get("gamesPlayed") or 0
            gs = st.get("gamesStarted") or 0
            if not team or not g or gs / g < 0.5:  # keep STARTERS only
                continue
            innings = mlb._parse_ip(st.get("inningsPitched"))  # noqa: SLF001
            try:
                era = float(st.get("era"))
            except (TypeError, ValueError):
                continue
            if innings <= 0:
                continue
            ip_sum[team] = ip_sum.get(team, 0.0) + innings
            era_ip[team] = era_ip.get(team, 0.0) + era * innings
    return {t: era_ip[t] / ip_sum[t] for t in ip_sum if ip_sum[t] > 0}


def team_rotation_rates(season: int, config: dict | None = None) -> dict[str, float]:
    """{canonical_team -> rotation-average rate}. FanGraphs xFIP if available,
    else the MLB Stats API starter-ERA proxy. Memoized per season. Empty only if
    BOTH sources fail (then neutralization degrades to BiffBet's base)."""
    if season in _RATES_CACHE:
        return _RATES_CACHE[season]
    rates, source = _from_fangraphs(season)
    if not rates:
        rates = _from_mlb_starter_era(season, config)
        source = "mlb_starter_era" if rates else None
    if rates:
        log.info("rotation rates (%s): %d teams via %s", season, len(rates), source)
    _RATES_CACHE[season] = rates
    _SOURCE_CACHE[season] = source
    return rates


def rotation_rate_source(season: int) -> str | None:
    """Which source produced the cached rotation rates (for the reasoning trail)."""
    return _SOURCE_CACHE.get(season)


def rotation_winpct_increment(team: str, season: int, config: dict) -> float | None:
    """Win% increment (vs the LEAGUE-AVERAGE rotation) for a team's season
    rotation-average starting pitching.

        increment = (league_mean_rate - team_rotation_rate) * pitcher_run_to_winpct

    Centering on the empirical league mean (not a fixed xFIP constant) keeps the
    xFIP and ERA-proxy paths scale-consistent and makes the increments sum to ~0
    across the league -- neutralization redistributes around the mean rather than
    shifting everyone. This is the amount stripped from a team's regressed win%
    before log5, so today's starter delta isn't double-counted with the season
    rotation quality already baked into the win-loss record.

    Returns None when no rotation rate exists for the team (that side stays
    un-neutralized -- graceful degradation)."""
    rates = team_rotation_rates(season, config)
    rate = rates.get(team)
    if rate is None or not rates:
        return None
    league_mean = sum(rates.values()) / len(rates)
    run_to_wp = float(config["model"].get("pitcher_run_to_winpct", 0.065))
    return (league_mean - rate) * run_to_wp
