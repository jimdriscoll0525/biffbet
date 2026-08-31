"""NFL QB availability guard — the football twin of MLB's scratched-starter
guard (2026-08-31, built at Jim's request).

The failure mode: team EPA was earned by QB A; this week QB B starts. The
market reprices in minutes; season-long unit stats don't. nflverse's injury
feed died after 2024, so status comes from ESPN's free public injuries feed
(one HTTP call for all 32 teams, cached via the parquet TTL cache).

The guard NEVER models the backup — it only refuses to commit a NEW pick on
a game whose primary passer is listed Out/Doubtful/IR/PUP, with a reason
string on the card. Already-committed bets stay frozen (upsert never
downgrades a bet — same grandfathering as every other filter).

Degradation contract: any failure (ESPN down, pbp parquet predates the
passer column, name mismatch) disables the guard for the run with a logged
note — it must never kill or silently thin the slate.

Known limitation (documented, not solved in v1): an OFFSEASON QB change
(free agency, trade, rookie) is the same staleness problem but is invisible
here — the departed QB is healthy, just elsewhere. The prior-blend decay
retires his stats by week 7.
"""
from __future__ import annotations

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mlb_value_bot.data.cache import cached_dataframe
from mlb_value_bot.football.data.teams import NFL_NAME_TO_ABBR
from mlb_value_bot.utils import get_logger

log = get_logger("football.data.qb_status")

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"

# Name suffixes ignored when matching "P.Mahomes" (pbp) to "Patrick Mahomes"
# (ESPN). Kept tiny on purpose: last name + first initial is the whole match.
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


@retry(reraise=True, stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)))
def _fetch_injuries(timeout: float) -> list | dict:
    resp = requests.get(ESPN_INJURIES_URL, timeout=timeout,
                        headers={"User-Agent": "biffbet-qb-guard/1.0"})
    resp.raise_for_status()
    return resp.json()


def _flatten(payload: dict) -> pd.DataFrame:
    """ESPN payload -> one row per injury entry with the canonical abbr."""
    rows = []
    for team in payload.get("injuries", []):
        abbr = NFL_NAME_TO_ABBR.get(team.get("displayName", ""))
        if not abbr:
            continue
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete") or {}
            rows.append({
                "team": abbr,
                "athlete": athlete.get("displayName") or "",
                "position": ((athlete.get("position") or {}).get("abbreviation") or ""),
                "status": inj.get("status") or "",
            })
    return pd.DataFrame(rows, columns=["team", "athlete", "position", "status"])


def league_injuries(config: dict, force_refresh: bool = False) -> pd.DataFrame:
    ttl = float(config.get("qb_guard", {}).get("ttl_hours", 3)) * 3600.0
    timeout = float(config.get("qb_guard", {}).get("timeout", 15))
    return cached_dataframe("nfl_injuries_espn",
                            lambda: _flatten(_fetch_injuries(timeout)),
                            ttl_seconds=ttl, force_refresh=force_refresh)


# --- pure pieces (tested directly) -------------------------------------------

def primary_passers(pbp: pd.DataFrame) -> dict[str, str]:
    """team abbr -> the passer with the most pass plays ("P.Mahomes" form).
    Empty dict when the frame predates the passer column (degrade)."""
    if pbp.empty or "passer_player_name" not in pbp.columns             or "posteam" not in pbp.columns:
        return {}
    df = pbp[pbp["passer_player_name"].notna() & pbp["posteam"].notna()]
    if df.empty:
        return {}
    counts = (df.groupby(["posteam", "passer_player_name"]).size()
                .rename("n").reset_index()
                .sort_values("n", ascending=False))
    return dict(counts.drop_duplicates("posteam")[["posteam", "passer_player_name"]].values)


def _name_matches(pbp_name: str, espn_name: str) -> bool:
    """"P.Mahomes" vs "Patrick Mahomes" -> last name + first initial."""
    if not pbp_name or not espn_name:
        return False
    pbp_name = pbp_name.strip()
    if "." not in pbp_name:
        return pbp_name.lower() == espn_name.lower()
    initial, _, last = pbp_name.partition(".")
    tokens = [t for t in espn_name.replace(".", " ").split()
              if t.lower() not in _SUFFIXES]
    if len(tokens) < 2:
        return False
    return (tokens[-1].lower() == last.strip().lower()
            and tokens[0][:1].lower() == initial.strip()[:1].lower())


def qb_flags(injuries: pd.DataFrame, primary: dict[str, str],
             hold_statuses: list[str]) -> dict[str, str]:
    """team abbr -> reason string, for teams whose PRIMARY passer carries a
    hold-worthy status. Third-stringers on IR never fire (they aren't the
    primary passer)."""
    if injuries.empty or not primary:
        return {}
    held = {s.lower() for s in hold_statuses}
    flags: dict[str, str] = {}
    qbs = injuries[injuries["position"] == "QB"]
    for _, row in qbs.iterrows():
        if str(row["status"]).lower() not in held:
            continue
        prim = primary.get(row["team"])
        if prim and _name_matches(prim, str(row["athlete"])):
            flags[row["team"]] = (f"{row['team']} primary QB {row['athlete']} "
                                  f"listed {row['status']}")
    return flags


def compute_flags(season: int, config: dict) -> tuple[dict[str, str], str | None]:
    """(flags, note). Never raises: any failure returns ({}, why) so the
    slate proceeds unguarded rather than dying on ESPN."""
    gcfg = config.get("qb_guard", {})
    if not gcfg.get("enabled", True):
        return {}, "qb guard disabled"
    try:
        from mlb_value_bot.football.data import nfl_client

        primary = primary_passers(nfl_client.pbp(season, config))
        if not primary:   # early season / pre-refresh parquet: prior year
            primary = primary_passers(nfl_client.pbp(season - 1, config))
        if not primary:
            return {}, "qb guard unavailable: no passer data in pbp yet"
        injuries = league_injuries(config)
        if injuries.empty:
            return {}, "qb guard unavailable: ESPN injuries feed empty"
        statuses = gcfg.get("hold_statuses",
                            ["Out", "Doubtful", "Injured Reserve",
                             "Physically Unable to Perform"])
        flags = qb_flags(injuries, primary, statuses)
        if flags:
            log.info("qb guard flags: %s", flags)
        return flags, None
    except Exception as exc:  # noqa: BLE001 — degrade, never kill the slate
        log.warning("qb guard unavailable: %s", exc)
        return {}, f"qb guard unavailable: {exc}"
