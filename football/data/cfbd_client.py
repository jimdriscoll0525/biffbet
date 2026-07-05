"""CollegeFootballData API v2 client (FBS stats, PPA, SP+, games, venues).

Key: CFBD_API_KEY in .env (required on every tier). The FREE tier is a
1,000-calls-per-MONTH quota, so this client is deliberately stingy:
  * every season-shaped pull (stats/PPA/SP+/returning/venues/teams) is cached
    for cfbd.season_stats_ttl_days (default 7d) via the shared parquet cache,
  * games are cached for a few hours (scores move on Saturdays),
  * every real HTTP call is counted + any quota-ish response header is logged,
    so `data-status` shows the burn rate long before the wall.

All responses are flattened to DataFrames before caching (parquet can't hold
nested dicts).
"""
from __future__ import annotations

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mlb_value_bot.data.cache import cached_dataframe
from mlb_value_bot.utils import get_env, get_logger

log = get_logger("football.data.cfbd")


class CfbdError(RuntimeError):
    """Non-retryable CFBD problems (missing/bad key, bad request, quota out)."""


class CfbdClient:
    def __init__(self, config: dict) -> None:
        cfg = config.get("cfbd", {})
        self.base_url = cfg.get("base_url", "https://apinext.collegefootballdata.com")
        self.timeout = float(cfg.get("request_timeout_seconds", 30))
        self.season_ttl = float(cfg.get("season_stats_ttl_days", 7)) * 86400.0
        self.games_ttl = 3 * 3600.0
        self.api_key = get_env("CFBD_API_KEY")
        self.session = requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # -- HTTP with retry -------------------------------------------------------
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _get(self, path: str, params: dict | None = None) -> list | dict:
        from mlb_value_bot.data import fetch_ledger

        if not self.api_key:
            raise CfbdError("No CFBD_API_KEY found. Add it to .env (free key at collegefootballdata.com).")
        resp = self.session.get(
            f"{self.base_url}{path}",
            params=params or {},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        for header, value in resp.headers.items():
            if "call" in header.lower() or "ratelimit" in header.lower():
                log.info("CFBD quota header %s=%s", header, value)
        if resp.status_code == 401:
            fetch_ledger.record("cfbd", "http_error", http_status=401)
            raise CfbdError("CFBD rejected the key (HTTP 401) -- check CFBD_API_KEY")
        if resp.status_code == 429:
            fetch_ledger.record("cfbd", "rate_limited", http_status=429)
            raise CfbdError("CFBD monthly quota exhausted / throttled (HTTP 429)")
        if resp.status_code >= 400:
            fetch_ledger.record("cfbd", "http_error", http_status=resp.status_code)
            raise CfbdError(f"CFBD error {resp.status_code} on {path}: {resp.text[:300]}")
        fetch_ledger.record("cfbd", "ok", http_status=resp.status_code)
        return resp.json()

    def _frame(self, key: str, path: str, params: dict | None, ttl: float,
               force_refresh: bool = False) -> pd.DataFrame:
        return cached_dataframe(
            key,
            lambda: pd.json_normalize(self._get(path, params)),
            ttl_seconds=ttl,
            force_refresh=force_refresh,
        )

    # -- Public pulls (each = 1 quota call on cache miss) ------------------------
    def fbs_teams(self, year: int) -> pd.DataFrame:
        """FBS team list: school, mascot, conference — the percentile pool +
        the CFB name matcher's input + the P4/G5 classification source."""
        return self._frame(f"cfbd_fbs_teams_{year}", "/teams/fbs", {"year": year}, self.season_ttl)

    def season_stats(self, year: int) -> pd.DataFrame:
        """Long-form team season stats (team, statName, statValue)."""
        return self._frame(f"cfbd_season_stats_{year}", "/stats/season", {"year": year}, self.season_ttl)

    def ppa_teams(self, year: int) -> pd.DataFrame:
        """Team PPA (EPA-like): offense/defense x passing/rushing, flattened."""
        return self._frame(f"cfbd_ppa_teams_{year}", "/ppa/teams", {"year": year}, self.season_ttl)

    def sp_ratings(self, year: int) -> pd.DataFrame:
        """SP+ ratings — the competition-level adjustment source."""
        return self._frame(f"cfbd_sp_{year}", "/ratings/sp", {"year": year}, self.season_ttl)

    def returning_production(self, year: int) -> pd.DataFrame:
        """Returning production — the CFB OL-continuity proxy."""
        return self._frame(f"cfbd_returning_{year}", "/player/returning", {"year": year}, self.season_ttl)

    def venues(self) -> pd.DataFrame:
        """Stadium coordinates + dome flag for CFB weather."""
        return self._frame("cfbd_venues", "/venues", None, self.season_ttl)

    def games(self, year: int, week: int | None = None, season_type: str = "regular",
              force_refresh: bool = False) -> pd.DataFrame:
        """FBS games with scores (grading) + venue ids (weather)."""
        params: dict = {"year": year, "seasonType": season_type, "classification": "fbs"}
        key = f"cfbd_games_{year}_{season_type}_{week if week is not None else 'all'}"
        if week is not None:
            params["week"] = week
        return self._frame(key, "/games", params, self.games_ttl, force_refresh=force_refresh)
