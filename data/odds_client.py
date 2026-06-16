"""The Odds API (v4) wrapper for MLB moneyline (h2h) odds.

Responsibilities:
  * Pull h2h odds for the configured books/regions in American format.
  * For each game, surface the BEST (highest) price available per side across
    the configured books — that's the line you'd actually bet.
  * Respect and LOG the remaining request quota from response headers.
  * Retry transient failures with backoff; surface rate-limit (429) clearly.

Docs: https://the-odds-api.com/liveapi/guides/v4/
"""
from __future__ import annotations

from dataclasses import dataclass, field

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mlb_value_bot.constants import normalize_team
from mlb_value_bot.utils import get_env, get_logger, load_config

log = get_logger("data.odds_client")


class OddsAPIError(RuntimeError):
    """Raised for non-retryable Odds API problems (auth, bad request)."""


class OddsAPIRateLimit(OddsAPIError):
    """Raised when the API returns 429 (quota/throttle)."""


@dataclass
class SidePrice:
    """Best available price for one side of a game."""
    team: str
    american_odds: int
    bookmaker: str


@dataclass
class GameOdds:
    """Best-priced moneyline for a single game."""
    event_id: str
    commence_time: str          # ISO8601 UTC from the API
    home_team: str              # canonical
    away_team: str              # canonical
    home: SidePrice | None = None
    away: SidePrice | None = None
    all_books: list[dict] = field(default_factory=list)  # raw, for opening-line capture

    def price_for(self, side: str) -> SidePrice | None:
        return self.home if side == "home" else self.away


class OddsClient:
    """Thin, resilient client around the Odds API h2h endpoint."""

    def __init__(self, api_key: str | None = None, config: dict | None = None) -> None:
        self.config = config or load_config()
        self._odds_cfg = self.config.get("odds_api", {})
        self.api_key = api_key or get_env("ODDS_API_KEY")
        self.base_url = self._odds_cfg.get("base_url", "https://api.the-odds-api.com/v4")
        self.sport_key = self._odds_cfg.get("sport_key", "baseball_mlb")
        self.timeout = float(self._odds_cfg.get("request_timeout_seconds", 20))
        self.session = requests.Session()

    # -- HTTP with retry ------------------------------------------------------
    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    )
    def _get(self, url: str, params: dict) -> requests.Response:
        from mlb_value_bot.data import fetch_ledger
        log.debug("GET %s params=%s", url, {k: v for k, v in params.items() if k != "apiKey"})
        resp = self.session.get(url, params=params, timeout=self.timeout)
        self._log_quota(resp)
        if resp.status_code == 429:
            fetch_ledger.record("odds_api", "rate_limited", http_status=429)
            raise OddsAPIRateLimit("Odds API rate limit / quota exhausted (HTTP 429)")
        if resp.status_code == 401:
            fetch_ledger.record("odds_api", "http_error", http_status=401)
            raise OddsAPIError("Odds API rejected the key (HTTP 401) — check ODDS_API_KEY")
        if resp.status_code >= 400:
            fetch_ledger.record("odds_api", "http_error", http_status=resp.status_code)
            raise OddsAPIError(f"Odds API error {resp.status_code}: {resp.text[:300]}")
        fetch_ledger.record("odds_api", "ok", http_status=resp.status_code)
        return resp

    @staticmethod
    def _log_quota(resp: requests.Response) -> None:
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        if remaining is not None:
            log.info("Odds API quota - remaining=%s used=%s", remaining, used)

    # -- Public API -----------------------------------------------------------
    def get_odds(self) -> list[GameOdds]:
        """Fetch current MLB h2h odds and return best price per side per game."""
        if not self.api_key:
            raise OddsAPIError(
                "No ODDS_API_KEY found. Copy .env.example to .env and set the key."
            )

        url = f"{self.base_url}/sports/{self.sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self._odds_cfg.get("regions", "us"),
            "markets": self._odds_cfg.get("markets", "h2h"),
            "oddsFormat": self._odds_cfg.get("odds_format", "american"),
        }
        books = self._odds_cfg.get("bookmakers") or []
        if books:
            params["bookmakers"] = ",".join(books)

        resp = self._get(url, params)
        try:
            events = resp.json()
        except ValueError as exc:
            raise OddsAPIError(f"Could not parse Odds API JSON: {exc}") from exc

        games = [self._parse_event(ev) for ev in events]
        parsed = [g for g in games if g is not None]
        log.info("Parsed odds for %d MLB games", len(parsed))
        return parsed

    # -- Parsing --------------------------------------------------------------
    def _parse_event(self, ev: dict) -> GameOdds | None:
        home_raw = ev.get("home_team")
        away_raw = ev.get("away_team")
        home = normalize_team(home_raw)
        away = normalize_team(away_raw)
        if not home or not away:
            log.warning("Skipping event with missing teams: %s", ev.get("id"))
            return None

        game = GameOdds(
            event_id=ev.get("id", ""),
            commence_time=ev.get("commence_time", ""),
            home_team=home,
            away_team=away,
            all_books=ev.get("bookmakers", []),
        )

        # bet_bookmaker pins the bet-price decision to a single book (so the
        # displayed line is the price you'll actually bet). When unset we fall
        # back to the old "best price across all listed books" line-shopping
        # behavior. Other books still populate all_books for sharp/square intel.
        bet_book = (self._odds_cfg.get("bet_bookmaker") or "").strip().lower() or None

        best_home: SidePrice | None = None
        best_away: SidePrice | None = None
        for book in ev.get("bookmakers", []):
            book_key = book.get("key", "?")
            if bet_book is not None and book_key != bet_book:
                continue
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    team = normalize_team(outcome.get("name"))
                    price = outcome.get("price")
                    if team is None or price is None:
                        continue
                    cand = SidePrice(team=team, american_odds=int(price), bookmaker=book_key)
                    if team == home and (best_home is None or self._is_better(price, best_home.american_odds)):
                        best_home = cand
                    elif team == away and (best_away is None or self._is_better(price, best_away.american_odds)):
                        best_away = cand

        game.home = best_home
        game.away = best_away
        return game

    @staticmethod
    def _is_better(candidate: int, current: int) -> bool:
        """Higher payout is better for the bettor.

        For American odds the bettor-preferred ordering is simply numeric:
        +150 > +120 > -110 > -140. So a plain `>` comparison is correct.
        """
        return candidate > current
