"""Football odds via The Odds API — spreads + totals for NFL and NCAAF.

Reuses the shared OddsClient for its resilient HTTP layer (tenacity retry,
quota-header logging, 401/429 surfacing) but does its OWN event parsing:
the MLB client's parse runs constants.normalize_team, whose nickname fallback
would silently map "New York Giants" -> "San Francisco Giants" and
"Arizona Cardinals" -> "St. Louis Cardinals". Football name resolution instead
goes through football.data.teams (NFL abbr map / CFB school matcher), applied
by the CALLER — this module keeps the raw Odds API display names so the CFB
matcher (which needs CFBD's team list) can be injected at pipeline level.

One request per league covers ALL games and both markets across all configured
books (billing: markets x regions = 4 credits per league per call on us,uk).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mlb_value_bot.data.odds_client import OddsClient
from mlb_value_bot.utils import get_logger

log = get_logger("football.data.odds")


@dataclass
class SpreadQuote:
    """One book's spread: the HOME line (away is its mirror) + both prices."""
    home_line: float
    home_price: int
    away_price: int


@dataclass
class TotalQuote:
    """One book's total: the number + over/under prices."""
    line: float
    over_price: int
    under_price: int


@dataclass
class FootballGameOdds:
    event_id: str
    commence_time: str            # ISO8601 UTC
    home_name_raw: str            # Odds API display names; canonical resolution
    away_name_raw: str            # is the caller's job (league-specific)
    spreads: dict[str, SpreadQuote] = field(default_factory=dict)   # by bookmaker key
    totals: dict[str, TotalQuote] = field(default_factory=dict)


def _client_for_league(league: str, config: dict) -> OddsClient:
    leagues = config.get("leagues", {})
    sport_key = leagues.get(league, {}).get("sport_key")
    if not sport_key:
        raise ValueError(f"No sport_key configured for league '{league}'")
    odds_cfg = dict(config.get("odds_api", {}))
    odds_cfg["sport_key"] = sport_key
    # OddsClient reads everything it needs from config["odds_api"].
    return OddsClient(config={"odds_api": odds_cfg})


def _parse_event(ev: dict) -> FootballGameOdds | None:
    home_raw, away_raw = ev.get("home_team"), ev.get("away_team")
    if not home_raw or not away_raw:
        log.warning("Skipping event with missing teams: %s", ev.get("id"))
        return None
    game = FootballGameOdds(
        event_id=ev.get("id", ""),
        commence_time=ev.get("commence_time", ""),
        home_name_raw=home_raw,
        away_name_raw=away_raw,
    )
    for book in ev.get("bookmakers", []):
        book_key = book.get("key", "?")
        for market in book.get("markets", []):
            outcomes = market.get("outcomes", [])
            if market.get("key") == "spreads":
                home = next((o for o in outcomes if o.get("name") == home_raw), None)
                away = next((o for o in outcomes if o.get("name") == away_raw), None)
                if home and away and home.get("point") is not None \
                        and home.get("price") is not None and away.get("price") is not None:
                    game.spreads[book_key] = SpreadQuote(
                        home_line=float(home["point"]),
                        home_price=int(home["price"]),
                        away_price=int(away["price"]),
                    )
            elif market.get("key") == "totals":
                over = next((o for o in outcomes if o.get("name") == "Over"), None)
                under = next((o for o in outcomes if o.get("name") == "Under"), None)
                if over and under and over.get("point") is not None \
                        and over.get("price") is not None and under.get("price") is not None:
                    game.totals[book_key] = TotalQuote(
                        line=float(over["point"]),
                        over_price=int(over["price"]),
                        under_price=int(under["price"]),
                    )
    return game


def fetch_league_odds(league: str, config: dict) -> list[FootballGameOdds]:
    """Current spreads + totals for every listed game in one league.

    One HTTP call. Games with no parseable market at any book are kept (the
    pipeline decides what's evaluable); events missing teams are dropped.
    """
    client = _client_for_league(league, config)
    if not client.api_key:
        from mlb_value_bot.data.odds_client import OddsAPIError
        raise OddsAPIError("No ODDS_API_KEY found. Copy .env.example to .env and set the key.")

    url = f"{client.base_url}/sports/{client.sport_key}/odds"
    odds_cfg = config.get("odds_api", {})
    params = {
        "apiKey": client.api_key,
        "regions": odds_cfg.get("regions", "us,uk"),
        "markets": odds_cfg.get("markets", "spreads,totals"),
        "oddsFormat": odds_cfg.get("odds_format", "american"),
    }
    books = odds_cfg.get("bookmakers") or []
    if books:
        params["bookmakers"] = ",".join(books)

    resp = client._get(url, params)  # noqa: SLF001 -- deliberate reuse of the shared retry/quota layer
    events = resp.json()
    games = [g for g in (_parse_event(ev) for ev in events) if g is not None]
    log.info("Parsed odds for %d %s games", len(games), league.upper())
    return games
