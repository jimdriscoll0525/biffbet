"""Sharp closing-line extraction (GriffBet only).

BiffBet stores only the best-price-across-books open/close; it never captured a
sharp closing line. GriffBet grades CLV against the SHARP close (the truest
price), so on each re-pricing run it pulls the sharp book's line per the
configured priority order (Pinnacle first). This is read-only over the same
GameOdds BiffBet already fetches -- no new feed.
"""
from __future__ import annotations

from dataclasses import dataclass

from mlb_value_bot.analysis.ev_calculator import devigged_market_probs
from mlb_value_bot.constants import normalize_team
from mlb_value_bot.data.odds_client import GameOdds
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.sharp_close")


@dataclass
class SharpLine:
    book: str
    home_line: int
    away_line: int
    home_prob: float   # de-vigged sharp home prob
    away_prob: float


def sharp_line_from_odds(
    game_odds: GameOdds,
    priority: list[str],
    devig_method: str = "power",
) -> SharpLine | None:
    """First book in `priority` that quoted BOTH sides, as a SharpLine.

    Pinnacle-preferred per config.sharp_close.priority. Returns None when no
    priority book quoted the game (caller falls back to best-available close /
    records the sharp close as unavailable).
    """
    if not game_odds or not priority:
        return None
    home, away = game_odds.home_team, game_odds.away_team

    # Index this game's per-book h2h prices once.
    by_book: dict[str, tuple[int, int]] = {}
    for book in game_odds.all_books or []:
        key = (book.get("key") or "").lower()
        if not key:
            continue
        for market in book.get("markets") or []:
            if (market.get("key") or "") != "h2h":
                continue
            hp = ap = None
            for outcome in market.get("outcomes") or []:
                team_norm = normalize_team(outcome.get("name"))
                price = outcome.get("price")
                if team_norm is None or price is None:
                    continue
                if team_norm == home:
                    hp = int(price)
                elif team_norm == away:
                    ap = int(price)
            if hp is not None and ap is not None:
                by_book[key] = (hp, ap)

    for name in priority:
        prices = by_book.get(name.lower())
        if prices is None:
            continue
        hp, ap = prices
        try:
            fair_home, fair_away = devigged_market_probs(hp, ap, devig_method)
        except Exception as exc:  # noqa: BLE001
            log.debug("sharp devig failed for %s: %s", name, exc)
            continue
        return SharpLine(book=name.lower(), home_line=hp, away_line=ap,
                         home_prob=fair_home, away_prob=fair_away)
    return None


def price_for_side(line_home: int | None, line_away: int | None, side: str) -> int | None:
    """Pick the home/away American price for a pick side."""
    if side == "home":
        return line_home
    if side == "away":
        return line_away
    return None
