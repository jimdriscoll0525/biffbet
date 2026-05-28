"""Sharp / square market intelligence from per-book moneyline data.

The Odds API returns moneyline prices from many books for the same game. We
group those books into "sharp" (Pinnacle, BetOnline, LowVig, ...) and
"square" (DraftKings, FanDuel, BetMGM, ...) per config.odds_api, de-vig
each side independently, and compute two consensus probabilities:

  * sharp consensus  = mean de-vigged home prob across the sharp books that
                       quoted both sides
  * square consensus = same, across the square books

Two derived signals the rest of the engine uses:

  * sharp_minus_square (+ = sharps like home more than squares) -- the bones
    of the "what do the smart people think the public is missing" indicator.
  * dispersion (stdev of devigged home prob across ALL books in pp) -- a
    flag for unsettled markets where books are visibly disagreeing.

A pick that the sharps clearly fade gets a confidence penalty in
pipeline.evaluate_game. If the disagreement is severe (> max_sharp_disagreement),
the pipeline skips it outright under the sanity guard -- same shape as the
raw-model-vs-market guard we added during the Sox/Braves live-game incident,
just with a different counterparty.

Graceful degradation: if either group has zero books quoting a given game
(API didn't return that book today, or you haven't configured any books in
that group), the consensus is None and the signal contributes 0.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from mlb_value_bot.analysis.ev_calculator import devigged_market_probs
from mlb_value_bot.constants import normalize_team
from mlb_value_bot.data.odds_client import GameOdds
from mlb_value_bot.utils import get_logger

log = get_logger("data.market_intel")


@dataclass
class MarketIntelligence:
    sharp_devig_home: float | None       # 0..1
    square_devig_home: float | None      # 0..1
    sharp_minus_square: float | None     # + = sharps like home more
    dispersion_pp: float | None          # stdev of devigged home prob across all books, in pp
    n_sharp_books: int
    n_square_books: int
    n_total_books: int

    @property
    def available(self) -> bool:
        """At least one sharp book quoted -- we have SOME sharp consensus signal."""
        return self.n_sharp_books > 0 and self.sharp_devig_home is not None

    def short_label(self) -> str:
        if not self.available:
            return "sharp signal unavailable"
        parts = [f"sharp {self.sharp_devig_home * 100:.1f}% (n={self.n_sharp_books})"]
        if self.square_devig_home is not None:
            parts.append(f"square {self.square_devig_home * 100:.1f}% (n={self.n_square_books})")
        if self.sharp_minus_square is not None:
            parts.append(f"S-Sq {self.sharp_minus_square * 100:+.1f}pp")
        if self.dispersion_pp is not None:
            parts.append(f"disp {self.dispersion_pp:.1f}pp")
        return " · ".join(parts)

    def disagreement_with(self, our_pick_home_prob: float) -> float | None:
        """Our pick-side prob minus sharps' pick-side prob, in probability points.

        Positive = we're MORE bullish on our side than the sharps are. A large
        positive number means we're fading sharp consensus -- the pipeline
        uses this for confidence + sanity-guard decisions.
        """
        if not self.available:
            return None
        return our_pick_home_prob - self.sharp_devig_home


def compute_market_intel(
    game_odds: GameOdds,
    sharp_books: list[str] | None,
    square_books: list[str] | None,
    devig_method: str = "power",
) -> MarketIntelligence:
    """Build a MarketIntelligence snapshot for one game from per-book pricing.

    Inputs are CASE-INSENSITIVE book keys; we normalize once here.
    """
    sharp_set = {b.lower() for b in (sharp_books or [])}
    square_set = {b.lower() for b in (square_books or [])}
    home = game_odds.home_team
    away = game_odds.away_team

    sharp_devigs: list[float] = []
    square_devigs: list[float] = []
    all_devigs: list[float] = []

    for book in game_odds.all_books or []:
        book_key = (book.get("key") or "").lower()
        if not book_key:
            continue
        for market in book.get("markets") or []:
            if (market.get("key") or "") != "h2h":
                continue
            home_price: int | None = None
            away_price: int | None = None
            for outcome in market.get("outcomes") or []:
                team_norm = normalize_team(outcome.get("name"))
                price = outcome.get("price")
                if team_norm is None or price is None:
                    continue
                if team_norm == home:
                    home_price = int(price)
                elif team_norm == away:
                    away_price = int(price)
            if home_price is None or away_price is None:
                continue
            try:
                fair_home, _fair_away = devigged_market_probs(home_price, away_price, devig_method)
            except Exception as exc:  # noqa: BLE001
                log.debug("devig failed for %s on %s: %s", book_key, game_odds.event_id, exc)
                continue
            all_devigs.append(fair_home)
            if book_key in sharp_set:
                sharp_devigs.append(fair_home)
            if book_key in square_set:
                square_devigs.append(fair_home)

    sharp_avg = sum(sharp_devigs) / len(sharp_devigs) if sharp_devigs else None
    square_avg = sum(square_devigs) / len(square_devigs) if square_devigs else None
    diff = (sharp_avg - square_avg) if (sharp_avg is not None and square_avg is not None) else None
    # pstdev for population (we have the whole set of returned books, not a sample).
    dispersion = statistics.pstdev(all_devigs) * 100.0 if len(all_devigs) >= 2 else None

    return MarketIntelligence(
        sharp_devig_home=sharp_avg,
        square_devig_home=square_avg,
        sharp_minus_square=diff,
        dispersion_pp=dispersion,
        n_sharp_books=len(sharp_devigs),
        n_square_books=len(square_devigs),
        n_total_books=len(all_devigs),
    )
