"""Totals (over/under) market intelligence -- the totals analog of market_intel.

Totals has its OWN line, its OWN de-vig, and its OWN sharp consensus, none of
which transfer from the moneyline. From the per-book totals payload we compute:
  * the bet-book's line + over/under prices, de-vigged -> market P(over);
  * sharp vs square consensus on the LINE and on de-vigged P(over);
  * the sharp totals closing line (Pinnacle-preferred) for CLV.

Degrades gracefully: a book that didn't post totals is skipped; no sharp book ->
consensus None.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from mlb_value_bot.analysis.ev_calculator import devigged_market_probs
from mlb_value_bot.utils import get_logger

log = get_logger("analysis.totals_market")


def book_totals(book: dict) -> tuple[float, int, int] | None:
    """(line, over_american, under_american) for one book's totals market, or None."""
    for market in book.get("markets") or []:
        if (market.get("key") or "") != "totals":
            continue
        line = over = under = None
        for o in market.get("outcomes") or []:
            name = (o.get("name") or "").lower()
            price, point = o.get("price"), o.get("point")
            if price is None or point is None:
                continue
            if name == "over":
                line, over = float(point), int(price)
            elif name == "under":
                line, under = float(point), int(price)
        if line is not None and over is not None and under is not None:
            return line, over, under
    return None


@dataclass
class TotalsMarketIntel:
    bet_line: float | None
    bet_over_price: int | None
    bet_under_price: int | None
    bet_devig_over: float | None        # de-vigged P(over) at the bet book
    best_over_price: int | None         # best (highest) over price across books
    best_under_price: int | None
    sharp_line: float | None
    sharp_devig_over: float | None
    square_line: float | None
    square_devig_over: float | None
    n_sharp: int
    n_square: int
    n_total: int

    @property
    def available(self) -> bool:
        return self.bet_line is not None and self.bet_devig_over is not None

    @property
    def sharp_available(self) -> bool:
        return self.n_sharp > 0 and self.sharp_devig_over is not None

    @property
    def sharp_minus_square(self) -> float | None:
        if self.sharp_devig_over is None or self.square_devig_over is None:
            return None
        return self.sharp_devig_over - self.square_devig_over

    def disagreement_with(self, our_p_over: float) -> float | None:
        """Our P(over) minus the sharps' P(over). + = we're higher on the over."""
        if not self.sharp_available:
            return None
        return our_p_over - self.sharp_devig_over


def compute_totals_market(all_books, bet_book, sharp_books, square_books,
                          devig_method="power") -> TotalsMarketIntel:
    bet_book = (bet_book or "").lower() or None
    sharp_set = {b.lower() for b in (sharp_books or [])}
    square_set = {b.lower() for b in (square_books or [])}

    bet = None
    best_over = best_under = None
    sharp_overs, square_overs, all_lines = [], [], []
    sharp_lines, square_lines = [], []

    for book in all_books or []:
        key = (book.get("key") or "").lower()
        bt = book_totals(book)
        if not bt:
            continue
        line, over, under = bt
        try:
            p_over, _ = devigged_market_probs(over, under, devig_method)
        except Exception:  # noqa: BLE001
            continue
        all_lines.append(line)
        if best_over is None or over > best_over:
            best_over = over
        if best_under is None or under > best_under:
            best_under = under
        if bet_book and key == bet_book:
            bet = (line, over, under, p_over)
        if key in sharp_set:
            sharp_overs.append(p_over)
            sharp_lines.append(line)
        if key in square_set:
            square_overs.append(p_over)
            square_lines.append(line)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return TotalsMarketIntel(
        bet_line=(bet[0] if bet else None),
        bet_over_price=(bet[1] if bet else None),
        bet_under_price=(bet[2] if bet else None),
        bet_devig_over=(bet[3] if bet else None),
        best_over_price=best_over, best_under_price=best_under,
        sharp_line=_mean(sharp_lines), sharp_devig_over=_mean(sharp_overs),
        square_line=_mean(square_lines), square_devig_over=_mean(square_overs),
        n_sharp=len(sharp_overs), n_square=len(square_overs), n_total=len(all_lines),
    )


@dataclass
class SharpTotalsLine:
    book: str
    line: float
    over_price: int
    under_price: int
    devig_over: float


def sharp_totals_close(all_books, priority, devig_method="power") -> SharpTotalsLine | None:
    """First priority book (Pinnacle-preferred) that posted totals, as a sharp
    closing line. Used to grade totals CLV."""
    by_book = {}
    for book in all_books or []:
        bt = book_totals(book)
        if bt:
            by_book[(book.get("key") or "").lower()] = bt
    for name in priority or []:
        bt = by_book.get(name.lower())
        if not bt:
            continue
        line, over, under = bt
        try:
            p_over, _ = devigged_market_probs(over, under, devig_method)
        except Exception:  # noqa: BLE001
            continue
        return SharpTotalsLine(name.lower(), line, over, under, round(p_over, 4))
    return None
