"""Football market views + blend + EV — mostly pure (de-vig via the shared
ev_calculator, push-aware EV, sharp consensus per market).

A MarketView is one game's one market as the model prices it: the bet-book
(DraftKings) line + prices, the de-vigged probability at that line, the sharp
consensus (median line + de-vigged prob across configured sharp books), and
the market-implied anchor mean the divergence guard measures against.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from mlb_value_bot.analysis.ev_calculator import american_to_decimal, devig
from mlb_value_bot.football.analysis.projections import market_anchor_mean
from mlb_value_bot.football.data.football_odds import FootballGameOdds


@dataclass
class MarketView:
    market: str                  # "spread" | "total"
    book: str
    line: float                  # home line (spread) or the total
    price_a: int                 # home / over price at the bet book
    price_b: int                 # away / under price
    devig_p_a: float             # de-vigged P(home covers) / P(over) at the book line
    sharp_line: float | None
    sharp_devig_p_a: float | None
    anchor_mean: float | None    # market-implied mean margin/total (sharp preferred)
    n_sharp_books: int


def _devig_pair(price_a: int, price_b: int, method: str) -> float:
    from mlb_value_bot.analysis.ev_calculator import american_to_implied

    p = devig([american_to_implied(price_a), american_to_implied(price_b)], method)
    return p[0]


def market_view(game: FootballGameOdds, market: str, config: dict,
                sigma: float) -> MarketView | None:
    """Build the view for one market, None when the bet book has no quote."""
    odds_cfg = config.get("odds_api", {})
    bet_book = (odds_cfg.get("bet_bookmaker") or "draftkings").lower()
    sharp_books = [b.lower() for b in odds_cfg.get("sharp_bookmakers", [])]
    method = config.get("ev", {}).get("devig_method", "power")

    quotes = game.spreads if market == "spread" else game.totals
    q = quotes.get(bet_book)
    if q is None:
        return None
    if market == "spread":
        line, price_a, price_b = q.home_line, q.home_price, q.away_price
    else:
        line, price_a, price_b = q.line, q.over_price, q.under_price
    devig_p_a = _devig_pair(price_a, price_b, method)

    sharp_lines, sharp_ps = [], []
    for book in sharp_books:
        sq = quotes.get(book)
        if sq is None:
            continue
        if market == "spread":
            sharp_lines.append(sq.home_line)
            sharp_ps.append(_devig_pair(sq.home_price, sq.away_price, method))
        else:
            sharp_lines.append(sq.line)
            sharp_ps.append(_devig_pair(sq.over_price, sq.under_price, method))

    sharp_line = median(sharp_lines) if sharp_lines else None
    sharp_p = median(sharp_ps) if sharp_ps else None
    kind = "total" if market == "total" else "spread"
    if sharp_line is not None and sharp_p is not None:
        anchor = market_anchor_mean(sharp_line, sharp_p, sigma, kind)
    else:
        anchor = market_anchor_mean(line, devig_p_a, sigma, kind)

    return MarketView(market=market, book=bet_book, line=line,
                      price_a=price_a, price_b=price_b, devig_p_a=devig_p_a,
                      sharp_line=sharp_line, sharp_devig_p_a=sharp_p,
                      anchor_mean=round(anchor, 2) if anchor is not None else None,
                      n_sharp_books=len(sharp_lines))


def blend_probability(p_model: float, p_market: float, config: dict) -> float:
    """final = market_blend x model + (1 - market_blend) x market — the same
    market-heavy anchoring philosophy as MLB (config projections.market_blend,
    0.35 to start)."""
    w = float(config.get("projections", {}).get("market_blend", 0.35))
    return w * p_model + (1.0 - w) * p_market


def ev_with_push(p_win: float, p_push: float, american: int) -> float:
    """EV per 1u staked with push mass returning the stake: p_win x (dec-1)
    - p_loss. A push contributes zero either way."""
    dec = american_to_decimal(american)
    p_loss = max(0.0, 1.0 - p_win - p_push)
    return p_win * (dec - 1.0) - p_loss
