"""Odds conversions, de-vigging, expected value, and Kelly staking.

This module is pure math (no I/O) so it's easy to reason about and unit test.
Accuracy of EV is a top priority per the project spec, so every conversion is
written explicitly rather than leaning on a library.

Definitions used throughout:
  * American odds: +150 means win 150 on 100 staked; -120 means stake 120 to win 100.
  * Decimal odds: total return per 1 unit staked (stake included). +150 -> 2.50.
  * Implied probability (raw): 1 / decimal_odds. Sums to >1 across a market (the vig).
  * De-vigged ("fair") probability: vig removed so the two sides sum to 1.
  * EV%: expected profit per 1 unit staked = model_prob * decimal_odds - 1.
"""
from __future__ import annotations

from dataclasses import dataclass


# --- Odds conversions --------------------------------------------------------
def american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal odds (total return per unit staked)."""
    american = float(american)
    if american == 0:
        raise ValueError("American odds cannot be 0")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds back to (rounded) American odds."""
    if decimal <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    return int(round(-100.0 / (decimal - 1.0)))


def american_to_implied(american: int | float) -> float:
    """Raw implied probability from American odds (includes vig)."""
    return 1.0 / american_to_decimal(american)


def decimal_to_implied(decimal: float) -> float:
    """Raw implied probability from decimal odds (includes vig)."""
    return 1.0 / decimal


# --- De-vigging --------------------------------------------------------------
def devig_proportional(raw_probs: list[float]) -> list[float]:
    """Remove vig by simple normalization: p_i / sum(p).

    Fast and unbiased toward favorites/longshots — it just rescales.
    """
    total = sum(raw_probs)
    if total <= 0:
        raise ValueError("Sum of raw probabilities must be positive")
    return [p / total for p in raw_probs]


def devig_power(raw_probs: list[float], tol: float = 1e-9, max_iter: int = 200) -> list[float]:
    """Remove vig with the power method: find k s.t. sum(p_i**k) == 1.

    The power method shrinks the overround multiplicatively in log-space, which
    better preserves the favorite-longshot structure of the market than simple
    proportional scaling. Solved by bisection on the exponent k.
    """
    probs = [max(min(p, 1 - 1e-12), 1e-12) for p in raw_probs]
    if abs(sum(probs) - 1.0) < tol:
        return probs

    def overround(k: float) -> float:
        return sum(p ** k for p in probs) - 1.0

    # sum at k=1 is >1 (vig). Increasing k lowers the sum -> bracket [1, hi].
    lo, hi = 1.0, 1.0
    # Expand hi until the sum drops below 1.
    for _ in range(100):
        hi *= 1.5
        if overround(hi) < 0:
            break
    # If even k=1 already <=1 (no vig / negative), search downward instead.
    if overround(1.0) < 0:
        lo, hi = 1e-6, 1.0

    k = 1.0
    for _ in range(max_iter):
        k = (lo + hi) / 2.0
        val = overround(k)
        if abs(val) < tol:
            break
        if val > 0:
            lo = k
        else:
            hi = k
    return [p ** k for p in probs]


def devig(raw_probs: list[float], method: str = "power") -> list[float]:
    """Dispatch to the configured de-vig method."""
    if method == "proportional":
        return devig_proportional(raw_probs)
    return devig_power(raw_probs)


def devigged_market_probs(
    home_american: int, away_american: int, method: str = "power"
) -> tuple[float, float]:
    """Fair (de-vigged) [home, away] probabilities from a two-way American market.

    The "fair" home probability is the sharpest publicly available estimate of
    the true win probability — used as the anchor the model blends toward.
    """
    home_dec = american_to_decimal(home_american)
    away_dec = american_to_decimal(away_american)
    fair = devig([decimal_to_implied(home_dec), decimal_to_implied(away_dec)], method=method)
    return fair[0], fair[1]


# --- EV + Kelly --------------------------------------------------------------
def ev_pct(model_prob: float, decimal_odds: float) -> float:
    """Expected value as a fraction of stake: p*dec - 1.

    e.g. model_prob=0.55, decimal=2.0 -> 0.10 (a 10% edge).
    """
    return model_prob * decimal_odds - 1.0


def kelly_fraction(
    model_prob: float,
    decimal_odds: float,
    kelly_multiplier: float = 0.25,
    cap: float = 0.02,
) -> float:
    """Fractional Kelly stake as a fraction of bankroll, floored at 0, capped.

    Full Kelly: f* = (b*p - q) / b, where b = decimal-1, q = 1-p.
    We then apply `kelly_multiplier` (default 0.25 = quarter-Kelly) and cap the
    result at `cap` (default 2% of bankroll). Returns 0.0 when there's no edge.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    full = (b * model_prob - q) / b
    if full <= 0:
        return 0.0
    return min(full * kelly_multiplier, cap)


@dataclass
class SideEvaluation:
    """Full EV picture for one side of a game."""
    side: str                    # "home" | "away"
    american_odds: int
    decimal_odds: float
    model_prob: float
    market_prob_raw: float       # raw implied (with vig) for this side
    market_prob_devigged: float  # fair prob for this side
    ev_pct: float
    kelly_stake: float           # fraction of bankroll (already capped)


def evaluate_sides(
    model_home_prob: float,
    home_american: int,
    away_american: int,
    devig_method: str = "power",
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.02,
) -> dict[str, SideEvaluation]:
    """Build SideEvaluation for both home and away from a two-way market.

    Returns {"home": SideEvaluation, "away": SideEvaluation}.
    """
    home_dec = american_to_decimal(home_american)
    away_dec = american_to_decimal(away_american)
    raw_home = decimal_to_implied(home_dec)
    raw_away = decimal_to_implied(away_dec)

    fair_home, fair_away = devig([raw_home, raw_away], method=devig_method)

    model_away_prob = 1.0 - model_home_prob

    home_eval = SideEvaluation(
        side="home",
        american_odds=int(home_american),
        decimal_odds=home_dec,
        model_prob=model_home_prob,
        market_prob_raw=raw_home,
        market_prob_devigged=fair_home,
        ev_pct=ev_pct(model_home_prob, home_dec),
        kelly_stake=kelly_fraction(model_home_prob, home_dec, kelly_multiplier, kelly_cap),
    )
    away_eval = SideEvaluation(
        side="away",
        american_odds=int(away_american),
        decimal_odds=away_dec,
        model_prob=model_away_prob,
        market_prob_raw=raw_away,
        market_prob_devigged=fair_away,
        ev_pct=ev_pct(model_away_prob, away_dec),
        kelly_stake=kelly_fraction(model_away_prob, away_dec, kelly_multiplier, kelly_cap),
    )
    return {"home": home_eval, "away": away_eval}
