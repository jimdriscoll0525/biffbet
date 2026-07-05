"""Stability flags — PURE, and flags ONLY.

HARD INVARIANT (the MLB consolidation rule): nothing in this module adjusts an
EV number. Sharp-fade / reverse-line-movement is measured here and NOTED; its
one-and-only EV magnitude application lives in pipeline_football's adjusted-EV
step. Tests lock this by asserting these functions return flags, not deltas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mlb_value_bot.football.analysis.football_ev import MarketView

LABEL_STABLE = "stable"
LABEL_MODERATE = "moderate"
LABEL_FRAGILE = "fragile"


@dataclass
class Stability:
    label: str
    flags: list[str] = field(default_factory=list)          # fragile signals
    rlm_note: str | None = None                             # informational only
    sharp_gap_pp: float | None = None                       # sharp - book, prob pts (side A)


def sharp_gap_pp(view: MarketView) -> float | None:
    """Sharp-vs-book de-vigged probability gap for side A (home/over), in
    probability points. Positive = sharps more bullish on side A than the bet
    book's price implies."""
    if view.sharp_devig_p_a is None:
        return None
    return (view.sharp_devig_p_a - view.devig_p_a) * 100.0


def assess(view: MarketView | None, *, games_min: float | None,
           epa_available: bool, weather_available: bool, outdoor_total: bool,
           ol_proxy_only: bool, config: dict) -> Stability:
    """Collect fragile signals + the RLM note for one market evaluation.

    games_min: the smaller of the two teams' games played (None = unknown).
    """
    flags: list[str] = []
    if not epa_available:
        flags.append("EPA inputs missing -> projection leans on league base")
    if games_min is not None and games_min < 4:
        flags.append(f"small sample ({games_min:.0f} games) — early-season prior blend active")
    if outdoor_total and not weather_available:
        flags.append("outdoor total without weather feed")
    if ol_proxy_only:
        flags.append("OL layer on sack/YPC proxy only")

    rlm_note = None
    gap = sharp_gap_pp(view) if view is not None else None
    if gap is not None and abs(gap) >= 1.0:
        side_a = "over" if view.market == "total" else "home"
        direction = f"toward {side_a}" if gap > 0 else f"against {side_a}"
        rlm_note = (f"sharp consensus {abs(gap):.1f}pp {direction} vs {view.book} "
                    f"(line {view.sharp_line} vs {view.line})")

    label = LABEL_STABLE if not flags else (LABEL_MODERATE if len(flags) == 1 else LABEL_FRAGILE)
    return Stability(label=label, flags=flags, rlm_note=rlm_note,
                     sharp_gap_pp=round(gap, 2) if gap is not None else None)
