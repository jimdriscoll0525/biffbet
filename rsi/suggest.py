"""Turn a finding (pool x dim x value x direction) into a plain-English
proposal and, for the combos we understand, a concrete config overlay.

The rule table is deliberately small and conservative: an overlay only exists
where the mechanism is obvious (a losing price band -> a price filter, a
losing weather bucket -> a weather coefficient). Everything else is an
`insight` (overlay None) so Jim still sees it, but nothing can be promoted
from it without a human writing the change.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from mlb_value_bot.rsi.segments import PRICE_RANGES

_CLV_UNIT = {"clv_pct": "%", "clv_pp": "pp", "clv_blended_vs_sharp": "pp"}


@dataclass
class Suggestion:
    kind: str                       # overlay | insight
    title: str
    description: str
    suggested_change: str | None
    overlay: dict | None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "title": self.title, "description": self.description,
                "suggested_change": self.suggested_change, "overlay": self.overlay}


def _cfg(base: dict, dotted: str, default=None):
    node: Any = base or {}
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _fmt_pct(x, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:+.{nd}f}%"


def _fmt_hit(x) -> str:
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _fmt_clv(stats: dict | None) -> str:
    if not stats or stats.get("avg_clv") is None:
        return "CLV n/a"
    unit = _CLV_UNIT.get(stats.get("clv_metric") or "", "")
    return f"avg CLV {stats['avg_clv']:+.2f}{unit} (n={stats.get('clv_tracked', 0)})"


def evidence_sentence(stats: dict | None) -> str:
    """'n=101 settled (58-43, hit 57% vs 54% breakeven), flat ROI +4.9%, avg CLV
    +0.80% (n=95), 2026-05-01 to 2026-09-20'."""
    if not stats:
        return "no stats"
    parts = [f"n={stats.get('settled', 0)} settled "
             f"({stats.get('wins', 0)}-{stats.get('losses', 0)}, hit {_fmt_hit(stats.get('hit_rate'))}"
             + (f" vs {_fmt_hit(stats.get('breakeven_hit_rate'))} breakeven)"
                if stats.get("breakeven_hit_rate") is not None else ")"),
             f"flat ROI {_fmt_pct(stats.get('flat_roi'))}", _fmt_clv(stats)]
    if stats.get("date_from") and stats.get("date_to"):
        parts.append(f"{stats['date_from']} to {stats['date_to']}")
    return ", ".join(parts)


def _human_pool(pool: str) -> str:
    return {
        "ml-bets": "MLB moneyline bets", "ml-passes": "MLB moneyline passes",
        "totals-bets": "MLB totals bets", "totals-passes": "MLB totals passes",
        "griff-bets": "GriffBet bets", "fb-bets": "football bets",
        "fb-spread-holds": "held football spreads", "fb-total-holds": "held football totals",
    }.get(pool, pool)


def _price_bound(value: str) -> int | None:
    """Tighter heavy-favorite bound for a losing favorite band: one tick
    shorter than the band's LONGEST price (e.g. 'fav -120..-149' -> -119)."""
    rng = PRICE_RANGES.get(value)
    if rng is None or rng[1] >= 0:
        return None
    hi = rng[1]                      # e.g. -120 for 'fav -120..-149'
    return int(hi) + 1 if hi < -101 else -101


def _spread_bound(value: str) -> float | None:
    m = re.match(r".*?([\d.]+)-([\d.]+)", str(value))
    if m:
        return float(m.group(1))
    if "28+" in str(value):
        return 28.0
    return None


# --- rule table ---------------------------------------------------------------
# (pool, dim, direction) -> builder(value, base_cfg, stats) -> (change text, overlay) | None
Rule = Callable[[str, dict, dict | None], tuple[str, dict] | None]


def _ml_fav_size(value: str, base: dict, stats) -> tuple[str, dict] | None:
    bound = _price_bound(value)
    if bound is None:
        return None
    cur = _cfg(base, "filters.heavy_favorite_american", -150)
    if cur is not None and bound <= cur:
        return None                  # already filtered
    return (f"Tighten filters.heavy_favorite_american from {cur} to {bound} "
            f"(no moneyline bets at {bound} or shorter).",
            {"filters.heavy_favorite_american": bound})


def _ml_sharp_fade(value: str, base: dict, stats):
    if "fade" not in value:
        return None
    return ("Lower sanity.max_sharp_disagreement_pp to 3.0 so picks that fade the sharp "
            "consensus by 3pp+ are skipped.", {"sanity.max_sharp_disagreement_pp": 3.0})


def _ml_fragile(value: str, base: dict, stats):
    if value != "fragile":
        return None
    cur = float(_cfg(base, "adjusted_ev.fragile_reduction", 0.01) or 0.01)
    new = round(cur + 0.01, 4)
    return (f"Raise adjusted_ev.fragile_reduction from {cur} to {new} "
            f"(fragile edges shaved by another 1pp of EV).", {"adjusted_ev.fragile_reduction": new})


def _ml_model_prob(value: str, base: dict, stats):
    if value != "<50%":
        return None
    return ("Set filters.min_model_prob to 0.50 (no bet when the blended probability "
            "on the pick is below 50%).", {"filters.min_model_prob": 0.50})


def _ml_pass_ev(value: str, base: dict, stats):
    cur = float(_cfg(base, "ev.threshold", 0.03) or 0.03)
    new = round(cur - 0.01, 4)
    if new <= 0:
        return None
    return (f"Lower ev.threshold from {cur} to {new} (paper first: passes just under the "
            f"bar would have been profitable).", {"ev.threshold": new})


def _totals_wind(value: str, base: dict, stats):
    if "out" not in value:
        return None
    cur = float(_cfg(base, "totals.weather.wind_out_coef", 0.01) or 0.01)
    new = round(cur + 0.005, 4)
    return (f"Raise totals.weather.wind_out_coef from {cur} to {new} so wind blowing out "
            f"lifts the projected total more.", {"totals.weather.wind_out_coef": new})


def _totals_roof(value: str, base: dict, stats):
    if value != "retractable_assumed_open":
        return None
    return ("Set totals.weather.require_verified_roof: true (hold totals in retractable-roof "
            "parks until the roof state is verified).", {"totals.weather.require_verified_roof": True})


def _fb_spread_band(value: str, base: dict, stats):
    bound = _spread_bound(value)
    if bound is None:
        return None
    cur = float(_cfg(base, "college.max_abs_spread", 28.0) or 28.0)
    if bound >= cur:
        return None
    return (f"Lower college.max_abs_spread from {cur} to {bound} (no CFB spread bets at "
            f"|line| >= {bound}).", {"college.max_abs_spread": bound})


def _fb_divergence(value: str, base: dict, stats):
    if "divergence" not in value:
        return None
    cur = float(_cfg(base, "projections.max_spread_divergence_pts_cfb", 6.0) or 6.0)
    new = round(cur + 1.5, 2)
    return (f"Raise projections.max_spread_divergence_pts_cfb from {cur} to {new} (paper: "
            f"the divergence guard is holding leans that cover and beat the close).",
            {"projections.max_spread_divergence_pts_cfb": new})


RULES: dict[tuple[str, str, str], Rule] = {
    ("ml-bets", "fav_size", "negative"): _ml_fav_size,
    ("ml-bets", "odds_bucket", "negative"): _ml_fav_size,
    ("ml-bets", "sharp_fade", "negative"): _ml_sharp_fade,
    ("ml-bets", "stability", "negative"): _ml_fragile,
    ("ml-bets", "model_prob_bucket", "negative"): _ml_model_prob,
    ("ml-passes", "ev_shortfall", "positive"): _ml_pass_ev,
    ("ml-passes", "ev_bucket", "positive"): _ml_pass_ev,
    ("totals-bets", "wind_bucket", "negative"): _totals_wind,
    ("totals-bets", "roof", "negative"): _totals_roof,
    ("fb-bets", "spread_band", "negative"): _fb_spread_band,
    ("fb-spread-holds", "hold_kind", "positive"): _fb_divergence,
}

_INSIGHT_ONLY = {("fb-bets", "rest_bucket", "negative")}


def suggest(engine: str, pool: str, dim: str, value: str, direction: str,
            base_cfg: dict | None, stats: dict | None = None,
            sport: str | None = None) -> Suggestion:
    """Proposal text (+ overlay when the combo is in the rule table)."""
    value = str(value)
    human = _human_pool(pool)
    ev = evidence_sentence(stats)
    is_pass_pool = pool.endswith("passes") or pool.endswith("holds")
    if direction == "negative":
        what = ("lose counterfactually (the engine was right to pass)" if is_pass_pool
                else "are losing money and/or losing to the close")
    else:
        what = ("would have been profitable (the engine is passing on a real edge)"
                if is_pass_pool else "are beating the market")
    title = f"{human}: {dim}={value} {'is losing' if direction == 'negative' else 'is winning'}"
    description = f"{human} where {dim} = {value} {what}: {ev}."

    change = overlay = None
    key = (pool, dim, direction)
    if key in RULES and key not in _INSIGHT_ONLY:
        if pool.startswith("fb") and dim == "spread_band" and sport == "nfl":
            change = overlay = None      # the CFB cap does not apply to NFL
        else:
            built = RULES[key](value, base_cfg or {}, stats)
            if built:
                change, overlay = built
    if overlay:
        return Suggestion("overlay", title, description, change, overlay)
    note = ("No automatic rule for this segment; recorded as an insight for a "
            "human-written change if it persists.")
    return Suggestion("insight", title, f"{description} {note}", None, None)


def small_favorites_suggestion(pool: str, direction: str, base_cfg: dict | None,
                               stats: dict | None) -> Suggestion:
    """Text for the pre-registered small-favorite cell (pooled -101..-149)."""
    if pool == "ml-bets" and direction == "negative":
        return Suggestion("overlay",
                          "MLB moneyline bets: small favorites (-101..-149) are losing",
                          f"Pre-registered test. Bet small favorites {evidence_sentence(stats)}.",
                          "Tighten filters.heavy_favorite_american to -101 (paper first).",
                          {"filters.heavy_favorite_american": -101})
    return suggest("mlb", pool, "fav_size", "small favs -101..-149", direction, base_cfg, stats)


def value_is_finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))
