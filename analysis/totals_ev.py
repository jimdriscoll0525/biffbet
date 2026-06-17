"""EV + bet-sizing tiers for the totals (over/under) model.

A totals pick is just a two-way market (over vs under) once we condition on no
push, so the EV/Kelly math is identical to the moneyline's and we DELEGATE to
the already-unit-tested `analysis.ev_calculator`. The only totals-specific piece
is the sizing tier table (config.totals.bet_sizing) and the "never Strong on a
fragile edge" rule, mirrored from the moneyline.

Per the design choice for v1, the totals model uses TIERS + STABILITY only --
there is no Adjusted-EV haircut layer. Selection is by raw EV >= totals.ev_
threshold; fragility caps the tier below Strong and low confidence downgrades
one step. The sharp-fade signal is handled by the SKIP guard in the pipeline,
not by an EV haircut here.
"""
from __future__ import annotations

from mlb_value_bot.analysis.ev_calculator import SideEvaluation, evaluate_sides

# Tier order, lowest -> highest, for single-step downgrades (mirrors pipeline).
_TIER_ORDER = ("pass", "small", "standard", "strong")


def evaluate_ou_sides(
    blended_p_over: float,
    over_american: int,
    under_american: int,
    devig_method: str = "power",
    kelly_multiplier: float = 0.25,
    kelly_cap: float = 0.02,
) -> dict[str, SideEvaluation]:
    """Over/under SideEvaluations from a conditional (no-push) blended P(over).

    `blended_p_over` is the model<->market blended probability the OVER resolves
    YES *given the bet doesn't push* -- both the model (p_over/(p_over+p_under))
    and the market de-vig are conditioned on no push, so they're comparable and
    pushes are correctly treated as stake-neutral. We reuse the moneyline two-way
    evaluator (home:=over, away:=under) and relabel, so EV/Kelly is the exact
    same code path.
    """
    raw = evaluate_sides(
        blended_p_over, over_american, under_american,
        devig_method=devig_method, kelly_multiplier=kelly_multiplier, kelly_cap=kelly_cap,
    )
    over, under = raw["home"], raw["away"]
    over.side, under.side = "over", "under"
    return {"over": over, "under": under}


def classify_totals_tier(
    ev: float,
    confidence: float,
    stability_label: str,
    config: dict,
    is_raw_pick: bool = False,
) -> tuple[str, list[str]]:
    """Classify a totals pick into Pass/Small/Standard/Strong on EV, then apply
    the single one-tier downgrade guardrail. Returns (tier, reasons).

    Bands come from config.totals.bet_sizing.*_ev (decimal fractions). Like the
    moneyline, selection (raw EV >= totals.ev_threshold) is DECOUPLED from sizing:
    a raw-qualifying pick is floored at `small` (`is_raw_pick=True`) so a
    fragility/low-confidence downgrade can size it down but never veto it to pass.
    Downgrade ONE step if confidence < downgrade_confidence, OR the edge is
    FRAGILE and the band is Strong (the inviolable "never Strong on a fragile
    edge" rule).
    """
    sizing = config.get("totals", {}).get("bet_sizing", {})
    small_ev = float(sizing.get("small_ev", 0.02))
    standard_ev = float(sizing.get("standard_ev", 0.05))
    strong_ev = float(sizing.get("strong_ev", 0.08))
    min_conf = float(sizing.get("downgrade_confidence", 60.0))

    floor_idx = _TIER_ORDER.index("small") if is_raw_pick else _TIER_ORDER.index("pass")
    reasons: list[str] = []

    if ev < small_ev:
        if not is_raw_pick:
            return "pass", [f"EV {ev * 100:.1f}% below {small_ev * 100:.0f}% threshold"]
        tier = "small"
        reasons.append(
            f"EV {ev * 100:.1f}% < {small_ev * 100:.0f}% but raw EV cleared the pick "
            f"threshold -> floored to small (selection is by raw EV; tiers only size)"
        )
    elif ev < standard_ev:
        tier = "small"
        reasons.append(f"EV {ev * 100:.1f}% in lean range [{small_ev * 100:.0f}%, {standard_ev * 100:.0f}%)")
    elif ev < strong_ev:
        tier = "standard"
        reasons.append(f"EV {ev * 100:.1f}% in standard range [{standard_ev * 100:.0f}%, {strong_ev * 100:.0f}%)")
    else:
        tier = "strong"
        reasons.append(
            f"EV {ev * 100:.1f}% >= {strong_ev * 100:.0f}% -- FLAG FOR MANUAL REVIEW "
            f"(8%+ totals edge is rare; suspect a stale line / scratch before trusting)"
        )

    triggers: list[str] = []
    if stability_label == "fragile" and tier == "strong":
        triggers.append("fragile edge (never Strong on fragile)")
    if confidence < min_conf:
        triggers.append(f"confidence {confidence:.0f} < {min_conf:.0f}")
    if triggers:
        idx = _TIER_ORDER.index(tier)
        new_tier = _TIER_ORDER[max(floor_idx, idx - 1)]
        if new_tier != tier:
            reasons.append(f"downgraded {tier} -> {new_tier}: {', '.join(triggers)}")
            tier = new_tier
        else:
            reasons.append(f"downgrade noted (already at floor): {', '.join(triggers)}")

    return tier, reasons


def kelly_cap_for_tier(tier: str, config: dict) -> float:
    """Per-tier Kelly cap (fraction of bankroll) from config.totals.bet_sizing."""
    caps = config.get("totals", {}).get("bet_sizing", {}).get("kelly_caps", {})
    default = {"pass": 0.0, "small": 0.005, "standard": 0.010, "strong": 0.020}
    return float(caps.get(tier, default.get(tier, 0.0)))
