"""Confidence % — PURE. Edge magnitude x data completeness, with the
college-noise haircuts Jim specified:

  * league-wide CFB multiplier (college outcomes are noisier than NFL),
  * a further Group-of-Five haircut (confidence AND the displayed stake),
  * the late-game explosiveness factor: explosive offenses widen the outcome
    distribution, so TOTALS confidence is cut — the projection itself is
    NEVER bumped (variance, not mean; locked by test).

Returns the number plus its component breakdown (transparency ethos).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Confidence:
    value: float                 # 0-100
    stake_mult: float            # G5 haircut also shrinks the displayed stake
    components: dict


def confidence_for_pick(*, edge_abs: float, completeness: float, league: str,
                        g5_involved: bool, market: str, explosive_involved: bool,
                        stability_label: str, config: dict) -> Confidence:
    ccfg = config.get("college", {})
    vcfg = config.get("variance", {})

    base = 30.0 + min(60.0, 0.6 * edge_abs)          # edge 0 -> 30, edge 100 -> 90
    comp_factor = 0.70 + 0.30 * max(0.0, min(1.0, completeness))
    value = base * comp_factor
    components = {"edge_base": round(base, 1), "completeness_factor": round(comp_factor, 3)}

    stake_mult = 1.0
    if league == "cfb":
        m = float(ccfg.get("league_confidence_mult", 0.90))
        value *= m
        components["cfb_league_mult"] = m
        if g5_involved:
            gm = float(ccfg.get("g5_confidence_mult", 0.85))
            value *= gm
            stake_mult = float(ccfg.get("g5_stake_mult", 0.75))
            components["g5_mult"] = gm
            components["g5_stake_mult"] = stake_mult

    if market == "total" and explosive_involved:
        em = float(vcfg.get("explosive_confidence_mult", 0.90))
        value *= em
        components["explosive_variance_mult"] = em

    if stability_label == "fragile":
        value *= 0.85
        components["fragile_mult"] = 0.85

    return Confidence(value=round(max(5.0, min(95.0, value)), 1),
                      stake_mult=stake_mult, components=components)
