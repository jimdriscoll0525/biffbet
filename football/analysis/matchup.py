"""Matchup archetypes + edge scores — PURE, the heart of the model.

Edge per phase = O_pct - D_pct (both oriented higher=better). This is exactly
Jim's "offense percentile minus (100 - defense percentile)" under his
defense-percentiled-by-badness convention: O - (100 - D_bad) with
D_bad = 100 - D_good  ==  O - D_good. Consequences the spec demands:
strong-vs-strong and weak-vs-weak land near 0 (neutral); strong O vs weak D
is a large positive (offense edge, leans over); weak O vs strong D is a large
negative (defense edge, leans under).

Four scored unit matchups per game (each side's pass O vs the other's pass D,
each side's rush O vs the other's run D), plus:
  * DUAL EDGE — one team holds BOTH the pass and rush offensive edge: the
    highest-conviction archetype, boosted multiplicatively,
  * turnover pairing flag — high-takeaway D meets turnover-prone O,
  * phase weights (pass-heavier by default) shiftable toward the phase the
    expected game script favors (the pipeline passes the market lean).

The net game edge is home-positive, in percentile points (~-100..+100).
"""
from __future__ import annotations

from dataclasses import dataclass, field

ARCH_OFFENSE = "strong_o_vs_weak_d"     # edge: offense (totals lean over)
ARCH_DEFENSE = "weak_o_vs_strong_d"     # edge: defense (totals lean under)
ARCH_NEUTRAL = "neutral"                # incl. strong-vs-strong / weak-vs-weak
ARCH_DUAL = "dual_edge"


@dataclass
class PhaseEdge:
    offense_side: str        # "home" | "away" — whose offense is on the field
    phase: str               # "pass" | "rush"
    o_pct: float | None
    d_pct: float | None
    edge: float | None       # O_pct - D_pct; None when either unit is unevaluable
    archetype: str
    note: str


@dataclass
class GameMatchup:
    edges: list[PhaseEdge]
    home_edge: float          # net phase-weighted, home-positive, pct points
    dual_edge_side: str | None
    turnover_flag_side: str | None   # side whose DEFENSE feasts on the other's giveaways
    pass_weight: float
    rush_weight: float
    notes: list[str] = field(default_factory=list)

    @property
    def archetype(self) -> str:
        if self.dual_edge_side:
            return ARCH_DUAL
        scored = [e for e in self.edges if e.edge is not None]
        if not scored:
            return ARCH_NEUTRAL
        return max(scored, key=lambda e: abs(e.edge)).archetype


def classify_phase(o_pct: float | None, d_pct: float | None, config: dict,
                   offense_side: str, phase: str) -> PhaseEdge:
    cfg = config.get("percentiles", {})
    strong = float(cfg.get("strong_threshold", 75))
    weak = float(cfg.get("weak_threshold", 25))
    band = float(config.get("matchup", {}).get("neutral_band", 10.0))

    if o_pct is None or d_pct is None or o_pct != o_pct or d_pct != d_pct:  # NaN-safe
        return PhaseEdge(offense_side, phase, None, None, None, ARCH_NEUTRAL,
                         "unit unevaluable (missing stats)")

    edge = o_pct - d_pct
    if abs(edge) <= band:
        arch, note = ARCH_NEUTRAL, "no exploitable gap"
    elif edge > 0 and o_pct >= strong and d_pct <= weak:
        arch = ARCH_OFFENSE
        note = f"strong {phase} O ({o_pct:.0f}th pct) vs weak D ({d_pct:.0f}th)"
    elif edge < 0 and o_pct <= weak and d_pct >= strong:
        arch = ARCH_DEFENSE
        note = f"weak {phase} O ({o_pct:.0f}th pct) vs strong D ({d_pct:.0f}th)"
    else:
        # A real gap that doesn't meet the strict quartile archetype (e.g. 70th
        # vs 35th): scored, labeled a lean rather than a named archetype.
        arch = ARCH_OFFENSE if edge > 0 else ARCH_DEFENSE
        note = f"{phase} lean ({o_pct:.0f}th pct O vs {d_pct:.0f}th pct D)"
    return PhaseEdge(offense_side, phase, o_pct, d_pct, edge, arch, note)


def phase_weights(config: dict, script_lean: float = 0.0) -> tuple[float, float]:
    """(pass_weight, rush_weight). script_lean in [-1, 1]: positive = expected
    game script favors the ground game (big favorite / bad weather), shifting
    weight from pass to rush, capped by script_shift_max."""
    cfg = config.get("matchup", {})
    w_pass = float(cfg.get("phase_weight_pass", 0.60))
    w_rush = float(cfg.get("phase_weight_rush", 0.40))
    shift = max(-1.0, min(1.0, script_lean)) * float(cfg.get("script_shift_max", 0.15))
    w_pass, w_rush = w_pass - shift, w_rush + shift
    total = w_pass + w_rush
    return w_pass / total, w_rush / total


def game_matchup(home: dict, away: dict, config: dict,
                 script_lean: float = 0.0) -> GameMatchup:
    """Score one game. `home`/`away` are unit-percentile dicts with keys
    pass_off_pct / pass_def_pct / rush_off_pct / rush_def_pct and optionally
    takeaway_pct / ball_security_pct (rows of percentiles.unit_percentiles)."""
    mcfg = config.get("matchup", {})
    band = float(mcfg.get("neutral_band", 10.0))

    hp = classify_phase(home.get("pass_off_pct"), away.get("pass_def_pct"), config, "home", "pass")
    hr = classify_phase(home.get("rush_off_pct"), away.get("rush_def_pct"), config, "home", "rush")
    ap = classify_phase(away.get("pass_off_pct"), home.get("pass_def_pct"), config, "away", "pass")
    ar = classify_phase(away.get("rush_off_pct"), home.get("rush_def_pct"), config, "away", "rush")
    edges = [hp, hr, ap, ar]

    w_pass, w_rush = phase_weights(config, script_lean)

    def net(pass_e: PhaseEdge, rush_e: PhaseEdge) -> float:
        return (w_pass * (pass_e.edge or 0.0)) + (w_rush * (rush_e.edge or 0.0))

    home_edge = net(hp, hr) - net(ap, ar)

    # Dual edge: one team's offense holds BOTH phase edges beyond the band.
    dual_side = None
    if (hp.edge or 0) > band and (hr.edge or 0) > band:
        dual_side = "home"
    elif (ap.edge or 0) > band and (ar.edge or 0) > band:
        dual_side = "away"
    if dual_side:
        bonus = 1.0 + float(mcfg.get("dual_edge_bonus", 0.15))
        boosted = home_edge * bonus
        # The bonus amplifies conviction toward the dual-edge side; never let it
        # flip or shrink an edge pointing the other way.
        if (dual_side == "home" and home_edge > 0) or (dual_side == "away" and home_edge < 0):
            home_edge = boosted

    # Turnover pairing: a top-quartile takeaway defense vs a bottom-quartile
    # ball-security offense (flag + note; EV magnitude belongs to adjusted EV).
    flag_thresh = float(mcfg.get("turnover_flag_threshold", 75))
    to_side = None
    if (home.get("takeaway_pct") or 0) >= flag_thresh and \
            (away.get("ball_security_pct") or 100) <= 100 - flag_thresh:
        to_side = "home"
    elif (away.get("takeaway_pct") or 0) >= flag_thresh and \
            (home.get("ball_security_pct") or 100) <= 100 - flag_thresh:
        to_side = "away"

    notes = [f"{e.offense_side} {e.phase}: {e.note}" for e in edges if e.edge is not None]
    if dual_side:
        notes.insert(0, f"DUAL EDGE: {dual_side} holds both the pass and rush matchup")
    if to_side:
        notes.append(f"turnover pairing: {to_side} defense vs turnover-prone offense")

    return GameMatchup(edges=edges, home_edge=round(home_edge, 2),
                       dual_edge_side=dual_side, turnover_flag_side=to_side,
                       pass_weight=round(w_pass, 3), rush_weight=round(w_rush, 3),
                       notes=notes)
