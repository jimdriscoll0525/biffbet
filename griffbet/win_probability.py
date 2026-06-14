"""GriffBet win-probability model — a thin wrapper over BiffBet's model.

DESIGN (plan §B, drift-control safeguard): rather than text-copy BiffBet's
~680-line model (which would rot), GriffBet REUSES BiffBet's
`compute_win_probability` by import and applies its ONE intended model
difference -- the starter-neutralized base rate -- as a post-step. With the
neutralization flag OFF, GriffBet calls BiffBet's function and returns its
result untouched, so GriffBet's win probabilities are IDENTICAL to BiffBet's by
construction (locked by the golden-equivalence test). Every divergence is then
attributable solely to GriffBet's config + the toggle, never accidental drift.

The neutralized base rate is recoverable as a clean post-adjustment because
BiffBet computes `home_win_prob = clamp(base + Σ weighted_deltas)` and the
component deltas do NOT depend on the base. So:

    raw_sum   = base_off + Σ weighted_deltas          (the pre-clamp prob)
    base_on   = log5(home_wp - home_inc, away_wp - away_inc)
    home_prob = clamp(raw_sum + (base_on - base_off), floor, ceiling)
"""
from __future__ import annotations

from mlb_value_bot.analysis.team_metrics import TeamProfile
from mlb_value_bot.analysis.pitcher_metrics import PitcherProfile
from mlb_value_bot.analysis.win_probability import (
    WinProbabilityResult,
    clamp,
    compute_win_probability,
    log5,
    regress_winpct,
)
from mlb_value_bot.griffbet.team_extras import rotation_rate_source, rotation_winpct_increment
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.win_probability")


def _neutralized_base(
    home_team: TeamProfile,
    away_team: TeamProfile,
    season: int,
    config: dict,
) -> tuple[float, dict]:
    """Recompute base_wp with each team's season rotation-average SP stripped out.

    Returns (base_on, info). `info` records the per-team win% increments and the
    off/on base values for the reasoning trail. Any team whose rotation rate is
    unavailable contributes a 0 increment (un-neutralized) so the result degrades
    gracefully toward BiffBet's base.
    """
    k = float(config["model"].get("team_regression_games", 30))
    home_wp = regress_winpct(home_team.raw_winpct, home_team.games, k)
    away_wp = regress_winpct(away_team.raw_winpct, away_team.games, k)
    base_off = log5(home_wp, away_wp)

    home_inc = rotation_winpct_increment(home_team.team, season, config)
    away_inc = rotation_winpct_increment(away_team.team, season, config)
    home_wp_n = clamp(home_wp - (home_inc or 0.0), 1e-6, 1 - 1e-6)
    away_wp_n = clamp(away_wp - (away_inc or 0.0), 1e-6, 1 - 1e-6)
    base_on = log5(home_wp_n, away_wp_n)

    info = {
        "applied": True,
        "source": rotation_rate_source(season),
        "home_rotation_increment": round(home_inc, 4) if home_inc is not None else None,
        "away_rotation_increment": round(away_inc, 4) if away_inc is not None else None,
        "base_off": round(base_off, 4),
        "base_on": round(base_on, 4),
        "base_shift": round(base_on - base_off, 4),
        "degraded": home_inc is None or away_inc is None,
    }
    return base_on, info


def compute_win_probability_griff(
    home_team: TeamProfile,
    away_team: TeamProfile,
    home_pitcher: PitcherProfile,
    away_pitcher: PitcherProfile,
    config: dict,
    season: int,
    *,
    home_bullpen_status=None,
    away_bullpen_status=None,
    home_lineup_status=None,
    away_lineup_status=None,
) -> tuple[WinProbabilityResult, dict | None]:
    """Run GriffBet's model. Returns (result, neutralization_info | None).

    With model.starter_neutralized_base OFF (default), this is BiffBet's model
    verbatim and neutralization_info is None. With it ON, the base rate is
    re-derived (rotation-average SP stripped) and the result re-clamped.
    """
    result = compute_win_probability(
        home_team, away_team, home_pitcher, away_pitcher, config,
        home_bullpen_status=home_bullpen_status,
        away_bullpen_status=away_bullpen_status,
        home_lineup_status=home_lineup_status,
        away_lineup_status=away_lineup_status,
    )
    if not config["model"].get("starter_neutralized_base", False):
        return result, None

    base_on, info = _neutralized_base(home_team, away_team, season, config)
    floor = float(config["model"]["prob_floor"])
    ceiling = float(config["model"]["prob_ceiling"])
    # Pre-clamp sum is base_off + Σ weighted_deltas; swap base_off -> base_on.
    raw_sum = result.base_prob + sum(c.weighted_delta for c in result.components)
    home_prob = clamp(raw_sum + (base_on - result.base_prob), floor, ceiling)

    result.base_prob = round(base_on, 4)
    result.home_win_prob = home_prob
    result.away_win_prob = 1.0 - home_prob
    return result, info


def base_off_on(
    home_team: TeamProfile,
    away_team: TeamProfile,
    home_pitcher: PitcherProfile,
    away_pitcher: PitcherProfile,
    season: int,
    config: dict,
) -> dict:
    """Diagnostic for `griff investigate-base`: total home win prob with
    neutralization OFF vs ON for this exact matchup/starters, plus the base
    shift. Holds everything else fixed so the delta is purely the neutralization
    effect (which interacts with starter quality -- the point of the report)."""
    off = compute_win_probability(
        home_team, away_team, home_pitcher, away_pitcher, config,
    )
    on_cfg = {**config, "model": {**config["model"], "starter_neutralized_base": True}}
    on_result, info = compute_win_probability_griff(
        home_team, away_team, home_pitcher, away_pitcher, on_cfg, season,
    )
    return {
        "home_prob_off": round(off.home_win_prob, 4),
        "home_prob_on": round(on_result.home_win_prob, 4),
        "prob_shift": round(on_result.home_win_prob - off.home_win_prob, 4),
        "neutralization": info,
    }
