"""Projected spread + total — PURE, transparent, normal-margin model.

Expected points per team:
    base_pts (league)
  + plays_pg x phase-weighted EPA/play edge (own offense vs what the opponent's
    defense allows — the standard opponent adjustment: mean of the two views)
  + red-zone efficiency kicker vs the pool average (when available)
HFA lands on the margin. Weather multiplies the TOTAL only, and only DOWNWARD
(football_weather is suppress-only by construction).

Cover / over probabilities come from a normal margin/total distribution
(league-specific sigmas from config) WITH integer-line push mass (the +/-0.5
continuity band around the number).

Over-bias guardrails (the MLB totals lesson, applied from day one):
  * the model's tilt is measured against the market-implied MEAN (anchor), not
    the posted line — `market_anchor_mean` recovers it from the de-vigged
    two-way price via the inverse normal, so skewed juice shifts the anchor,
  * the pipeline holds any projection diverging beyond the config guard from
    the anchor to analysis-only (tested for symmetry in test_football.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

_N = NormalDist()


@dataclass
class TeamPoints:
    points: float
    epa_pass: float | None
    epa_rush: float | None
    plays_pg: float | None
    rz_edge: float | None
    notes: list[str] = field(default_factory=list)


@dataclass
class GameProjection:
    home_pts: float
    away_pts: float
    margin: float               # home - away, HFA included
    total_raw: float            # before weather
    total: float                # after the (suppress-only) weather multiplier
    weather_mult: float
    home_detail: TeamPoints
    away_detail: TeamPoints


def _opponent_adjusted_epa(off_epa: float | None, def_allowed: float | None) -> float | None:
    """Mean of 'what this offense does' and 'what this defense allows'."""
    vals = [v for v in (off_epa, def_allowed) if v is not None and v == v]
    return sum(vals) / len(vals) if vals else None


def expected_team_points(off_units: dict, opp_def_units: dict, league: str,
                         config: dict, pass_weight: float, rush_weight: float,
                         pool_rz_avg: float | None = None) -> TeamPoints:
    cfg = config.get("projections", {})
    base = float(cfg.get(f"base_pts_{league}", 22.5))
    notes: list[str] = []

    epa_pass = _opponent_adjusted_epa(off_units.get("epa_dropback"),
                                      opp_def_units.get("epa_dropback_allowed"))
    epa_rush = _opponent_adjusted_epa(off_units.get("rush_epa"),
                                      opp_def_units.get("rush_epa_allowed"))
    plays = off_units.get("plays_pg")
    plays = float(plays) if plays is not None and plays == plays else None

    pts = base
    if plays is not None and (epa_pass is not None or epa_rush is not None):
        epa_play = (pass_weight * (epa_pass or 0.0)) + (rush_weight * (epa_rush or 0.0))
        pts += epa_play * plays
        notes.append(f"EPA/play {epa_play:+.3f} x {plays:.0f} plays")
    else:
        notes.append("EPA or pace unavailable -> league-base points")

    rz_edge = None
    rz = off_units.get("rz_td_rate")
    if rz is not None and rz == rz and pool_rz_avg is not None:
        rz_edge = float(rz) - pool_rz_avg
        pts += rz_edge * float(cfg.get("rz_coef_pts", 8.0))
        notes.append(f"red-zone TD rate {float(rz):.0%} vs pool {pool_rz_avg:.0%}")

    return TeamPoints(points=round(pts, 2), epa_pass=epa_pass, epa_rush=epa_rush,
                      plays_pg=plays, rz_edge=rz_edge, notes=notes)


def project_game(home_units: dict, away_units: dict, league: str, config: dict,
                 pass_weight: float, rush_weight: float,
                 weather_mult: float = 1.0,
                 pool_rz_avg: float | None = None) -> GameProjection:
    cfg = config.get("projections", {})
    hfa = float(cfg.get(f"hfa_pts_{league}", 1.5))
    home = expected_team_points(home_units, away_units, league, config,
                                pass_weight, rush_weight, pool_rz_avg)
    away = expected_team_points(away_units, home_units, league, config,
                                pass_weight, rush_weight, pool_rz_avg)
    margin = home.points - away.points + hfa
    total_raw = home.points + away.points
    mult = min(1.0, float(weather_mult))       # suppress-only, belt & suspenders
    return GameProjection(
        home_pts=round(home.points + hfa / 2, 2),
        away_pts=round(away.points - hfa / 2, 2),
        margin=round(margin, 2), total_raw=round(total_raw, 2),
        total=round(total_raw * mult, 2), weather_mult=mult,
        home_detail=home, away_detail=away,
    )


# --- Normal-margin probabilities with push mass --------------------------------

def _is_integer_line(line: float) -> bool:
    return abs(line - round(line)) < 1e-9


def cover_probabilities(mu_margin: float, home_line: float, sigma: float) -> tuple[float, float, float]:
    """(p_home_cover, p_push, p_away_cover) for a HOME spread of `home_line`
    (home covers when margin + home_line > 0). Integer lines carry push mass
    (the +/-0.5 band around exactly -home_line)."""
    x = -home_line                      # the margin that exactly hits the number
    if _is_integer_line(home_line):
        p_push = (_N.cdf((x + 0.5 - mu_margin) / sigma)
                  - _N.cdf((x - 0.5 - mu_margin) / sigma))
        p_home = 1.0 - _N.cdf((x + 0.5 - mu_margin) / sigma)
        p_away = _N.cdf((x - 0.5 - mu_margin) / sigma)
    else:
        p_push = 0.0
        p_home = 1.0 - _N.cdf((x - mu_margin) / sigma)
        p_away = 1.0 - p_home
    return p_home, p_push, p_away


def total_probabilities(mu_total: float, line: float, sigma: float) -> tuple[float, float, float]:
    """(p_over, p_push, p_under) for a totals line."""
    if _is_integer_line(line):
        p_push = (_N.cdf((line + 0.5 - mu_total) / sigma)
                  - _N.cdf((line - 0.5 - mu_total) / sigma))
        p_over = 1.0 - _N.cdf((line + 0.5 - mu_total) / sigma)
        p_under = _N.cdf((line - 0.5 - mu_total) / sigma)
    else:
        p_push = 0.0
        p_over = 1.0 - _N.cdf((line - mu_total) / sigma)
        p_under = 1.0 - p_over
    return p_over, p_push, p_under


def market_anchor_mean(line: float, p_favored_devig: float, sigma: float,
                       kind: str) -> float:
    """Market-implied MEAN recovered from the de-vigged price at the line.

    kind="total": p is P(over) -> mean = line + sigma * z(p).
    kind="spread": p is P(home cover) of home_line -> mean margin =
    -line + sigma * z(p). With symmetric juice p=0.5 and the anchor IS the
    line; skewed juice shifts it — this is the mean the tilt guard measures
    against (never the raw line)."""
    p = min(0.9999, max(0.0001, p_favored_devig))
    z = _N.inv_cdf(p)
    if kind == "total":
        return line + sigma * z
    return -line + sigma * z
