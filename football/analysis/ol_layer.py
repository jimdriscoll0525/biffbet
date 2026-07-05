"""Offensive-line layer — PURE. Modifies BOTH of a team's phase edges.

Two graded components (each -1..+1, linear between the config thresholds):
  * pass protection: the sacks+QB-hits-per-dropback PROXY (real pressure data
    isn't free in-season — labeled a proxy in every surfaced note),
  * run blocking: yards/carry (league-specific thresholds; CFB's are the
    SP+-adjusted baselines from config).

Continuity dampens otherwise-strong grades (an OL that just lost starters
doesn't deserve its season-long numbers): NFL from week-over-week starting-
five snap overlap, CFB from returning production. Continuity NEVER boosts —
it can only strip up to continuity_dampener_max of a POSITIVE grade.

Output is an edge-point adjustment (grade x edge_points_scale) the pipeline
adds to that team's pass and rush edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_OL_POSITIONS = {"T", "G", "C", "OT", "OG", "LT", "LG", "RG", "RT"}


@dataclass
class OLGrade:
    grade: float                  # -1..+1 after continuity dampening
    points: float                 # grade x edge_points_scale (edge-point units)
    pass_pro: float | None        # component grades pre-dampening
    run_block: float | None
    continuity: float | None      # 0..1, None = unknown (treated neutral)
    notes: list[str]


def _linear_grade(value: float, good: float, bad: float) -> float:
    """-1..+1, +1 at/beyond `good`, -1 at/beyond `bad` (handles either order)."""
    if good == bad:
        return 0.0
    t = (value - bad) / (good - bad)
    return max(-1.0, min(1.0, 2.0 * t - 1.0))


def ol_grade(unit_row: dict, league: str, config: dict,
             continuity: float | None = None) -> OLGrade:
    cfg = config.get("ol_layer", {})
    notes: list[str] = []

    pass_pro = None
    pressure = unit_row.get("pressure_proxy_rate")
    if pressure is None or pressure != pressure:
        pressure = unit_row.get("sack_rate_allowed")   # CFB fallback: sacks only
    if pressure is not None and pressure == pressure:
        pass_pro = _linear_grade(float(pressure),
                                 good=float(cfg.get("pressure_elite", 0.15)),
                                 bad=float(cfg.get("pressure_poor", 0.28)))
        label = ("elite" if pass_pro > 0.5 else
                 "poor" if pass_pro < -0.5 else "average")
        notes.append(f"pass-pro proxy {float(pressure):.1%}/dropback ({label})")

    run_block = None
    ypc = unit_row.get("ypc")
    if ypc is not None and ypc == ypc:
        good = float(cfg.get(f"ypc_good_{league}", cfg.get("ypc_good_nfl", 4.5)))
        bad = float(cfg.get(f"ypc_bad_{league}", cfg.get("ypc_bad_nfl", 4.0)))
        run_block = _linear_grade(float(ypc), good=good, bad=bad)
        notes.append(f"run-block proxy {float(ypc):.2f} ypc")

    components = [c for c in (pass_pro, run_block) if c is not None]
    grade = sum(components) / len(components) if components else 0.0

    # Continuity dampener: strips up to continuity_dampener_max of a POSITIVE
    # grade as continuity falls from 1 -> 0. Negative grades stand (a bad OL
    # that also lost starters is not thereby less bad).
    if continuity is not None and grade > 0:
        max_damp = float(cfg.get("continuity_dampener_max", 0.30))
        factor = 1.0 - max_damp * (1.0 - max(0.0, min(1.0, continuity)))
        if factor < 1.0:
            grade *= factor
            notes.append(f"continuity dampener x{factor:.2f} "
                         f"(continuity {continuity:.0%})")

    scale = float(cfg.get("edge_points_scale", 10.0))
    return OLGrade(grade=round(grade, 3), points=round(grade * scale, 2),
                   pass_pro=pass_pro, run_block=run_block,
                   continuity=continuity, notes=notes)


def nfl_ol_continuity(snap_counts: pd.DataFrame, team: str,
                      week: int) -> float | None:
    """Starting-five overlap between this team's two most recent completed
    weeks before `week` (the nflverse-injuries-are-dead substitute). None when
    there isn't enough history (week 1/2, missing feed) — treated neutral."""
    if snap_counts is None or snap_counts.empty:
        return None
    need = {"team", "week", "position", "player", "offense_snaps"}
    if not need <= set(snap_counts.columns):
        return None
    ol = snap_counts[(snap_counts["team"] == team)
                     & (snap_counts["position"].isin(_OL_POSITIONS))
                     & (snap_counts["week"] < week)]
    weeks = sorted(ol["week"].unique())
    if len(weeks) < 2:
        return None
    last, prev = weeks[-1], weeks[-2]

    def starters(wk) -> set[str]:
        w = ol[ol["week"] == wk]
        return set(w.nlargest(5, "offense_snaps")["player"])

    s_last, s_prev = starters(last), starters(prev)
    if not s_last or not s_prev:
        return None
    return len(s_last & s_prev) / 5.0
