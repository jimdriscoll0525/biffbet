"""Unit percentiles — PURE. 0-100 percentile-rank every unit stat WITHIN a
league pool (the caller passes one league's frame at a time; NFL and FBS are
never concatenated — regression-tested).

Every unit percentile is oriented HIGHER = BETTER (for defense that means
better at stopping). Jim's edge formula "O_pct - (100 - D_pct)" assumed
defense percentiled by badness-allowed; with defense oriented higher=better
the identical quantity is O_pct - D_pct (matchup.py documents the same).

Early-season, single-league stats are noise, so the caller may blend the
current frame with the PRIOR season's via `blend_with_prior` (linear weight
decay by week, config-driven, 0 from prior_out_week on).
"""
from __future__ import annotations

import pandas as pd

# unit -> {stat column: higher_is_better}. Missing columns are skipped and the
# unit percentile is the mean of the stat percentiles that DO exist (this is
# how NFL and CFB share one spec despite different feeds).
UNIT_SPECS: dict[str, dict[str, bool]] = {
    "pass_off": {
        "pass_ypg": True, "pass_td_pg": True, "ypa": True,
        "sack_rate_allowed": False, "int_rate": False, "epa_dropback": True,
    },
    "pass_def": {
        "pass_ypg_allowed": False, "ypa_allowed": False, "sack_rate_made": True,
        "int_created_rate": True, "epa_dropback_allowed": False,
    },
    "rush_off": {"rush_ypg": True, "ypc": True, "rush_epa": True},
    "rush_def": {"rush_ypg_allowed": False, "ypc_allowed": False, "rush_epa_allowed": False},
    # Turnover profile: security = avoiding giveaways; takeaway = creating them.
    "ball_security": {"giveaway_pg": False},
    "takeaway": {"takeaway_pg": True},
}


def percentile_series(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """0-100 percentile rank within the pool. NaNs stay NaN."""
    ranked = values.rank(pct=True, ascending=higher_is_better)
    return ranked * 100.0


def unit_percentiles(unit_stats: pd.DataFrame,
                     specs: dict[str, dict[str, bool]] | None = None) -> pd.DataFrame:
    """Per-stat percentiles (`<stat>_pct`) + per-unit composites (`<unit>_pct`).

    A unit with NO available stats gets NaN (downstream treats that matchup
    as unevaluable rather than silently average).
    """
    specs = specs or UNIT_SPECS
    if unit_stats.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=unit_stats.index)
    for unit, stats in specs.items():
        stat_pcts = []
        for col, higher_better in stats.items():
            if col not in unit_stats.columns:
                continue
            pct = percentile_series(unit_stats[col], higher_better)
            out[f"{col}_pct"] = pct
            stat_pcts.append(pct)
        out[f"{unit}_pct"] = (pd.concat(stat_pcts, axis=1).mean(axis=1)
                              if stat_pcts else float("nan"))
    return out


def prior_weight(week: int, config: dict) -> float:
    """Linear prior-season weight decay: prior_weight_week1 at week 1 -> 0 at
    prior_out_week (and beyond)."""
    cfg = config.get("percentiles", {})
    w1 = float(cfg.get("prior_weight_week1", 0.8))
    out_week = int(cfg.get("prior_out_week", 7))
    if week >= out_week:
        return 0.0
    if week <= 1:
        return w1
    return w1 * (out_week - week) / (out_week - 1)


def blend_with_prior(current: pd.DataFrame, prior: pd.DataFrame,
                     week: int, config: dict) -> pd.DataFrame:
    """Blend current-season unit stats with the prior season's, weighted by
    week. Teams absent from the prior frame (promoted FBS programs) keep their
    current values; `games` stays current (it drives evaluability checks)."""
    w = prior_weight(week, config)
    if w <= 0.0 or prior is None or prior.empty:
        return current
    if current is None or current.empty:
        return prior
    blended = current.copy()
    shared_cols = [c for c in current.columns
                   if c != "games" and c in prior.columns]
    aligned_prior = prior.reindex(current.index)
    for col in shared_cols:
        cur, pri = current[col], aligned_prior[col]
        merged = (1.0 - w) * cur + w * pri
        blended[col] = merged.where(pri.notna(), cur)
    return blended
