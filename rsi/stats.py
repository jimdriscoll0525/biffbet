"""Per-cell statistics and candidate detection (ported from the retro
sidecar's _cell_stats / _segment_report / _wilson_ci / _t_test).

A cell = the rows of one pool sharing one bucket value. cell_stats gives its
record, flat 1u P/L, a one-sample t on flat P/L, Wilson CI on hit rate vs
the odds-implied breakeven, and avg CLV in the engine's own unit
(clv_metric). segment_report turns cells clearing the evidence bar into
Candidates; bonferroni() is the multiple-comparisons guard the review applies
once it knows how many cells were inspected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from mlb_value_bot.rsi.segments import DIAGNOSTIC_DIMS, NA_VALUES, SETTLED


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half, center + half)


def t_test(series: pd.Series) -> tuple[float, float]:
    """One-sample t of mean(series) vs 0 -> (t, p). NaN-safe; scipy p-value."""
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    n = len(x)
    if n < 2 or float(x.std(ddof=1)) == 0.0:
        return (float("nan"), float("nan"))
    t, p = _scipy_stats.ttest_1samp(x, 0.0)
    return (float(t), float(p))


def bonferroni(p: float | None, cells_tested: int) -> bool:
    """True when p survives a Bonferroni correction over `cells_tested`."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return False
    return p * max(int(cells_tested), 1) < 0.05


def _round(x, nd: int):
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def cell_stats(sub: pd.DataFrame, clv_metric: str = "clv_pct") -> dict:
    """Metric bundle for one cell. Expects the prepared columns `_result`,
    `flat_pl`, `_dec`, `_clv`, `ev_pct`, `_date` (see rsi.segments)."""
    results = sub["_result"] if "_result" in sub.columns else sub["result"].fillna("pending").astype(str)
    settled = sub[results.isin(SETTLED)]
    n = len(settled)
    wins = int((results.loc[settled.index] == "win").sum())
    pushes = int((results == "push").sum())
    flat = pd.to_numeric(settled["flat_pl"], errors="coerce") if n else pd.Series(dtype=float)
    t, p = t_test(flat)
    lo, hi = wilson_ci(wins, n)
    dec = pd.to_numeric(settled["_dec"], errors="coerce").dropna() if n and "_dec" in settled.columns else pd.Series(dtype=float)
    breakeven = float((1.0 / dec).mean()) if len(dec) else float("nan")
    clv = pd.to_numeric(sub["_clv"], errors="coerce").dropna() if "_clv" in sub.columns else pd.Series(dtype=float)
    ev = pd.to_numeric(sub["ev_pct"], errors="coerce").dropna() if "ev_pct" in sub.columns else pd.Series(dtype=float)
    dates = sub["_date"].astype(str) if "_date" in sub.columns else sub["date"].astype(str)
    dates = dates[dates.str.len() >= 10]
    return {
        "rows": int(len(sub)),
        "settled": n,
        "wins": wins,
        "losses": n - wins,
        "pushes": pushes,
        "hit_rate": _round(wins / n, 4) if n else None,
        "hit_rate_wilson_95": [_round(lo, 4), _round(hi, 4)] if n else None,
        "breakeven_hit_rate": _round(breakeven, 4),
        "flat_pl_units": _round(flat.sum(), 3) if n else 0.0,
        "flat_roi": _round(flat.mean(), 4) if n else None,
        "t_stat": _round(t, 2),
        "p_value": _round(p, 4),
        "avg_ev_pct": _round(ev.mean() * 100, 2) if len(ev) else None,
        "avg_clv": _round(clv.mean(), 3) if len(clv) else None,
        "clv_metric": clv_metric,
        "clv_tracked": int(len(clv)),
        "clv_positive": int((clv > 0).sum()),
        "date_from": str(dates.min()) if len(dates) else None,
        "date_to": str(dates.max()) if len(dates) else None,
    }


@dataclass
class Candidate:
    key: str
    pool: str
    dimension: str
    value: str
    direction: str                 # positive | negative
    stats: dict
    engine: str = ""
    sport: str = ""
    t_trigger: bool = False
    clv_trigger: bool = False
    clv_agrees: bool | None = None
    bonferroni_significant: bool = False
    preregistered: bool = False
    cells_tested: int = 0
    delta_clv: float | None = None
    delta_roi: float | None = None
    suggestion: dict | None = None
    corroborating: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key, "pool": self.pool, "dimension": self.dimension,
            "value": self.value, "direction": self.direction, "engine": self.engine,
            "sport": self.sport, "t_trigger": self.t_trigger, "clv_trigger": self.clv_trigger,
            "clv_agrees": self.clv_agrees, "bonferroni_significant": self.bonferroni_significant,
            "preregistered": self.preregistered, "cells_tested": self.cells_tested,
            "delta_clv": self.delta_clv, "delta_roi": self.delta_roi, **self.stats,
        }


def clv_strength(sub: pd.DataFrame, t_threshold: float) -> tuple[float, bool]:
    """(avg_clv sign as +1/-1/0, strong?) -- strong when the one-sample t on
    the CLV series clears t_threshold."""
    clv = pd.to_numeric(sub["_clv"], errors="coerce").dropna() if "_clv" in sub.columns else pd.Series(dtype=float)
    if len(clv) == 0:
        return (0.0, False)
    t, _ = t_test(clv)
    return (float(np.sign(clv.mean())), (not math.isnan(t)) and abs(t) >= t_threshold)


def segment_report(df: pd.DataFrame, dims: list[str], pool: str, min_n: int,
                   t_threshold: float, clv_min_tracked: int = 10,
                   clv_metric: str = "clv_pct") -> tuple[list[Candidate], dict, int]:
    """(candidates, tables, cells_tested) for one pool.

    A cell is a candidate when settled >= min_n and either |t| >= t_threshold
    on flat P/L, or its CLV sign is strong (|t_clv| >= t_threshold) with at
    least clv_min_tracked CLV rows. cells_tested counts every cell with
    settled >= 5 (the Bonferroni denominator).
    """
    candidates: list[Candidate] = []
    tables: dict[str, dict] = {}
    cells_tested = 0
    if df is None or df.empty:
        return candidates, tables, cells_tested
    sport_col = df["sport"] if "sport" in df.columns else None
    for dim in dims:
        if dim not in df.columns:
            continue
        table: dict[str, dict] = {}
        for value, sub in df.groupby(dim, dropna=True, observed=True):
            stats = cell_stats(sub, clv_metric)
            table[str(value)] = stats
            if stats["settled"] >= 5:
                cells_tested += 1
            if stats["settled"] < min_n:
                continue
            # Tabulated, never a finding: circular dims (clv_sign) and
            # "data absent" buckets.
            if dim in DIAGNOSTIC_DIMS or str(value).strip().lower() in NA_VALUES:
                continue
            t_trigger = stats["t_stat"] is not None and abs(stats["t_stat"]) >= t_threshold
            clv_sign, strong = clv_strength(sub, t_threshold)
            clv_trigger = strong and stats["clv_tracked"] >= clv_min_tracked
            if not (t_trigger or clv_trigger):
                continue
            if t_trigger:
                direction = "positive" if (stats["flat_roi"] or 0) > 0 else "negative"
            else:
                direction = "positive" if clv_sign > 0 else "negative"
            clv_agrees: bool | None = None
            if stats["avg_clv"] is not None and stats["clv_tracked"] >= clv_min_tracked:
                clv_agrees = (stats["avg_clv"] > 0) == (direction == "positive")
            sport = ""
            if sport_col is not None:
                mode = sub["sport"].dropna().astype(str).mode()
                sport = str(mode.iloc[0]) if len(mode) else ""
            candidates.append(Candidate(
                key=f"{pool}|{dim}={value}", pool=pool, dimension=dim, value=str(value),
                direction=direction, stats=stats, sport=sport, t_trigger=t_trigger,
                clv_trigger=clv_trigger, clv_agrees=clv_agrees))
        tables[dim] = table
    return candidates, tables, cells_tested
