"""Performance analytics: ROI, hit rate, CLV — overall and segmented.

The whole point of this tool is to MEASURE edge, not to feel good about picks.
So performance is reported two ways and sliced every way the spec asks:

  * Kelly ROI  — profit_loss (bankroll-fraction) / total staked. Reflects how
    the bankroll actually moved given Kelly sizing.
  * Flat ROI   — every bet treated as 1 unit; comparable across stake schemes.
  * Hit rate   — wins / settled. Secondary at low N (variance dominates).
  * CLV        — average closing-line value; the most stable early-sample signal
    of whether the model is finding genuine market inefficiency.

Segments: confidence bucket, EV bucket, favorite/underdog, home/road,
CLV positive/negative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlb_value_bot.analysis.ev_calculator import american_to_decimal
from mlb_value_bot.tracking import recommendations as recs
from mlb_value_bot.utils import get_logger

log = get_logger("tracking.performance")

SETTLED = {"win", "loss"}


@dataclass
class PerformanceReport:
    overall: dict
    segments: dict[str, pd.DataFrame]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used for segmentation and flat-stake P&L."""
    if df.empty:
        return df
    df = df.copy()
    df["settled"] = df["result"].isin(SETTLED)

    # Flat-stake (1u) P/L for settled bets.
    def _flat_pl(row) -> float:
        if row["result"] == "win":
            return american_to_decimal(row["american_odds"]) - 1.0
        if row["result"] == "loss":
            return -1.0
        return np.nan

    df["flat_pl"] = df.apply(_flat_pl, axis=1)

    # Segmentation dimensions.
    df["confidence_bucket"] = pd.cut(
        df["confidence"], bins=[0, 40, 60, 80, 100],
        labels=["0-40", "40-60", "60-80", "80-100"], include_lowest=True,
    )
    df["ev_bucket"] = pd.cut(
        df["ev_pct"], bins=[-np.inf, 0.03, 0.05, 0.08, 0.12, np.inf],
        labels=["<3%", "3-5%", "5-8%", "8-12%", "12%+"],
    )
    df["side_type"] = np.where(df["american_odds"] < 0, "favorite", "underdog")
    df["venue_side"] = np.where(df["recommended_side"] == "home", "home", "road")
    df["clv_sign"] = np.where(
        df["clv_pct"].isna(), "unknown",
        np.where(df["clv_pct"] > 0, "CLV+", "CLV-"),
    )
    return df


def _stats(df: pd.DataFrame) -> dict:
    """Compute the metric bundle for a (sub)set of recommendations."""
    settled = df[df["settled"]]
    n_settled = len(settled)
    wins = int((settled["result"] == "win").sum())
    staked = float(settled["kelly_stake"].sum())
    kelly_pl = float(settled["profit_loss"].fillna(0).sum())
    flat_pl = float(settled["flat_pl"].sum()) if n_settled else 0.0
    clv_series = df["clv_pct"].dropna()

    return {
        "bets": len(df),
        "settled": n_settled,
        "wins": wins,
        "losses": int((settled["result"] == "loss").sum()),
        "hit_rate": (wins / n_settled) if n_settled else float("nan"),
        "avg_ev_pct": float(df["ev_pct"].mean() * 100) if len(df) else float("nan"),
        "kelly_roi": (kelly_pl / staked) if staked > 0 else float("nan"),
        "kelly_pl_units": kelly_pl,
        "flat_roi": (flat_pl / n_settled) if n_settled else float("nan"),
        "flat_pl_units": flat_pl,
        "avg_clv_pct": float(clv_series.mean()) if len(clv_series) else float("nan"),
        "clv_tracked": int(len(clv_series)),
    }


def _segment(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-bucket stats for one segmentation column, as a tidy DataFrame."""
    rows = []
    # Preserve categorical order where applicable.
    groups = df[column].cat.categories if hasattr(df[column], "cat") else sorted(df[column].dropna().unique())
    for value in groups:
        sub = df[df[column] == value]
        if sub.empty:
            continue
        stats = _stats(sub)
        stats = {column: str(value), **stats}
        rows.append(stats)
    return pd.DataFrame(rows)


def compute_performance(since: str | None = None) -> PerformanceReport:
    """Build the full performance report from stored recommendations."""
    df = recs.to_dataframe(since=since)
    if df.empty:
        return PerformanceReport(overall={"bets": 0, "settled": 0}, segments={})

    df = _prepare(df)
    overall = _stats(df)

    segments = {
        "By confidence bucket": _segment(df, "confidence_bucket"),
        "By EV bucket": _segment(df, "ev_bucket"),
        "Favorite vs underdog": _segment(df, "side_type"),
        "Home vs road": _segment(df, "venue_side"),
        "CLV positive vs negative": _segment(df, "clv_sign"),
    }
    return PerformanceReport(overall=overall, segments=segments)
