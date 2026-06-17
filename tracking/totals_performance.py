"""Performance analytics for the TOTALS model.

The headline metric is **CLV in probability points vs the sharp totals close**,
NOT win/loss -- the paper-trade gate is meant to be lifted on proven positive
CLV, never a hot record. ROI/hit-rate are reported (paper) but secondary.

Parallel to tracking/performance.py but reads the totals table and uses the
totals-specific columns (bet_odds, clv_pp, pick_side, stability, tier).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlb_value_bot.analysis.ev_calculator import american_to_decimal
from mlb_value_bot.tracking import totals_recommendations as totals
from mlb_value_bot.utils import get_logger

log = get_logger("tracking.totals_performance")

SETTLED = {"win", "loss"}


@dataclass
class TotalsPerformanceReport:
    overall: dict
    segments: dict[str, pd.DataFrame]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["settled"] = df["result"].isin(SETTLED)

    def _flat_pl(row) -> float:
        if row["result"] == "win":
            return american_to_decimal(row["bet_odds"]) - 1.0
        if row["result"] == "loss":
            return -1.0
        return np.nan      # push / void / pending excluded from flat ROI

    df["flat_pl"] = df.apply(_flat_pl, axis=1)
    df["confidence_bucket"] = pd.cut(
        df["confidence"], bins=[0, 40, 60, 80, 100],
        labels=["0-40", "40-60", "60-80", "80-100"], include_lowest=True,
    )
    df["ev_bucket"] = pd.cut(
        df["ev_pct"], bins=[-np.inf, 0.03, 0.05, 0.08, np.inf],
        labels=["<3%", "3-5%", "5-8%", "8%+"],
    )
    df["clv_sign"] = np.where(
        df["clv_pp"].isna(), "unknown",
        np.where(df["clv_pp"] > 0, "CLV+", "CLV-"),
    )
    df["stability"] = df["stability"].fillna("n/a").astype("string")
    df["pick_side"] = df["pick_side"].astype("string")
    df["tier"] = df["tier"].fillna("n/a").astype("string")
    return df


def _stats(df: pd.DataFrame) -> dict:
    settled = df[df["settled"]]
    n_settled = len(settled)
    wins = int((settled["result"] == "win").sum())
    staked = float(settled["kelly_stake"].sum())
    kelly_pl = float(settled["profit_loss"].fillna(0).sum())
    flat_pl = float(settled["flat_pl"].sum()) if n_settled else 0.0
    clv = df["clv_pp"].dropna()
    return {
        "bets": len(df),
        "settled": n_settled,
        "wins": wins,
        "losses": int((settled["result"] == "loss").sum()),
        "pushes": int((df["result"] == "push").sum()),
        "hit_rate": (wins / n_settled) if n_settled else float("nan"),
        "avg_ev_pct": float(df["ev_pct"].mean() * 100) if len(df) else float("nan"),
        "kelly_roi": (kelly_pl / staked) if staked > 0 else float("nan"),
        "kelly_pl_units": kelly_pl,
        "flat_roi": (flat_pl / n_settled) if n_settled else float("nan"),
        "flat_pl_units": flat_pl,
        # CLV in probability points vs the sharp totals close -- the scoreboard.
        "avg_clv_pp": float(clv.mean()) if len(clv) else float("nan"),
        "clv_positive_rate": float((clv > 0).mean()) if len(clv) else float("nan"),
        "clv_tracked": int(len(clv)),
    }


def _segment(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    groups = df[column].cat.categories if hasattr(df[column], "cat") else sorted(df[column].dropna().unique())
    for value in groups:
        sub = df[df[column] == value]
        if sub.empty:
            continue
        rows.append({column: str(value), **_stats(sub)})
    return pd.DataFrame(rows)


def compute_totals_performance(since: str | None = None) -> TotalsPerformanceReport:
    df = totals.to_dataframe(since=since)
    if df.empty:
        return TotalsPerformanceReport(overall={"bets": 0, "settled": 0}, segments={})
    if "is_value" in df.columns:
        df = df[df["is_value"].fillna(1).astype(int) == 1]
    if df.empty:
        return TotalsPerformanceReport(overall={"bets": 0, "settled": 0}, segments={})

    df = _prepare(df)
    overall = _stats(df)
    segments = {
        "By CLV sign": _segment(df, "clv_sign"),
        "Over vs under": _segment(df, "pick_side"),
        "By edge stability": _segment(df, "stability"),
        "By bet tier": _segment(df, "tier"),
        "By confidence bucket": _segment(df, "confidence_bucket"),
        "By EV bucket": _segment(df, "ev_bucket"),
    }
    return TotalsPerformanceReport(overall=overall, segments=segments)
