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

import json

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
    """Add derived columns used for segmentation and flat-stake P&L.

    Bucket definitions live in rsi/segments.py (the single source shared with
    the weekly RSI review) -- this is a thin delegation that keeps the exact
    columns, labels and categorical dtypes this module has always produced:
    settled, flat_pl, confidence_bucket, ev_bucket, kelly_bucket, side_type,
    venue_side, clv_sign, bet_tier, blend_tier, stability, sharp_fade,
    odds_bucket (plus the RSI-only lenses, which the report ignores).
    """
    if df.empty:
        return df
    from mlb_value_bot.rsi.segments import prepare_mlb

    return prepare_mlb(df)


def _extract_bet_tier(raw: str | None) -> str:
    if not isinstance(raw, str) or not raw:
        return "n/a"
    try:
        return json.loads(raw).get("bet_sizing", {}).get("tier") or "n/a"
    except (json.JSONDecodeError, AttributeError):
        return "n/a"


def _extract_blend_tier(raw: str | None) -> str:
    if not isinstance(raw, str) or not raw:
        return "n/a"
    try:
        return json.loads(raw).get("market_anchor", {}).get("blend_tier") or "n/a"
    except (json.JSONDecodeError, AttributeError):
        return "n/a"


def _extract_stability(raw: str | None) -> str:
    if not isinstance(raw, str) or not raw:
        return "n/a"
    try:
        return json.loads(raw).get("stability", {}).get("label") or "n/a"
    except (json.JSONDecodeError, AttributeError):
        return "n/a"


def _sharp_fade_pp(row) -> float:
    """Signed sharp-fade in probability points, reconstructed per row.

    Positive = our blended prob on the pick side exceeds the sharp consensus
    (we are FADING the sharps); negative = the sharps are even more bullish
    than we are (sharp support). Same sign convention as pipeline's
    sharp_fade_pp. Reconstructed from market_intel.sharp_devig_home +
    the stored pick-side model_prob, so it covers every row that captured
    market intel — there is no separate stored field to depend on.
    """
    raw = row.get("reasoning_json")
    if not isinstance(raw, str) or not raw:
        return np.nan
    try:
        sharp_home = json.loads(raw).get("market_intel", {}).get("sharp_devig_home")
    except (json.JSONDecodeError, AttributeError):
        return np.nan
    if sharp_home is None:
        return np.nan
    sharp_pick = sharp_home if row["recommended_side"] == "home" else 1.0 - sharp_home
    return (float(row["model_prob"]) - float(sharp_pick)) * 100.0


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

    # Only actual bets (is_value=1) count toward performance. Non-value rows
    # exist on the site to show "the full slate we looked at today" but they
    # are analyses, not bets -- including them would inflate `bets` and drag
    # every ROI / hit-rate / CLV figure with non-action games.
    if "is_value" in df.columns:
        df = df[df["is_value"].fillna(1).astype(int) == 1]
    if df.empty:
        return PerformanceReport(overall={"bets": 0, "settled": 0}, segments={})

    df = _prepare(df)
    overall = _stats(df)

    segments = {
        "By edge stability": _segment(df, "stability"),
        "By sharp fade": _segment(df, "sharp_fade"),
        "By odds bucket": _segment(df, "odds_bucket"),
        "By confidence bucket": _segment(df, "confidence_bucket"),
        "By EV bucket": _segment(df, "ev_bucket"),
        "By Kelly bucket": _segment(df, "kelly_bucket"),
        "By bet tier": _segment(df, "bet_tier"),
        "By blend tier": _segment(df, "blend_tier"),
        "Favorite vs underdog": _segment(df, "side_type"),
        "Home vs road": _segment(df, "venue_side"),
        "CLV positive vs negative": _segment(df, "clv_sign"),
    }
    return PerformanceReport(overall=overall, segments=segments)
