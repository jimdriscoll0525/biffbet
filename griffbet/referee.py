"""Referee — read-only cross-model reporting that grades BiffBet AND GriffBet
over the same games.

This module READS BiffBet's already-stored pick data (read-only; it never alters
BiffBet's schema) and GriffBet's richer store, and produces calibration +
EV-monotonicity + sliced metrics for BOTH, so they can be compared apples to
apples on a process basis. Each model also carries its OWN W-L-P record
(computed strictly from its own store via `model_record`) so the site can show
it -- clearly labeled, with the small-sample warning, never as the verdict.

Probability calibration is produced separately for the RAW-model and BLENDED
probabilities, both evaluated on the committed pick side. EV monotonicity buckets
realized ROI by predicted-EV band so a non-monotone result is obvious.

Sharp-close CLV: BiffBet never captured a sharp closing line, so historically it
has NONE -- we do not backfill (per the freeze). Going forward, GriffBet records
the sharp close for every slate game; the referee joins BiffBet's committed bets
to GriffBet's sharp close by (date, game_id) to grade BiffBet's CLV-vs-sharp on
shared future games, and reports the historical games where no sharp close exists
as a GAP rather than inventing one.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pandas as pd

from mlb_value_bot.analysis.ev_calculator import american_to_decimal
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.referee")

SETTLED = {"win", "loss"}
_EPS = 1e-9


# --- pure metric helpers (unit-testable) -------------------------------------
def brier_score(probs: list[float], outcomes: list[int]) -> float | None:
    """Mean squared error of predicted prob vs realized outcome (lower better)."""
    if not probs:
        return None
    return round(sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs), 4)


def log_loss(probs: list[float], outcomes: list[int]) -> float | None:
    """Mean negative log-likelihood (lower better). Probs clipped off 0/1."""
    if not probs:
        return None
    tot = 0.0
    for p, o in zip(probs, outcomes):
        p = min(max(p, _EPS), 1 - _EPS)
        tot += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return round(tot / len(probs), 4)


def reliability_buckets(probs: list[float], outcomes: list[int], bands: list[float]) -> list[dict]:
    """Reliability curve: per predicted-prob band, mean predicted vs mean actual."""
    out = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        idx = [i for i, p in enumerate(probs) if (p >= lo and (p < hi or hi == bands[-1]))]
        if not idx:
            continue
        pr = [probs[i] for i in idx]
        oc = [outcomes[i] for i in idx]
        out.append({
            "band": f"{lo:.1f}-{hi:.1f}",
            "n": len(idx),
            "mean_predicted": round(sum(pr) / len(pr), 4),
            "mean_actual": round(sum(oc) / len(oc), 4),
        })
    return out


def _flat_pl(result: str, american: float) -> float | None:
    if result == "win":
        return american_to_decimal(american) - 1.0
    if result == "loss":
        return -1.0
    return None


def ev_monotonicity(rows: pd.DataFrame, ev_bands: list[float]) -> list[dict]:
    """Realized flat-1u ROI and Kelly ROI bucketed by predicted-EV band."""
    settled = rows[rows["result"].isin(SETTLED)].copy()
    if settled.empty:
        return []
    settled["flat_pl"] = settled.apply(
        lambda r: _flat_pl(r["result"], r["american_odds"]), axis=1)
    out = []
    for lo, hi in zip(ev_bands[:-1], ev_bands[1:]):
        sub = settled[(settled["ev_pct"] >= lo) & (settled["ev_pct"] < hi)]
        if sub.empty:
            continue
        staked = float(sub["kelly_stake"].sum())
        kelly_pl = float(sub["profit_loss"].fillna(0).sum())
        flat_pl = float(sub["flat_pl"].sum())
        out.append({
            "ev_band": f"{lo * 100:.0f}%..{hi * 100:.0f}%",
            "n": int(len(sub)),
            "wins": int((sub["result"] == "win").sum()),
            "losses": int((sub["result"] == "loss").sum()),
            "flat_roi": round(flat_pl / len(sub), 4),
            "kelly_roi": round(kelly_pl / staked, 4) if staked > 0 else None,
        })
    return out


# --- per-model extraction ----------------------------------------------------
@dataclass
class ModelSeries:
    """Committed, settled picks for one model with both probability views."""
    name: str
    df: pd.DataFrame          # is_value=1 rows (any result)
    blended_pick_prob: list[float]
    raw_pick_prob: list[float]
    outcomes: list[int]       # 1 win / 0 loss, aligned to the two prob lists (settled only)


def _reasoning(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _biff_raw_home_prob(reasoning: dict) -> float | None:
    ma = reasoning.get("market_anchor") or {}
    return ma.get("raw_model_home_prob")


def biffbet_series(df: pd.DataFrame) -> ModelSeries:
    """BiffBet committed picks. raw pick prob from reasoning.market_anchor."""
    bets = df[df.get("is_value", 1).fillna(1).astype(int) == 1].copy() if not df.empty else df
    blended, raw, outcomes = [], [], []
    for _, r in bets.iterrows():
        if r["result"] not in SETTLED:
            continue
        reasoning = _reasoning(r.get("reasoning_json"))
        raw_home = _biff_raw_home_prob(reasoning)
        side = r["recommended_side"]
        blended_pick = float(r["model_prob"])               # already pick-side
        raw_pick = (raw_home if side == "home" else 1.0 - raw_home) if raw_home is not None else None
        if raw_pick is None:
            continue
        blended.append(blended_pick)
        raw.append(float(raw_pick))
        outcomes.append(1 if r["result"] == "win" else 0)
    return ModelSeries("biffbet", bets, blended, raw, outcomes)


def griffbet_series(df: pd.DataFrame) -> ModelSeries:
    """GriffBet committed picks. raw pick prob from raw_model_prob (home) resolved
    to the committed pick side."""
    bets = df[df.get("is_value", 1).fillna(1).astype(int) == 1].copy() if not df.empty else df
    blended, raw, outcomes = [], [], []
    for _, r in bets.iterrows():
        if r["result"] not in SETTLED:
            continue
        side = r["recommended_side"]
        raw_home = r.get("raw_model_prob")
        if raw_home is None or pd.isna(raw_home):
            continue
        raw_pick = float(raw_home) if side == "home" else 1.0 - float(raw_home)
        blended.append(float(r["model_prob"]))
        raw.append(raw_pick)
        outcomes.append(1 if r["result"] == "win" else 0)
    return ModelSeries("griffbet", bets, blended, raw, outcomes)


def model_record(df: pd.DataFrame) -> dict:
    """W-L-P record over one model's OWN committed picks (is_value=1).

    Counted strictly from the DataFrame passed in -- the caller is responsible
    for passing each model its own store, so records can never blend across
    models. Pushes/voids are stake-neutral and broken out, never losses.
    """
    empty = {"wins": 0, "losses": 0, "pushes": 0, "voids": 0, "n_graded": 0}
    if df.empty or "result" not in df.columns:
        return empty
    bets = df[df.get("is_value", 1).fillna(1).astype(int) == 1]
    res = bets["result"]
    wins = int((res == "win").sum())
    losses = int((res == "loss").sum())
    pushes = int((res == "push").sum())
    voids = int((res == "void").sum())
    return {"wins": wins, "losses": losses, "pushes": pushes, "voids": voids,
            "n_graded": wins + losses + pushes + voids}


def _calibration(series: ModelSeries, bands: list[float]) -> dict:
    return {
        "n_graded": len(series.outcomes),
        "blended": {
            "brier": brier_score(series.blended_pick_prob, series.outcomes),
            "log_loss": log_loss(series.blended_pick_prob, series.outcomes),
            "reliability": reliability_buckets(series.blended_pick_prob, series.outcomes, bands),
        },
        "raw": {
            "brier": brier_score(series.raw_pick_prob, series.outcomes),
            "log_loss": log_loss(series.raw_pick_prob, series.outcomes),
            "reliability": reliability_buckets(series.raw_pick_prob, series.outcomes, bands),
        },
    }


def _clv_summary(df: pd.DataFrame, col: str) -> dict | None:
    if df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    return {"n": int(len(s)), "avg": round(float(s.mean()), 3),
            "positive": int((s > 0).sum())}


def biff_sharp_clv_via_join(biff_df: pd.DataFrame, griff_df: pd.DataFrame) -> dict:
    """Grade BiffBet CLV-vs-sharp by joining to GriffBet's sharp close on
    (date, game_id). Reports the historical gap (no sharp close) explicitly --
    no backfill, no BiffBet-schema change."""
    if biff_df.empty:
        return {"available": False, "reason": "no BiffBet picks"}
    bets = biff_df[biff_df.get("is_value", 1).fillna(1).astype(int) == 1]
    griff_idx = {}
    if not griff_df.empty:
        for _, g in griff_df.iterrows():
            griff_idx[(g["date"], int(g["game_id"]))] = g
    clvs, matched, gap = [], 0, 0
    for _, r in bets.iterrows():
        g = griff_idx.get((r["date"], int(r["game_id"])))
        if g is None or pd.isna(g.get("sharp_close_home_line")):
            gap += 1
            continue
        side = r["recommended_side"]
        sharp_close = g["sharp_close_home_line"] if side == "home" else g["sharp_close_away_line"]
        if pd.isna(sharp_close) or pd.isna(r.get("opening_line")):
            gap += 1
            continue
        try:
            clv = (american_to_decimal(int(r["opening_line"])) /
                   american_to_decimal(int(sharp_close)) - 1.0) * 100.0
        except (ValueError, ZeroDivisionError):
            gap += 1
            continue
        clvs.append(clv)
        matched += 1
    summary = {
        "available": matched > 0,
        "n_matched": matched,
        "n_gap_no_sharp_close": gap,
        "note": ("BiffBet captured no sharp closing line historically; CLV-vs-sharp "
                 "is available only for games GriffBet priced going forward (joined "
                 "read-only). Historical picks are reported as a gap, not backfilled."),
    }
    if clvs:
        summary["avg"] = round(sum(clvs) / len(clvs), 3)
        summary["positive"] = int(sum(1 for c in clvs if c > 0))
    return summary


def compute_referee(
    biff_df: pd.DataFrame,
    griff_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Build the full referee snapshot from BiffBet + GriffBet stored picks."""
    ref = config.get("referee", {})
    bands = ref.get("reliability_bands", [i / 10 for i in range(11)])
    ev_bands = ref.get("ev_bands", [-1.0, 0.0, 0.03, 0.05, 0.08, 0.12, 1.0])
    min_graded = int(ref.get("min_graded_bets", 100))

    biff = biffbet_series(biff_df)
    griff = griffbet_series(griff_df)

    biff_bets = biff_df[biff_df.get("is_value", 1).fillna(1).astype(int) == 1] if not biff_df.empty else biff_df
    griff_bets = griff_df[griff_df.get("is_value", 1).fillna(1).astype(int) == 1] if not griff_df.empty else griff_df

    return {
        "min_graded_bets": min_graded,
        "small_sample": {
            "biffbet": len(biff.outcomes) < min_graded,
            "griffbet": len(griff.outcomes) < min_graded,
        },
        "models": {
            "biffbet": {
                "n_graded": len(biff.outcomes),
                "record": model_record(biff_df),
                "calibration": _calibration(biff, bands),
                "ev_monotonicity": ev_monotonicity(biff_bets, ev_bands) if not biff_bets.empty else [],
                "clv_vs_best": _clv_summary(biff_bets, "clv_pct"),
                "clv_vs_sharp": biff_sharp_clv_via_join(biff_df, griff_df),
            },
            "griffbet": {
                "n_graded": len(griff.outcomes),
                "record": model_record(griff_df),
                "calibration": _calibration(griff, bands),
                "ev_monotonicity": ev_monotonicity(griff_bets, ev_bands) if not griff_bets.empty else [],
                "clv_vs_best": _clv_summary(griff_bets, "clv_pct"),
                "clv_blended_vs_sharp": _clv_summary(griff_bets, "clv_blended_vs_sharp"),
                "clv_raw_vs_sharp": _clv_summary(griff_bets, "clv_raw_vs_sharp"),
            },
        },
    }
