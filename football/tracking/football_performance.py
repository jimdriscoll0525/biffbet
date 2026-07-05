"""Football records, CLV summaries, the pick-distribution monitor, and
calibration — everything the site's football tab reads via football_snapshot.

THE RECORD RULE (the GriffBet record-bug lesson, commit 3fba802): every
aggregate here is computed from football's OWN store and ALWAYS filtered by
model_tag x league x market (or an explicit 'all' bucket that is still
model_tag-filtered and is_value-only). Pushes are stake-neutral and broken out
(never losses); voids are excluded from `graded`. Locked by
test_record_filtering_by_tag_league_market.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from mlb_value_bot.football.tracking import football_store as store
from mlb_value_bot.utils import get_logger

log = get_logger("football.tracking.performance")

_SETTLED = ("win", "loss", "push", "void")


def _bets(df: pd.DataFrame, model_tag: str, league: str | None,
          market: str | None) -> pd.DataFrame:
    """The one and only row filter aggregates may use."""
    out = df[(df["is_value"] == 1) & (df["model_tag"] == model_tag)]
    if league is not None:
        out = out[out["league"] == league]
    if market is not None:
        out = out[out["market"] == market]
    return out


def record(df: pd.DataFrame, model_tag: str, league: str | None = None,
           market: str | None = None) -> dict:
    """W-L-P record + flat ROI + CLV summary for one tag/league/market cell."""
    bets = _bets(df, model_tag, league, market)
    settled = bets[bets["result"].isin(_SETTLED)]
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())
    pushes = int((settled["result"] == "push").sum())
    voids = int((settled["result"] == "void").sum())
    graded = wins + losses + pushes            # voids excluded
    staked = settled.loc[settled["result"].isin(("win", "loss")), "flat_stake"].sum()
    pl = settled["profit_loss"].fillna(0.0).sum()
    clv = bets["clv_pp"].dropna()
    return {
        "league": league or "all", "market": market or "all", "model_tag": model_tag,
        "bets": int(len(bets)), "graded": graded,
        "wins": wins, "losses": losses, "pushes": pushes, "voids": voids,
        "pending": int((bets["result"] == "pending").sum()),
        "flat_pl_units": round(float(pl), 4),
        "flat_roi": round(float(pl / staked), 4) if staked > 0 else None,
        "avg_clv_pp": round(float(clv.mean()), 3) if len(clv) else None,
        "clv_tracked": int(len(clv)),
        "clv_positive": int((clv > 0).sum()),
    }


def pick_distribution(df: pd.DataFrame, model_tag: str, config: dict,
                      league: str | None = None) -> dict:
    """Rolling over/under share across the last `window` committed totals
    picks — the over-bias tripwire. Alert is symmetric and needs min_picks."""
    cfg = config.get("distribution_monitor", {})
    window = int(cfg.get("window", 50))
    alert_share = float(cfg.get("alert_share", 0.60))
    min_picks = int(cfg.get("min_picks", 25))

    totals = _bets(df, model_tag, league, "total").sort_values("created_at").tail(window)
    n = len(totals)
    overs = int((totals["pick_side"] == "over").sum())
    over_share = overs / n if n else None
    alert = bool(n >= min_picks and over_share is not None
                 and (over_share > alert_share or over_share < 1.0 - alert_share))
    return {
        "league": league or "all", "window": window, "n": n,
        "overs": overs, "unders": n - overs,
        "over_share": round(over_share, 3) if over_share is not None else None,
        "alert_share": alert_share, "min_picks": min_picks, "alert": alert,
    }


def calibration(df: pd.DataFrame, model_tag: str, config: dict) -> dict:
    """Brier + reliability buckets over graded picks (blended P(pick side) vs
    won), reusing GriffBet's referee helpers — the same calibration-table
    approach the MLB review produced. Pushes/voids excluded (no outcome)."""
    from mlb_value_bot.griffbet.referee import brier_score, reliability_buckets

    bets = _bets(df, model_tag, None, None)
    graded = bets[bets["result"].isin(("win", "loss"))]
    if graded.empty:
        return {"n_graded": 0, "brier": None, "reliability": []}
    probs = graded["model_prob"].astype(float).tolist()
    outcomes = (graded["result"] == "win").astype(int).tolist()
    bands = config.get("calibration_bands",
                       [0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
    return {
        "n_graded": int(len(graded)),
        "brier": brier_score(probs, outcomes),
        "reliability": reliability_buckets(probs, outcomes, bands),
    }


def compute_snapshot(config: dict) -> dict[str, dict]:
    """All football_snapshot scopes, keyed like referee/performance snapshots:
    record:<league>:<market>, clv:<league>, distribution:<league>:total,
    calibration:<model_tag>."""
    model_tag = config.get("model_tag", "matchup_v1")
    df = store.to_dataframe()
    if df.empty:
        return {}

    scopes: dict[str, dict] = {}
    for league in ("nfl", "cfb"):
        for market in ("spread", "total", None):
            r = record(df, model_tag, league, market)
            scopes[f"record:{league}:{market or 'all'}"] = r
        scopes[f"distribution:{league}:total"] = pick_distribution(df, model_tag, config, league)
    scopes["record:all:all"] = record(df, model_tag)
    scopes["distribution:all:total"] = pick_distribution(df, model_tag, config)
    scopes[f"calibration:{model_tag}"] = calibration(df, model_tag, config)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for payload in scopes.values():
        payload["computed_at"] = stamp
    return scopes
