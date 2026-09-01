"""NFL ATS core rating — PURE (M2 of the ATS module, approved 2026-09-01).

rating_pts(team) = plays_pg * (epa_share * net_epa + sr_share * sr_epa_equiv * net_sr)

  net_epa = off EPA/play - def EPA/play allowed     (garbage-time re-weighted,
  net_sr  = off success rate - def success allowed   recency-decayed, prior-
                                                     blended)

EPA/play x plays is points by construction, so the rating is a POINTS-scale
power number; a matchup's core margin is rating_home - rating_away + HFA —
the anchor the (M3) matchup layer tilts, exactly the CFB Elo-anchor shape.

Every knob lives in config `nfl_rating.*` and is deliberately provisional:
the M6 walk-forward backtest (2023-24 tune / 2025 holdout) fits decay,
gt_weight, the blend schedule, and the epa/sr shares before this ever
prices a live board (`projections.margin_anchor_nfl` stays null until then).
"""
from __future__ import annotations

import pandas as pd


def _weighted_stats(ws: pd.DataFrame, as_of_week: int, decay: float,
                    gt_weight: float, recency: bool) -> pd.DataFrame:
    """Recency+garbage-time weighted per-team stats from week aggregates.
    Uses ONLY weeks strictly before `as_of_week` (walk-forward safe)."""
    d = ws[ws["week"] < as_of_week].copy()
    if d.empty:
        return pd.DataFrame()
    w = decay ** (as_of_week - 1 - d["week"]) if recency else 1.0
    for side in ("off", "def"):
        d[f"{side}_plays_w"] = w * (d[f"{side}_all_ngt_plays"] + gt_weight * d[f"{side}_all_gt_plays"])
        d[f"{side}_epa_w"] = w * (d[f"{side}_all_ngt_epa"] + gt_weight * d[f"{side}_all_gt_epa"])
        d[f"{side}_succ_w"] = w * (d[f"{side}_all_ngt_succ"] + gt_weight * d[f"{side}_all_gt_succ"])
    d["games_w"] = (w if recency else 1.0) * d["off_games"].clip(lower=1)
    d["plays_raw"] = d["off_all_ngt_plays"] + d["off_all_gt_plays"]
    g = d.groupby("team")
    out = pd.DataFrame({
        "off_epa": g["off_epa_w"].sum() / g["off_plays_w"].sum(),
        "off_sr": g["off_succ_w"].sum() / g["off_plays_w"].sum(),
        "def_epa": g["def_epa_w"].sum() / g["def_plays_w"].sum(),
        "def_sr": g["def_succ_w"].sum() / g["def_plays_w"].sum(),
        "plays_pg": g["plays_raw"].sum() / g["off_games"].sum().clip(lower=1),
        "games": g["off_games"].sum(),
    })
    return out


def prior_weight(as_of_week: int, config: dict) -> float:
    cfg = config.get("nfl_rating", {})
    w1 = float(cfg.get("prior_weight_week1", 0.8))
    out_week = int(cfg.get("prior_out_week", 8))
    if as_of_week >= out_week:
        return 0.0
    if as_of_week <= 1:
        return w1
    return w1 * (out_week - as_of_week) / (out_week - 1)


def team_ratings(cur_ws: pd.DataFrame, prior_ws: pd.DataFrame,
                 as_of_week: int, config: dict) -> pd.DataFrame:
    """Points-scale team ratings as of `as_of_week` (exclusive). Returns a
    frame indexed by team: components + `rating_pts`. Empty when neither
    season has data."""
    cfg = config.get("nfl_rating", {})
    decay = float(cfg.get("decay", 0.90))
    gt_w = float(cfg.get("gt_weight", 0.2))
    epa_share = float(cfg.get("epa_share", 0.7))
    sr_share = float(cfg.get("sr_share", 0.3))
    sr_equiv = float(cfg.get("sr_epa_equiv", 1.6))

    cur = _weighted_stats(cur_ws, as_of_week, decay, gt_w, recency=True)         if cur_ws is not None and not cur_ws.empty else pd.DataFrame()
    # Prior season: whole-season aggregate, no recency inside it.
    pri = _weighted_stats(prior_ws, 99, decay, gt_w, recency=False)         if prior_ws is not None and not prior_ws.empty else pd.DataFrame()

    pw = prior_weight(as_of_week, config)
    if cur.empty and pri.empty:
        return pd.DataFrame()
    if cur.empty:
        blended, pw_eff = pri, 1.0
    elif pri.empty or pw <= 0:
        blended, pw_eff = cur, 0.0
    else:
        idx = cur.index.union(pri.index)
        c, p = cur.reindex(idx), pri.reindex(idx)
        blended = (1 - pw) * c + pw * p
        # Teams missing a side keep the side they have.
        blended = blended.where(c.notna() | p.isna(), p)
        blended = blended.where(p.notna() | c.isna(), c)
        blended["games"] = c["games"].fillna(0)
        pw_eff = pw

    net_epa = blended["off_epa"] - blended["def_epa"]
    net_sr = blended["off_sr"] - blended["def_sr"]
    out = blended.copy()
    out["net_epa"] = net_epa
    out["net_sr"] = net_sr
    out["prior_weight"] = pw_eff
    out["rating_pts"] = out["plays_pg"] * (
        epa_share * net_epa + sr_share * sr_equiv * net_sr)
    return out.sort_values("rating_pts", ascending=False)
