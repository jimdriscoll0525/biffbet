"""Data-confidence (0..100) for the totals model.

Mirrors the moneyline split: this score measures HOW TRUSTWORTHY THE INPUTS ARE
(both starters rated, offense/bullpen data present, weather feed up, lineups
confirmed, and our raw projection close to the market). It deliberately EXCLUDES
EV -- weighting the market blend by the model's self-assessed edge would be a
feedback loop that amplifies overconfidence (the same rule the moneyline obeys).

The score is used for BOTH:
  * the totals market-blend tier (config.totals.market_blend, via
    resolve_market_blend) -- the model earns more weight only on trustworthy data;
  * the displayed confidence + the sizing tier downgrade (confidence < bar).

Every input degrades to a 0 sub-score (never raises), so a missing feed lowers
confidence rather than crashing the slate.
"""
from __future__ import annotations


def compute_totals_confidence(profiles, weather, rd, home_lu, away_lu, config: dict) -> float:
    """0..100 data confidence for a totals pick.

    `profiles` is the pipeline's GameProfiles (carries the two PitcherProfiles +
    TeamProfiles); `weather` a WeatherEnv (or None); `rd` the RunDistribution;
    `home_lu`/`away_lu` LineupStatus objects (or None). Sub-scores are a weighted
    average (weights config.totals.confidence.weights, normalized at use).
    """
    cfg = config.get("totals", {}).get("confidence", {})
    weights = cfg.get("weights", {})

    home_pp, away_pp = profiles.home_pp, profiles.away_pp
    home_tp, away_tp = profiles.home_tp, profiles.away_tp

    # 1. Data completeness: both starters rated + offense (wRC+) + bullpen FIP
    #    present, over six possible inputs.
    present = sum((
        home_pp.primary_rate() is not None,
        away_pp.primary_rate() is not None,
        home_tp.offense_wrc_plus is not None,
        away_tp.offense_wrc_plus is not None,
        home_tp.bullpen_fip is not None,
        away_tp.bullpen_fip is not None,
    ))
    completeness = present / 6.0

    # 2. Weather: available (real read OR known fixed-dome) -> 1.0, else 0.
    weather_score = 1.0 if (weather is not None and weather.available) else 0.0

    # 3. Lineups: fraction of the two batting orders that are CONFIRMED (a
    #    projected/late lineup is timing, not a data gap, but confirmed orders
    #    are strictly better -- so projected scores partial).
    statuses = [getattr(lu, "status", None) for lu in (home_lu, away_lu)]
    lineups = sum(1 for s in statuses if s == "confirmed") / 2.0

    # 4. Market agreement: a SMALL raw-vs-market run gap means our independent
    #    build corroborates the market (trust the read); a large gap means we're
    #    likely missing something (don't). Linear to 0 at divergence_full_runs.
    full = float(cfg.get("divergence_full_runs", 1.75))
    if rd is not None and rd.raw_model_total is not None and rd.market_total:
        gap = abs(rd.raw_model_total - rd.market_total)
        agreement = max(0.0, 1.0 - gap / full) if full > 0 else 0.0
    else:
        agreement = 0.0

    parts = {
        "data_completeness": completeness,
        "weather": weather_score,
        "lineups": lineups,
        "market_agreement": agreement,
    }
    total_w = sum(float(weights.get(k, 0.0)) for k in parts) or 1.0
    score = sum(float(weights.get(k, 0.0)) * v for k, v in parts.items()) / total_w
    return round(max(0.0, min(1.0, score)) * 100.0, 1)
