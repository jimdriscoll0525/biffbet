"""Totals (over/under) evaluation -- the parallel pipeline to the moneyline.

Independent of the moneyline in its math and data: own line, own de-vig, own
sharp consensus, own close. Mirrors the moneyline PHILOSOPHY (transparent,
market-anchored, edge-measured via CLV) but shares NOTHING of its probability
machinery. Reuses the per-game profiles built once in `pipeline.analyze_slate`.

Flow per game (all degrade-safe; a missing input lowers confidence / flags
fragility, never crashes or fabricates a tilt):

  1. Totals market intel from the raw per-book payload (line, over/under prices,
     de-vigged P(over), sharp vs square consensus, sharp close).
  2. Purpose-built run distribution (negative binomial) anchored to the market.
  3. Blend the model's conditional P(over) toward the de-vigged market, tiered by
     totals DATA confidence (more model weight only on trustworthy inputs).
  4. Sanity guards (market-total bounds, raw-vs-market run divergence, sharp
     fade, implausible EV).
  5. EV + quarter-Kelly off the blended prob; tier + stability sizing.
  6. PAPER gate: every pick is simulated until CLV vs the totals close proves out.

PAPER-ONLY: while config.totals.paper_only is true, picks are flagged paper and
the UI marks them simulated. CLV is graded against the SHARP TOTALS close, never
the moneyline close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mlb_value_bot.analysis.ev_calculator import devigged_market_probs  # noqa: F401 (kept for parity/tests)
from mlb_value_bot.analysis.run_distribution import run_distribution
from mlb_value_bot.analysis.totals_confidence import compute_totals_confidence
from mlb_value_bot.analysis.totals_ev import (
    classify_totals_tier,
    evaluate_ou_sides,
    kelly_cap_for_tier,
)
from mlb_value_bot.analysis.totals_market import compute_totals_market, sharp_totals_close
from mlb_value_bot.analysis.totals_stability import classify_totals_stability
from mlb_value_bot.analysis.win_probability import resolve_market_blend
from mlb_value_bot.data.weather import weather_env
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("pipeline.totals")


@dataclass
class TotalsAnalysis:
    game_id: int
    game_date: str
    home_team: str
    away_team: str
    status: str
    home_pitcher: str | None
    away_pitcher: str | None
    game_datetime: str | None = None
    skipped_reason: str | None = None

    # Market + model carriers (kept even on a sanity skip so a committed paper
    # bet on a now-skipped game can still get its sharp-close + CLV refreshed).
    intel: object | None = None          # TotalsMarketIntel
    sharp_close: object | None = None    # SharpTotalsLine | None
    weather: object | None = None        # WeatherEnv | None
    rd: object | None = None             # RunDistribution | None
    market_total: float | None = None

    # Probabilities (all conditional on no-push, so model<->market comparable).
    model_p_over: float | None = None
    market_devig_over: float | None = None
    blended_over: float | None = None
    blend: float = 1.0
    blend_tier: str = "fixed"
    data_confidence: float = 0.0

    # Pick / EV / sizing.
    evals: dict | None = None            # {"over": SideEvaluation, "under": SideEvaluation}
    pick_side: str | None = None         # "over" | "under"
    confidence: float = 0.0
    stability: object | None = None      # TotalsStability
    tier: str = "pass"
    tier_kelly_cap: float = 0.0
    tier_reasons: list = field(default_factory=list)

    # Paper-trade gate + weather hold.
    paper: bool = True
    weather_held: bool = False           # held to analysis-only by the weather rule
    roof_unverified: bool = False        # held: roof state assumed, not verified
                                         # (config totals.weather.require_verified_roof)
    flags: list = field(default_factory=list)
    _orig_kelly: float | None = None
    # RSI (2026-09-26): model version stamp, persisted as model_tag.
    model_tag: str = "totals_v1"

    @property
    def best_eval(self):
        if not self.evals or not self.pick_side:
            return None
        return self.evals[self.pick_side]

    def pass_reason(self, threshold: float) -> str | None:
        """Why this total was NOT bet (RSI vocabulary), None for a value pick or
        when nothing is persisted (no pick / no distribution). Vocabulary:
        below_threshold | weather_held | roof_unverified | skip:divergence |
        skip:max_ev | skip:sharp_fade, several joined with '+'."""
        be = self.best_eval
        if be is None or self.rd is None or self.is_value(threshold):
            return None
        parts: list[str] = []
        kind = _skip_kind(self.skipped_reason)
        if kind:
            parts.append(kind)
        if self.weather_held:
            parts.append("weather_held")
        if self.roof_unverified:
            parts.append("roof_unverified")
        if be.ev_pct < threshold or not parts:
            parts.append("below_threshold")
        return "+".join(dict.fromkeys(parts))

    def opening_devig_for(self, side: str) -> float | None:
        """De-vigged market P(side) at the bet book (the CLV ENTRY reference)."""
        if self.market_devig_over is None:
            return None
        return self.market_devig_over if side == "over" else 1.0 - self.market_devig_over

    def sharp_close_devig_for(self, side: str) -> float | None:
        """De-vigged P(side) at the sharp totals close (the CLV CLOSE reference)."""
        if self.sharp_close is None:
            return None
        return self.sharp_close.devig_over if side == "over" else 1.0 - self.sharp_close.devig_over

    def is_value(self, threshold: float) -> bool:
        """A simulated 'value' totals pick: clears EV + positive Kelly, not
        skipped, and not held back by the weather rule. (Still PAPER while
        config.totals.paper_only is true -- value just means 'we'd bet it'.)"""
        be = self.best_eval
        if be is None or self.skipped_reason or self.weather_held or self.roof_unverified:
            return False
        return be.ev_pct >= threshold and be.kelly_stake > 0

    def reasoning(self) -> dict:
        """Full JSON-able breakdown for the DB / site (mirrors the moneyline)."""
        rd = self.rd
        data: dict = {"market_type": "totals", "paper": self.paper, "model_tag": self.model_tag}
        if self.skipped_reason:
            data["skipped_reason"] = self.skipped_reason
        if rd is not None:
            data["run_distribution"] = {
                "raw_model_total": rd.raw_model_total,
                "expected_total": rd.expected_total,
                # Mean provenance, explicit (not just prose in notes): the mean
                # is anchor_mean + tilt, NOT the sum of the component runs.
                "anchor_mean": rd.anchor_mean,
                "tilt": rd.tilt,
                "variance": rd.variance,
                "home_runs": rd.home_runs,
                "away_runs": rd.away_runs,
                "p_over": rd.p_over,
                "p_under": rd.p_under,
                "p_push": rd.p_push,
                "components": list(rd.components),
                "notes": list(rd.notes),
            }
        if self.weather is not None:
            w = self.weather
            data["weather"] = {
                "available": w.available, "multiplier": w.multiplier, "roof": w.roof,
                "temp_c": w.temp_c, "wind_kmh": w.wind_kmh,
                "wind_out_component": w.wind_out_component, "note": w.note,
                # Precipitation at first pitch (logged only, not a model input).
                "precip_mm": getattr(w, "precip_mm", None),
                "precip_prob": getattr(w, "precip_prob", None),
            }
        if self.intel is not None:
            mi = self.intel
            data["market"] = {
                "line": mi.bet_line, "over_price": mi.bet_over_price, "under_price": mi.bet_under_price,
                "best_over_price": mi.best_over_price, "best_under_price": mi.best_under_price,
                "market_devig_over": mi.bet_devig_over,
                "sharp_line": mi.sharp_line, "sharp_devig_over": mi.sharp_devig_over,
                "square_line": mi.square_line, "square_devig_over": mi.square_devig_over,
                "n_sharp": mi.n_sharp, "n_square": mi.n_square, "n_total": mi.n_total,
            }
        data["market_anchor"] = {
            "model_p_over": self.model_p_over,
            "market_devig_over": self.market_devig_over,
            "blend_weight": round(self.blend, 3),
            "blend_tier": self.blend_tier,
            "data_confidence": self.data_confidence,
            "blended_p_over": self.blended_over,
        }
        be = self.best_eval
        data["pick"] = {
            "side": self.pick_side,
            "line": self.market_total,
            "price": be.american_odds if be is not None else None,
            "ev_pct": round(be.ev_pct, 4) if be is not None else None,
            "kelly_stake": round(be.kelly_stake, 6) if be is not None else None,
            "raw_kelly": self._orig_kelly,
            "tier": self.tier,
            "kelly_cap": self.tier_kelly_cap,
            "tier_reasons": list(self.tier_reasons),
        }
        if self.stability is not None:
            s = self.stability
            data["stability"] = {
                "label": s.label,
                "hard_fragile_signals": list(s.hard_fragile_signals),
                "drivers": list(s.drivers),
            }
        if self.sharp_close is not None:
            sc = self.sharp_close
            data["sharp_close"] = {
                "book": sc.book, "line": sc.line,
                "over_price": sc.over_price, "under_price": sc.under_price,
                "devig_over": sc.devig_over,
            }
        if self.flags:
            data["flags"] = list(self.flags)
        data["pitchers"] = {"home": self.home_pitcher, "away": self.away_pitcher}
        data["game_datetime"] = self.game_datetime
        return data


# RSI pass vocabulary: prose skip reason -> canonical `skip:<kind>` label.
_SKIP_KINDS: tuple[tuple[str, str], ...] = (
    ("diverge", "skip:divergence"),
    ("implausible EV", "skip:max_ev"),
    ("fading sharp", "skip:sharp_fade"),
)


def _skip_kind(reason: str | None) -> str | None:
    if not reason:
        return None
    for needle, kind in _SKIP_KINDS:
        if needle in reason:
            return kind
    return "skip:other"


def evaluate_totals_game(scheduled, game_odds, profiles, weather, config=None) -> TotalsAnalysis:
    """Run the totals model + EV evaluation for one game. Always returns a
    TotalsAnalysis (skip reasons live on `.skipped_reason`); never raises.

    The three post-market sanity guards (divergence / max EV / sharp fade) do
    NOT early-return (RSI phase 2): the pick is still evaluated in full and
    persisted as an is_value=0 pass with pass_reason skip:<kind>; the skip
    forces tier pass + stake 0 and is_value() is False. First to fire wins.
    Pre-market skips (no odds / no market / implausible total / no
    distribution) still return early with nothing to persist."""
    config = config or load_config()
    tcfg = config.get("totals", {})
    analysis = TotalsAnalysis(
        game_id=scheduled.game_id, game_date=scheduled.game_date,
        home_team=scheduled.home_team, away_team=scheduled.away_team, status=scheduled.status,
        home_pitcher=scheduled.home_pitcher.name, away_pitcher=scheduled.away_pitcher.name,
        game_datetime=scheduled.game_datetime, weather=weather,
        paper=bool(tcfg.get("paper_only", True)),
    )

    if not scheduled.is_playable:
        analysis.skipped_reason = f"not playable ({scheduled.status})"
        return analysis
    if game_odds is None:
        analysis.skipped_reason = "no odds found"
        return analysis
    if profiles is None:
        analysis.skipped_reason = "no metric profiles"
        return analysis

    devig_method = config.get("ev", {}).get("devig_method", "power")
    odds_cfg = config.get("odds_api", {})
    intel = compute_totals_market(
        game_odds.all_books,
        bet_book=odds_cfg.get("bet_bookmaker"),
        sharp_books=odds_cfg.get("sharp_bookmakers") or [],
        square_books=odds_cfg.get("square_bookmakers") or [],
        devig_method=devig_method,
    )
    analysis.intel = intel
    # Sharp totals close (Pinnacle-preferred) -- kept even on later skips so a
    # committed paper bet's CLV keeps refreshing.
    analysis.sharp_close = sharp_totals_close(
        game_odds.all_books, tcfg.get("odds", {}).get("sharp_close_priority") or [], devig_method,
    )

    if not intel.available:
        analysis.skipped_reason = "no totals market posted"
        return analysis

    # Sanity: implausible posted total -> bad/stale feed.
    sanity = tcfg.get("sanity", {})
    lo = float(sanity.get("market_total_min", 5.0))
    hi = float(sanity.get("market_total_max", 16.0))
    market_total = float(intel.bet_line)
    analysis.market_total = market_total
    if not (lo <= market_total <= hi):
        analysis.skipped_reason = f"implausible market total {market_total:.1f} (outside [{lo:.0f},{hi:.0f}])"
        return analysis

    # Run distribution (anchored to the market's de-vigged P(over)).
    rd = run_distribution(
        profiles.home_tp, profiles.away_tp, profiles.home_pp, profiles.away_pp,
        market_total, intel.bet_devig_over, weather, config,
    )
    analysis.rd = rd
    if not rd.available or rd.p_over is None:
        analysis.skipped_reason = "run distribution unavailable"
        return analysis

    # Condition the model probability on no-push so it's comparable to the
    # two-way de-vigged market (pushes are stake-neutral, handled in grading).
    denom = (rd.p_over or 0.0) + (rd.p_under or 0.0)
    model_cond_over = (rd.p_over / denom) if denom > 0 else 0.5
    analysis.model_p_over = round(model_cond_over, 4)
    analysis.market_devig_over = round(intel.bet_devig_over, 4) if intel.bet_devig_over is not None else None

    # Totals DATA confidence -> market-blend tier (more model weight only when
    # inputs are trustworthy; EV is excluded by design).
    data_conf = compute_totals_confidence(profiles, weather, rd, profiles.home_lu, profiles.away_lu, config)
    blend, blend_tier = resolve_market_blend(data_conf, tcfg)
    market_over = intel.bet_devig_over if intel.bet_devig_over is not None else model_cond_over
    blended_over = blend * model_cond_over + (1.0 - blend) * market_over
    analysis.data_confidence = data_conf
    analysis.blend = blend
    analysis.blend_tier = blend_tier
    analysis.blended_over = round(blended_over, 4)

    # Sanity: raw-vs-market run divergence. A big gap almost always means we're
    # missing what the market has (weather, a scratch), not that we found edge.
    # Measured against the market-implied MEAN (anchor), not the posted line --
    # same mean-vs-median reasoning as the tilt: raw is a mean, the line is ~a
    # median, so a line-based guard would be asymmetric (looser on overs).
    max_div = float(sanity.get("max_total_divergence_runs", 1.75))
    anchor_ref = rd.anchor_mean if rd.anchor_mean is not None else market_total
    divergence = abs(rd.raw_model_total - anchor_ref)
    if divergence > max_div:
        analysis.skipped_reason = (
            f"raw projected total {rd.raw_model_total:.2f} vs market-implied mean {anchor_ref:.2f} "
            f"diverge by {divergence:.2f} > {max_div:.2f} runs - likely missing weather / a scratch"
        )

    # EV + quarter-Kelly on the blended P(over) at the actual over/under prices.
    evals = evaluate_ou_sides(
        blended_over, intel.bet_over_price, intel.bet_under_price,
        devig_method=devig_method,
        kelly_multiplier=float(tcfg.get("kelly", {}).get("fraction", 0.25)),
        kelly_cap=float(tcfg.get("kelly", {}).get("max_bankroll_fraction", 0.02)),
    )
    analysis.evals = evals
    pick_side = max(evals, key=lambda s: evals[s].ev_pct)
    analysis.pick_side = pick_side
    best = evals[pick_side]

    # Sanity: implausibly large EV is a data error, not real edge.
    max_ev = float(sanity.get("max_ev", 0.30))
    if best.ev_pct > max_ev and analysis.skipped_reason is None:
        analysis.skipped_reason = f"implausible EV ({best.ev_pct * 100:.0f}%) - likely bad totals data"

    # Sharp-fade on the picked side (+ve = we're more bullish on our side than
    # the sharp totals consensus).
    sharp_fade_pp = None
    if intel.sharp_available and intel.sharp_devig_over is not None:
        over_gap = blended_over - intel.sharp_devig_over
        sharp_fade_pp = over_gap if pick_side == "over" else -over_gap
        max_fade = float(sanity.get("max_sharp_disagreement_pp", 4.0))
        if sharp_fade_pp * 100.0 > max_fade and analysis.skipped_reason is None:
            analysis.skipped_reason = (
                f"fading sharp total by {sharp_fade_pp * 100:.1f}pp on {pick_side} "
                f"(our {blended_over if pick_side == 'over' else 1 - blended_over:.3f} vs sharps) > {max_fade:.1f}pp"
            )

    # Stability + confidence + tier sizing.
    stability = classify_totals_stability(
        profiles, weather, rd, profiles.home_lu, profiles.away_lu, sharp_fade_pp, config,
    )
    analysis.stability = stability
    analysis.confidence = data_conf

    threshold = float(tcfg.get("ev_threshold", 0.03))
    is_raw_pick = best.ev_pct >= threshold
    tier, tier_reasons = classify_totals_tier(best.ev_pct, data_conf, stability.label, config, is_raw_pick=is_raw_pick)
    kelly_cap = kelly_cap_for_tier(tier, config)
    if best.kelly_stake > kelly_cap:
        analysis._orig_kelly = best.kelly_stake
        best.kelly_stake = round(kelly_cap, 6)
    analysis.tier = tier
    analysis.tier_kelly_cap = kelly_cap
    analysis.tier_reasons = tier_reasons

    # Weather rule: never bet a total blind to weather. A missing feed zeroes the
    # weather component (already done in run_distribution) AND holds the pick to
    # ANALYSIS-ONLY (flagged, classified fragile) -- it is shown, never bet.
    weather_ok = weather is not None and weather.available
    if not weather_ok and bool(tcfg.get("weather", {}).get("require_for_bet", True)):
        analysis.weather_held = True
        analysis.flags.append("weather unavailable -> analysis only (not bet)")

    # Roof rule (RSI phase 2, config totals.weather.require_verified_roof):
    # a retractable roof is ASSUMED open by data/weather.py (free data can't
    # see a closed roof). When the operator wants only verified roof states
    # bet, an assumed state holds the pick to analysis-only, like weather_held.
    if bool(tcfg.get("weather", {}).get("require_verified_roof", False)):
        roof = getattr(weather, "roof", None) if weather is not None else None
        if roof is not None and "assumed" in str(roof):
            analysis.roof_unverified = True
            analysis.flags.append(f"roof state assumed ({roof}) -> analysis only (not bet)")

    # Post-market sanity skip: fully evaluated for the pass record, never bet.
    if analysis.skipped_reason:
        analysis.tier = "pass"
        analysis.tier_reasons = list(analysis.tier_reasons) + [f"skipped: {analysis.skipped_reason}"]
        best.kelly_stake = 0.0

    return analysis


def save_totals_slate(analyses, threshold: float, game_date: str,
                      model_tag: str = "totals_v1") -> tuple[int, int]:
    """Persist the evaluable totals slate (upsert). Returns (total, n_value).

    Every analysis with a pick + run distribution is persisted -- including
    the post-market sanity skips (divergence / max EV / sharp fade), which
    land as is_value=0 passes with pass_reason skip:<kind>. Ones that clear
    `threshold` with positive Kelly (and aren't weather-/roof-held or
    skipped) are flagged is_value=True (simulated bets). Pre-market skips
    (no market / implausible total / no distribution) are dropped. Every row
    is stamped with `model_tag`.
    """
    from mlb_value_bot.tracking.totals_recommendations import (
        TotalsRecommendationRecord,
        upsert_totals_recommendation,
    )

    total = 0
    n_value = 0
    for a in analyses:
        be = a.best_eval
        if be is None or a.rd is None or a.intel is None:
            continue
        a.model_tag = model_tag
        is_value = a.is_value(threshold)
        rec = TotalsRecommendationRecord(
            date=game_date,
            game_id=a.game_id,
            home_team=a.home_team,
            away_team=a.away_team,
            pick_side=a.pick_side or "",
            market_total=a.market_total,
            over_odds=a.intel.bet_over_price,
            under_odds=a.intel.bet_under_price,
            bet_odds=be.american_odds,
            decimal_odds=be.decimal_odds,
            model_p_over=a.model_p_over,
            market_devig_over=a.market_devig_over,
            blended_p_over=a.blended_over,
            model_prob=be.model_prob,
            market_prob_devigged=be.market_prob_devigged,
            ev_pct=be.ev_pct,
            kelly_stake=be.kelly_stake,
            confidence=a.confidence,
            tier=a.tier,
            stability=a.stability.label if a.stability else None,
            raw_model_total=a.rd.raw_model_total,
            expected_total=a.rd.expected_total,
            paper=a.paper,
            opening_devig_p_side=a.opening_devig_for(a.pick_side or ""),
            sharp_close=a.sharp_close,
            best_close_line=a.intel.bet_line,
            best_close_price=(a.intel.best_over_price if a.pick_side == "over" else a.intel.best_under_price),
            reasoning=a.reasoning(),
            is_value=is_value,
            model_tag=model_tag,
            pass_reason=a.pass_reason(threshold),
        )
        upsert_totals_recommendation(rec)
        total += 1
        if is_value:
            n_value += 1
    return total, n_value


def refresh_skipped_totals_closing(analyses, game_date: str) -> int:
    """Refresh the sharp close + CLV on committed paper bets whose totals game was
    skipped this run WITHOUT a persistable evaluation (no market / implausible
    total / no distribution) so the close doesn't freeze. Post-market sanity
    skips are persisted by save_totals_slate since RSI phase 2, so their close
    arrives through the upsert. Mirrors the moneyline version."""
    from mlb_value_bot.tracking.totals_recommendations import refresh_totals_close

    n = 0
    for a in analyses:
        if a.best_eval is not None and a.rd is not None and a.intel is not None:
            continue  # persisted by save_totals_slate; upsert handled the close
        if a.sharp_close is None and a.intel is None:
            continue  # nothing to refresh from
        if refresh_totals_close(game_date, a.game_id, a):
            n += 1
    return n
