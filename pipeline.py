"""Orchestration: turn a date's slate into ranked, evaluated game analyses.

This is the glue layer the `today` command (and the backtester) use so the
"pull odds -> match schedule -> build metrics -> run model -> compute EV" flow
lives in one reusable place rather than being copy-pasted into the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_cls, datetime, timezone

from mlb_value_bot.analysis.ev_calculator import (
    SideEvaluation,
    devigged_market_probs,
    evaluate_sides,
)
from mlb_value_bot.analysis.pitcher_metrics import build_pitcher_profile
from mlb_value_bot.analysis.team_metrics import TeamMetricsProvider
from mlb_value_bot.analysis.win_probability import (
    WinProbabilityResult,
    compute_confidence,
    compute_data_confidence,
    compute_win_probability,
    resolve_market_blend,
)
from mlb_value_bot.data.mlb_client import MLBClient, ScheduledGame
from mlb_value_bot.data.odds_client import GameOdds, OddsClient
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("pipeline")


@dataclass
class GameAnalysis:
    game_id: int
    game_date: str
    home_team: str
    away_team: str
    status: str
    home_pitcher: str | None
    away_pitcher: str | None
    game_datetime: str | None = None         # ISO UTC first-pitch time
    wp: WinProbabilityResult | None = None
    evals: dict[str, SideEvaluation] | None = None
    best_side: str | None = None
    confidence: float = 0.0
    skipped_reason: str | None = None
    # Market-anchoring: the EV uses `blended_home_prob`, not the raw model prob.
    market_home_prob: float | None = None    # de-vigged "fair" market prob (home)
    blend: float = 1.0                        # market_blend weight applied
    blend_tier: str = "fixed"                 # "high"/"mid"/"low"/"fixed"
    data_confidence: float = 0.0              # 0..100, drives blend tier
    blended_home_prob: float | None = None    # blend*model + (1-blend)*market
    # Bet sizing tier (independent of blend): "pass" | "small" | "standard" | "strong"
    tier: str = "pass"
    tier_kelly_cap: float = 0.0               # per-tier Kelly cap applied (fraction of bankroll)
    tier_reasons: list[str] = field(default_factory=list)
    # Lineup snapshots used to populate reasoning["lineup"] (UI chips + breakdown).
    home_lineup_status: "object | None" = None
    away_lineup_status: "object | None" = None
    # Sharp/square market intelligence (added 2026-05-28). Used to populate
    # reasoning["market_intel"] (UI agree/fade chip + breakdown).
    market_intel: "object | None" = None
    # Projected score (display-only middle-ground from #6, added 2026-05-28).
    # The win-prob model is UNCHANGED -- this is purely an additional output
    # so users can see the run-environment tilt in concrete terms.
    projected_score: "object | None" = None
    # Edge stability classification (Step 3, 2026-05-30). One of
    # "stable" / "moderate" / "fragile". Gates the "Strong" sizing tier
    # via the hard rule below. Also feeds the UI badge + Edge Drivers list.
    stability: "object | None" = None
    # Adjusted EV (Step 4, 2026-05-30). Raw EV (best_eval.ev_pct) is the pure
    # model-vs-price edge, shown unchanged. adjusted_ev_pct applies small,
    # signed context haircuts/boosts (sharp support/fade, projected lineup,
    # fragile edge) and is what the sizing tiers read. None until computed.
    adjusted_ev_pct: float | None = None
    adjusted_ev_reasons: list[str] = field(default_factory=list)

    @property
    def best_eval(self) -> SideEvaluation | None:
        if not self.evals or not self.best_side:
            return None
        return self.evals[self.best_side]

    def is_value(self, threshold: float) -> bool:
        be = self.best_eval
        return be is not None and be.ev_pct >= threshold and be.kelly_stake > 0

    def reasoning(self) -> dict:
        """Full JSON-able breakdown (model components + market-blend + sizing) for the DB."""
        data = self.wp.reasoning() if self.wp else {}
        data["market_anchor"] = {
            "raw_model_home_prob": round(self.wp.home_win_prob, 4) if self.wp else None,
            "market_devig_home_prob": round(self.market_home_prob, 4) if self.market_home_prob is not None else None,
            "blend_weight": round(self.blend, 3),
            "blend_tier": self.blend_tier,
            "data_confidence": self.data_confidence,
            "blended_home_prob": round(self.blended_home_prob, 4) if self.blended_home_prob is not None else None,
        }
        data["bet_sizing"] = {
            "tier": self.tier,
            "kelly_cap": self.tier_kelly_cap,
            "reasons": self.tier_reasons,
            # Raw quarter-Kelly before the per-tier cap was applied (only set
            # when the cap actually bound the stake).
            "raw_kelly": getattr(self, "_tier_original_kelly", None),
        }
        # Adjusted EV (Step 4): raw vs adjusted side by side + the per-line
        # adjustments, so the frontend can show both and we can backtest how
        # much the context haircuts moved sizing. raw_ev mirrors best_eval.ev_pct.
        be = self.best_eval
        data["adjusted_ev"] = {
            "raw_ev_pct": round(be.ev_pct, 4) if be is not None else None,
            "adjusted_ev_pct": round(self.adjusted_ev_pct, 4) if self.adjusted_ev_pct is not None else None,
            "adjustments": list(self.adjusted_ev_reasons),
        }
        data["pitchers"] = {"home": self.home_pitcher, "away": self.away_pitcher}
        data["game_datetime"] = self.game_datetime
        # Lineup status block: lets the UI render a "Projected lineup" chip,
        # show missing key bats per side, and segment performance by status.
        # Omitted when neither side has a status (legacy reasoning shape).
        if self.home_lineup_status is not None or self.away_lineup_status is not None:
            data["lineup"] = {
                "home": _lineup_to_dict(self.home_lineup_status),
                "away": _lineup_to_dict(self.away_lineup_status),
            }
        # Edge stability (Step 3): pick-level label + per-driver shares.
        # Used by the UI badge / chip and by Step 5's tier downgrade rule.
        if self.stability is not None:
            s = self.stability
            data["stability"] = {
                "label": s.label,
                "stable_share": s.stable_share,
                "fragile_share": s.fragile_share,
                "hard_fragile_signals": list(s.hard_fragile_signals),
                "drivers": list(s.drivers),
            }
        # Projected score (run-environment display): home_runs / away_runs /
        # total / pitcher basis. Pure additional output; the win-prob model
        # is unchanged. Omitted when any input was missing.
        if self.projected_score is not None and self.projected_score.available:
            ps = self.projected_score
            data["projected_score"] = {
                "home_runs": ps.home_runs,
                "away_runs": ps.away_runs,
                "total": ps.total,
                "pitcher_basis": ps.pitcher_basis,
            }
        # Sharp/square market intel: sharp consensus, square consensus, the
        # gap between them, and dispersion. UI uses this for the "Sharps
        # agree/fade" chip and the breakdown line. Omitted when the API
        # returned no usable sharp books (legacy + degraded configs).
        if self.market_intel is not None:
            mi = self.market_intel
            data["market_intel"] = {
                "sharp_devig_home": mi.sharp_devig_home,
                "square_devig_home": mi.square_devig_home,
                "sharp_minus_square_pp": (
                    round(mi.sharp_minus_square * 100, 2)
                    if mi.sharp_minus_square is not None else None
                ),
                "dispersion_pp": round(mi.dispersion_pp, 2) if mi.dispersion_pp is not None else None,
                "n_sharp_books": mi.n_sharp_books,
                "n_square_books": mi.n_square_books,
                "n_total_books": mi.n_total_books,
            }
        return data


def _lineup_to_dict(status) -> dict | None:
    """Project a LineupStatus into the reasoning_json shape (or None if absent)."""
    if status is None:
        return None
    return {
        "status": status.status,
        "key_bats_total": status.key_bats_total,
        "key_bats_present": status.key_bats_present,
        "missing_key_bats": list(status.missing_key_bats),
        "notes": list(status.notes),
    }


def _odds_by_team(game: GameOdds) -> dict[str, int]:
    out: dict[str, int] = {}
    if game.home:
        out[game.home.team] = game.home.american_odds
    if game.away:
        out[game.away.team] = game.away.american_odds
    return out


def _match_odds(scheduled: ScheduledGame, odds: list[GameOdds]) -> GameOdds | None:
    """Match a scheduled game to its odds by team pair AND date.

    The Odds API returns every upcoming event, so a multi-day series has several
    events sharing the same team pair. Matching on the pair alone (the old bug)
    grabbed whichever appeared first — often the WRONG day's line. We pick the
    event whose start is on the game date, and if the only candidates fall on
    other days we return None (skip) rather than publish a misleading line.
    """
    target = frozenset({scheduled.home_team, scheduled.away_team})
    candidates = [g for g in odds if frozenset({g.home_team, g.away_team}) == target]
    if not candidates:
        return None

    # Reference ~mid-slate on the game date (21:00 UTC ≈ 5pm ET). Same-day MLB
    # first pitches land within ~8h of this; the next day's are ~17h+ away, so a
    # 12h tolerance cleanly separates "today's game" from the rest of the series.
    try:
        ref = datetime.fromisoformat(scheduled.game_date).replace(hour=21, tzinfo=timezone.utc)
    except ValueError:
        return candidates[0]

    def _hours_off(g: GameOdds) -> float:
        try:
            dt = datetime.fromisoformat((g.commence_time or "").replace("Z", "+00:00"))
        except ValueError:
            return float("inf")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return abs((dt - ref).total_seconds()) / 3600.0

    best = min(candidates, key=_hours_off)
    return best if _hours_off(best) <= 12.0 else None


def evaluate_game(
    scheduled: ScheduledGame,
    game_odds: GameOdds | None,
    team_provider: TeamMetricsProvider,
    season: int,
    as_of: date_cls,
    config: dict | None = None,
    bullpen_status_provider=None,  # callable: (team_name, team_id) -> BullpenStatus | None
    lineup_status_provider=None,   # callable: (team_name, game_pk, side, first_pitch_iso) -> LineupStatus | None
) -> GameAnalysis:
    """Run the full model + EV evaluation for a single game.

    `bullpen_status_provider` and `lineup_status_provider` are optional
    callables that return per-team status snapshots. Both follow the same
    provider pattern: the slate-wide pulls (per-pitcher reliever stats,
    per-player hitting stats) happen ONCE in analyze_slate; per-team /
    per-game status is computed on demand and memoized. Pass None for
    either to fall back to the pre-feature behavior (component contributes
    0 with "data unavailable").
    """
    config = config or load_config()
    analysis = GameAnalysis(
        game_id=scheduled.game_id,
        game_date=scheduled.game_date,
        home_team=scheduled.home_team,
        away_team=scheduled.away_team,
        status=scheduled.status,
        home_pitcher=scheduled.home_pitcher.name,
        away_pitcher=scheduled.away_pitcher.name,
        game_datetime=scheduled.game_datetime,
    )

    if not scheduled.is_playable:
        analysis.skipped_reason = f"not playable ({scheduled.status})"
        return analysis
    if game_odds is None:
        analysis.skipped_reason = "no odds found"
        return analysis

    team_prices = _odds_by_team(game_odds)
    home_odds = team_prices.get(scheduled.home_team)
    away_odds = team_prices.get(scheduled.away_team)
    if home_odds is None or away_odds is None:
        analysis.skipped_reason = "incomplete moneyline (missing a side)"
        return analysis

    # Sanity guard (odds): a placeholder/stale feed can emit absurd lines (e.g.
    # +3300) that let the model "find" enormous fake edge. Real MLB moneylines
    # never approach this, so skip the game with a clear reason instead of risking
    # a garbage pick reaching the DB or the public site. Tunable in config.sanity.
    max_abs_odds = float(config.get("sanity", {}).get("max_abs_odds", 800))
    if abs(home_odds) > max_abs_odds or abs(away_odds) > max_abs_odds:
        analysis.skipped_reason = (
            f"implausible odds ({home_odds:+d}/{away_odds:+d}) - likely bad market data"
        )
        return analysis

    # Build metric profiles. Missing probable pitchers degrade gracefully:
    # the starter/form components fall to 0 and confidence drops.
    home_pp = build_pitcher_profile(scheduled.home_pitcher.player_id, scheduled.home_pitcher.name, season, as_of)
    away_pp = build_pitcher_profile(scheduled.away_pitcher.player_id, scheduled.away_pitcher.name, season, as_of)
    home_tp = team_provider.build_team_profile(scheduled.home_team, is_home=True)
    away_tp = team_provider.build_team_profile(scheduled.away_team, is_home=False)

    # Bullpen fatigue: optional, never fatal. Missing -> component contributes 0.
    home_bp = None
    away_bp = None
    if bullpen_status_provider is not None:
        try:
            home_bp = bullpen_status_provider(scheduled.home_team, scheduled.home_team_id)
            away_bp = bullpen_status_provider(scheduled.away_team, scheduled.away_team_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("bullpen status provider failed for game %s (%s)", scheduled.game_id, exc)

    # Lineup status: optional, same shape. Drives the `lineup` component AND
    # a confidence penalty when lineups are projected (not yet confirmed).
    home_lu = None
    away_lu = None
    if lineup_status_provider is not None:
        try:
            home_lu = lineup_status_provider(
                scheduled.home_team, scheduled.game_id, "home", scheduled.game_datetime,
            )
            away_lu = lineup_status_provider(
                scheduled.away_team, scheduled.game_id, "away", scheduled.game_datetime,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("lineup status provider failed for game %s (%s)", scheduled.game_id, exc)

    wp = compute_win_probability(
        home_tp, away_tp, home_pp, away_pp, config,
        home_bullpen_status=home_bp, away_bullpen_status=away_bp,
        home_lineup_status=home_lu, away_lineup_status=away_lu,
    )

    # Projected score (display-only -- the moneyline EV below is computed off
    # `wp.home_win_prob` exactly as before, this is purely additional output).
    from mlb_value_bot.analysis.run_environment import projected_score as _proj_score
    proj_score = _proj_score(home_tp, away_tp, home_pp, away_pp, config)

    # Market anchoring: blend the raw model toward the de-vigged market so EV
    # reflects a *bounded tilt* off the sharp consensus, not raw model
    # overconfidence (a standalone heuristic otherwise "finds" edges everywhere).
    devig_method = config["ev"].get("devig_method", "power")
    market_home, _market_away = devigged_market_probs(home_odds, away_odds, devig_method)

    # Sharp/square market intelligence (#5). Built from the per-book pricing
    # already returned by The Odds API in `game_odds.all_books`. No extra
    # quota cost. Used below as: (a) sharp-fade sanity guard, (b) confidence
    # penalty, (c) tier downgrade. Degrades to None on missing config / no
    # sharp books returned today.
    from mlb_value_bot.data.market_intel import compute_market_intel
    market_intel = compute_market_intel(
        game_odds,
        sharp_books=config.get("odds_api", {}).get("sharp_bookmakers") or [],
        square_books=config.get("odds_api", {}).get("square_bookmakers") or [],
        devig_method=devig_method,
    )

    # Dynamic blend: the model earns more weight when the underlying DATA is
    # trustworthy (good pitcher samples, team data complete, components in
    # agreement). EV is deliberately NOT part of this -- letting EV drive the
    # blend would create a feedback loop where the model talks itself into
    # bigger edges. Falls back to a fixed blend if config.model.market_blend is
    # still a scalar.
    #
    # Missing-data penalties shave confidence (and therefore data_confidence,
    # which feeds the blend tier table -> more market anchoring on a thin
    # data game). We DO NOT fabricate probability signal when data is
    # missing -- we just lower the score the UI shows.
    #   lineup_penalty: applied when either team's lineup is still projected.
    #   bullpen_penalty: applied when the bullpen-availability feed is
    #                    unavailable for either team (genuine API gap, not
    #                    "3/3 available which the schedule maps to 0%").
    lineup_penalty = _lineup_confidence_penalty(home_lu, away_lu, config)
    bullpen_penalty = _bullpen_confidence_penalty(home_bp, away_bp, config)
    data_conf = compute_data_confidence(
        wp, home_pp, away_pp, home_tp, away_tp, config,
        lineup_penalty=lineup_penalty, bullpen_penalty=bullpen_penalty,
    )
    blend, blend_tier = resolve_market_blend(data_conf, config["model"])
    blended_home = blend * wp.home_win_prob + (1.0 - blend) * market_home

    # Sanity guard (model/market divergence): if the RAW model and the
    # de-vigged market disagree by more than `max_model_market_divergence`,
    # the market is almost certainly reacting to information the model
    # doesn't have -- late starter scratch, lineup change, weather, etc.
    # PRODUCTION INCIDENT 2026-05-28: Sox/Braves flipped from -149 to +295
    # in minutes (Sox went to a bullpen game). Our probable-pitcher data was
    # still on the announced starter, so the model said ~49% home while
    # market said ~27%. Mid-tier blend pulled only partway -> fake +12.6% EV.
    # We check BEFORE evaluating EV because the blended prob masks the
    # underlying disagreement. Tunable in config.sanity.
    max_div = float(config.get("sanity", {}).get("max_model_market_divergence", 0.15))
    divergence = abs(wp.home_win_prob - market_home)
    if divergence > max_div:
        analysis.skipped_reason = (
            f"raw model ({wp.home_win_prob:.3f}) vs market ({market_home:.3f}) "
            f"diverge by {divergence:.3f} > {max_div:.2f} - market likely on news the model doesn't see"
        )
        return analysis

    evals = evaluate_sides(
        blended_home,
        home_odds,
        away_odds,
        devig_method=devig_method,
        kelly_multiplier=config["kelly"]["fraction"],
        kelly_cap=config["kelly"]["max_bankroll_fraction"],
    )

    best_side = max(evals, key=lambda s: evals[s].ev_pct)

    # Sanity guard (EV): an implausibly large EV is a data error, not real edge.
    # Catches anything that slips past the odds-band and divergence checks.
    # Tunable in config.sanity.
    max_ev = float(config.get("sanity", {}).get("max_ev", 0.30))
    if evals[best_side].ev_pct > max_ev:
        analysis.skipped_reason = (
            f"implausible EV ({evals[best_side].ev_pct * 100:.0f}%) - likely bad market data"
        )
        return analysis

    confidence = compute_confidence(
        wp, home_pp, away_pp, home_tp, away_tp, evals[best_side].ev_pct, config,
        lineup_penalty=lineup_penalty, bullpen_penalty=bullpen_penalty,
    )

    # Market-intel disagreement: compute how much we're "fading the sharps"
    # on the side we're recommending. Used by both the sanity guard
    # (skip on extreme fade) and the tier downgrade (Strong -> Standard when
    # mildly fading sharps).
    our_pick_home_prob = blended_home if best_side == "home" else 1.0 - blended_home
    sharp_fade_pp: float | None = None
    if market_intel.available:
        gap_home = market_intel.disagreement_with(blended_home)  # +ve = we more bullish on home than sharps
        if gap_home is not None:
            sharp_fade_pp = gap_home if best_side == "home" else -gap_home

    # Sanity guard: if we'd be fading sharp consensus by more than this many
    # probability points on the side we're recommending, skip the game. The
    # sharps are the smartest counterparty on the board; betting hard against
    # them when they disagree by 5pp+ is essentially betting we know more.
    # PRODUCTION INCIDENT 2026-05-28: would have been a second line of
    # defense against the Sox/Braves live-line fake +EV (sharps would have
    # priced the in-play moneyline similarly, but the divergence guard
    # already catches that one. This guard catches DIFFERENT failure modes:
    # market reading a news headline, weather, lineup change we missed.)
    max_sharp_fade = float(config.get("sanity", {}).get("max_sharp_disagreement_pp", 5.0))
    if sharp_fade_pp is not None and sharp_fade_pp * 100 > max_sharp_fade:
        analysis.skipped_reason = (
            f"fading sharp consensus by {sharp_fade_pp * 100:.1f}pp on {best_side} "
            f"(blended {our_pick_home_prob:.3f} vs sharps {market_intel.sharp_devig_home:.3f}) "
            f"> {max_sharp_fade:.1f}pp"
        )
        analysis.market_intel = market_intel
        return analysis

    # Edge stability (Step 3, 2026-05-30). Classify the pick as
    # STABLE / MODERATE / FRAGILE based on WHICH components are driving the
    # edge and whether sharp fade / projected lineups / missing data are in
    # play. Computed BEFORE tier classification so the hard-rule downgrade
    # below can demote Strong -> Standard on a fragile edge.
    from mlb_value_bot.analysis.stability import classify_edge_stability
    stability = classify_edge_stability(
        components=wp.components,
        best_side=best_side,
        sharp_fade_pp=sharp_fade_pp,
        home_lineup_status=home_lu,
        away_lineup_status=away_lu,
        config=config,
    )

    # Adjusted EV (Step 4): Raw EV (evals[best_side].ev_pct) is displayed
    # unchanged; Adjusted EV applies small, signed context haircuts/boosts and
    # is what the sizing tiers below read. This is the SINGLE home for the
    # sharp-fade penalty -- the old confidence-score penalty and the old
    # Strong->Standard sharp-fade tier downgrade were both REMOVED here so the
    # sharp signal is counted exactly once (no double/triple counting).
    raw_ev = evals[best_side].ev_pct
    adjusted_ev, adjusted_ev_reasons = _compute_adjusted_ev(
        raw_ev, sharp_fade_pp, stability.label == "fragile", config
    )

    # Bet sizing tiers (Step 5): classify the pick into Pass/Small/Standard/
    # Strong on ADJUSTED EV, apply the unified one-tier downgrade guardrail
    # (low confidence / the Step 3 "never Strong on fragile" hard rule), then
    # cap the Kelly stake per tier. Reads ADJUSTED EV (not raw) so the Step 4
    # context haircuts flow through to the stake.
    #
    # NOTE (2026-06-04): projected/unconfirmed lineups no longer add an EV
    # haircut OR a tier downgrade. Both were a flat tax on EVERY pre-lineup
    # (morning) run, which never discriminated -- it just shifted the whole
    # slate down. The uncertainty is still captured by the graduated lineup
    # CONFIDENCE penalty (config.lineup.confidence_penalty), which feeds the
    # market blend and can still trip the `confidence < downgrade_confidence`
    # tier downgrade below when it actually drags confidence under the bar.
    # Selection is decided by RAW EV (config.ev.threshold); Adjusted EV / fragility
    # only SIZE the bet. A game that already cleared the raw bar is floored at the
    # `small` tier inside _classify_bet_tier so the context haircuts can't veto it
    # back to `pass` (the 2026-06-05 decouple-selection-from-sizing change).
    ev_threshold = float(config.get("ev", {}).get("threshold", 0.03))
    is_raw_pick = raw_ev >= ev_threshold
    tier, tier_reasons = _classify_bet_tier(
        adjusted_ev, confidence, stability.label, config, is_raw_pick=is_raw_pick
    )
    kelly_cap = _kelly_cap_for_tier(tier, config)
    original_kelly = evals[best_side].kelly_stake
    capped_kelly = round(min(original_kelly, kelly_cap), 6)
    if capped_kelly != original_kelly:
        # Preserve the raw quarter-Kelly for the audit trail in reasoning_json.
        analysis._tier_original_kelly = original_kelly  # type: ignore[attr-defined]
    evals[best_side].kelly_stake = capped_kelly

    analysis.adjusted_ev_pct = adjusted_ev
    analysis.adjusted_ev_reasons = adjusted_ev_reasons
    analysis.tier_kelly_cap = kelly_cap
    analysis.wp = wp
    analysis.evals = evals
    analysis.best_side = best_side
    analysis.confidence = confidence
    analysis.market_home_prob = market_home
    analysis.blend = blend
    analysis.blend_tier = blend_tier
    analysis.data_confidence = data_conf
    analysis.blended_home_prob = blended_home
    analysis.tier = tier
    analysis.tier_reasons = tier_reasons
    analysis.home_lineup_status = home_lu
    analysis.away_lineup_status = away_lu
    analysis.market_intel = market_intel
    analysis.projected_score = proj_score
    analysis.stability = stability
    return analysis


# --- Adjusted EV -------------------------------------------------------------
def _compute_adjusted_ev(
    raw_ev: float,
    sharp_fade_pp: float | None,
    fragile: bool,
    config: dict,
) -> tuple[float, list[str]]:
    """Turn Raw EV into the sizing-grade Adjusted EV (Step 4).

    Raw EV is the pure model-vs-price edge (kept and displayed unchanged).
    Adjusted EV applies a few SMALL, signed adjustments for context the raw
    number can't see, and is the value the sizing tiers read. Returns
    (adjusted_ev, reasons). Magnitudes are config.adjusted_ev.*.

    This is the SINGLE home for the sharp-fade penalty. Sign convention:
    `sharp_fade_pp` (a fraction, e.g. 0.04 = 4pp) is POSITIVE when WE are
    more bullish on the pick side than the sharp consensus (we are FADING
    the sharps) and NEGATIVE when the sharps are even more bullish on our
    side than we are (sharp SUPPORT). Sharp support and fade are mutually
    exclusive; the fade reduction is tiered (mild vs large), not stacked.
    """
    cfg = config.get("adjusted_ev", {})
    adj = raw_ev
    reasons: list[str] = []

    support_pp = float(cfg.get("sharp_support_pp", 3.0))
    fade_pp = float(cfg.get("sharp_fade_pp", 3.0))
    fade_large_pp = float(cfg.get("sharp_fade_large_pp", 5.0))
    if sharp_fade_pp is not None:
        fade = sharp_fade_pp * 100.0  # to pp
        if fade <= -support_pp:
            boost = float(cfg.get("sharp_support_boost", 0.010))
            adj += boost
            reasons.append(f"+{boost * 100:.1f}pp sharps support pick ({-fade:.1f}pp)")
        elif fade >= fade_large_pp:
            cut = float(cfg.get("sharp_fade_large_reduction", 0.020))
            adj -= cut
            reasons.append(f"-{cut * 100:.1f}pp large sharp fade ({fade:.1f}pp)")
        elif fade >= fade_pp:
            cut = float(cfg.get("sharp_fade_reduction", 0.010))
            adj -= cut
            reasons.append(f"-{cut * 100:.1f}pp sharp fade ({fade:.1f}pp)")

    # NOTE (2026-06-04): the projected/unconfirmed-lineup haircut was removed.
    # It fired on every pre-lineup (morning) run, so it was a flat tax rather
    # than a discriminating signal -- equivalent to raising the EV threshold by
    # 1pp on the whole slate. Projected-lineup uncertainty is instead carried by
    # the graduated lineup CONFIDENCE penalty (config.lineup.confidence_penalty).

    if fragile:
        cut = float(cfg.get("fragile_reduction", 0.015))
        adj -= cut
        reasons.append(f"-{cut * 100:.1f}pp fragile edge")

    return round(adj, 6), reasons


# --- Bet sizing tiers --------------------------------------------------------
# Tier order, lowest -> highest, for single-step downgrades.
_TIER_ORDER = ("pass", "small", "standard", "strong")


def _classify_bet_tier(
    adjusted_ev: float,
    confidence: float,
    stability_label: str,
    config: dict,
    is_raw_pick: bool = False,
) -> tuple[str, list[str]]:
    """Classify a pick into Pass/Small/Standard/Strong on ADJUSTED EV, then
    apply a single one-tier downgrade guardrail. Returns (tier, reasons).

    Tier bands (config.bet_sizing.*_ev, decimal fractions):
        adj EV < small_ev    (2%) -> pass        (not action)
        [small_ev, standard_ev)   -> small/lean
        [standard_ev, strong_ev)  -> standard
        adj EV >= strong_ev  (8%) -> strong

    SELECTION vs SIZING (2026-06-05): `is_raw_pick` is True when the game already
    cleared the raw-EV pick threshold (config.ev.threshold). Selection is decided
    by RAW EV alone; Adjusted EV / fragility only SIZE the bet. So a raw-qualifying
    pick is floored at `small` -- the context haircuts can keep it from sizing UP to
    standard/strong, but never veto it back to `pass`. Only a game that never cleared
    the raw bar can land in `pass`. (Before this, a fragile -1.5pp + sharp-fade -1.0pp
    stack routinely dragged a +3-4% raw edge under the 2% adj-EV floor -> pass -> no
    pick, which silenced the slate for 6 straight days. CLV on the now-published small
    picks is the honest test of whether fragile edges deserve to be bet at all.)

    NOTE on STRONG: 8%+ EV on an MLB moneyline is RARE and far more often a
    model/data error (stale line, scratched starter, bad de-vig) than a real
    edge. Treat a Strong pick as "FLAG FOR MANUAL REVIEW", not "auto-trust" --
    the sanity guards in evaluate_game skip the most egregious cases, but a
    clean-looking 8%+ still deserves a human glance before sizing up.

    Downgrade ONE tier (a single step, never below pass) if ANY of: confidence
    < downgrade_confidence, OR the edge is FRAGILE *and the band tier is Strong*
    (the "never Strong on a fragile edge" hard rule, Step 3).

    NOTE (2026-06-04): the `lineup_unconfirmed` downgrade trigger was removed.
    It fired on every pre-lineup (morning) run, so it knocked EVERY pick down a
    full tier rather than discriminating -- a heavier version of the (also
    removed) projected-lineup EV haircut. Projected-lineup uncertainty now lives
    solely in the graduated lineup CONFIDENCE penalty, which can still trip the
    `confidence < downgrade_confidence` step below when it actually matters.

    De-dup note (2026-06-03): the sharp-fade AND the fragile penalties both
    already live in Adjusted EV (Step 4: -fragile_reduction pp), so neither
    adds a *general* tier step here -- doing so would DOUBLE-COUNT the same
    signal (a pp haircut AND a full tier). The only fragile-driven step that
    survives is the never-Strong ceiling, which bites the Strong band alone; a
    fragile small/standard pick keeps its tier because its fragility is already
    priced into Adjusted EV. (Before this fix, fragility knocked ordinary
    small/standard picks down a tier on top of the EV haircut, flipping
    marginal picks straight to pass.)
    """
    sizing = config.get("bet_sizing", {})
    small_ev = float(sizing.get("small_ev", 0.02))
    standard_ev = float(sizing.get("standard_ev", 0.05))
    strong_ev = float(sizing.get("strong_ev", 0.08))
    min_conf = float(sizing.get("downgrade_confidence", 65.0))

    # Floor index for the downgrade step below: a raw-qualifying pick never
    # drops below `small` (selection is by raw EV; haircuts only size it).
    floor_idx = _TIER_ORDER.index("small") if is_raw_pick else _TIER_ORDER.index("pass")

    reasons: list[str] = []
    if adjusted_ev < small_ev:
        if not is_raw_pick:
            return "pass", [f"adj EV {adjusted_ev * 100:.1f}% below {small_ev * 100:.1f}% threshold"]
        tier = "small"
        reasons.append(
            f"adj EV {adjusted_ev * 100:.1f}% < {small_ev * 100:.0f}% but raw EV cleared the pick "
            f"threshold -> floored to small (selection is by raw EV; haircuts only size)"
        )
    elif adjusted_ev < standard_ev:
        tier = "small"
        reasons.append(f"adj EV {adjusted_ev * 100:.1f}% in lean range [{small_ev * 100:.0f}%, {standard_ev * 100:.0f}%)")
    elif adjusted_ev < strong_ev:
        tier = "standard"
        reasons.append(f"adj EV {adjusted_ev * 100:.1f}% in standard range [{standard_ev * 100:.0f}%, {strong_ev * 100:.0f}%)")
    else:
        tier = "strong"
        reasons.append(
            f"adj EV {adjusted_ev * 100:.1f}% >= {strong_ev * 100:.0f}% -- FLAG FOR MANUAL REVIEW "
            f"(8%+ MLB ML edge is rare; suspect model/data error before trusting)"
        )

    # One-tier downgrade guardrail: a single step regardless of how many fire.
    # FRAGILE participates ONLY at the Strong band (the never-Strong hard rule);
    # its magnitude penalty already lives in Adjusted EV, so it does NOT add a
    # general tier step at small/standard (that was the double-count we removed).
    triggers: list[str] = []
    if stability_label == "fragile" and tier == "strong":
        triggers.append("fragile edge (never Strong on fragile)")
    if confidence < min_conf:
        triggers.append(f"confidence {confidence:.0f} < {min_conf:.0f}")
    if triggers:
        idx = _TIER_ORDER.index(tier)
        new_tier = _TIER_ORDER[max(floor_idx, idx - 1)]
        if new_tier != tier:
            reasons.append(f"downgraded {tier} -> {new_tier}: {', '.join(triggers)}")
            tier = new_tier
        else:
            reasons.append(f"downgrade noted (already at floor): {', '.join(triggers)}")

    return tier, reasons


def _kelly_cap_for_tier(tier: str, config: dict) -> float:
    """Per-tier Kelly cap (fraction of bankroll). The final stake is
    min(raw quarter-Kelly, this cap); pass -> 0. Caps live in
    config.bet_sizing.kelly_caps and are tighter than kelly.max_bankroll_
    fraction for small/standard on purpose (the model is unproven)."""
    caps = config.get("bet_sizing", {}).get("kelly_caps", {})
    default = {"pass": 0.0, "small": 0.005, "standard": 0.010, "strong": 0.020}
    return float(caps.get(tier, default.get(tier, 0.0)))


def _lineup_confidence_penalty(home_lu, away_lu, config: dict) -> float:
    """Confidence points to subtract based on lineup confirmation state.

    DISTINGUISHES 'projected' (lineups simply not posted yet -- a timing gap
    that resolves as first pitch nears, since the engine re-runs through the
    day) from 'unavailable' (a genuine feed/API failure, or the feature
    disabled so lu is None). Both are real info loss, but an outright data
    outage is strictly worse than "too early", so it draws the largest
    penalty. This is the model-side half of "distinguish projected vs
    unavailable" -- the reasoning_json already carries the per-side status
    string for the UI chip.

    Scheme (config.lineup.confidence_penalty; non-linear on purpose):
        both confirmed        ->  0
        one side projected    -> -3   (other side confirmed)
        both sides projected  -> -5   (sub-additive: a quiet morning, not a gap)
        any side unavailable  -> -6   (hard data gap dominates, worst case)
    """
    cfg = config.get("lineup", {})
    pen = cfg.get("confidence_penalty", {})
    p_one = float(pen.get("one_projected", 3.0))
    p_both = float(pen.get("both_projected", 5.0))
    p_unavail = float(pen.get("data_unavailable", 6.0))

    # lu is None when the feature is disabled / provider returned nothing ->
    # treat as a hard data gap (unavailable), same as an explicit API failure.
    statuses = [getattr(lu, "status", "unavailable") if lu is not None else "unavailable"
                for lu in (home_lu, away_lu)]
    if any(s == "unavailable" for s in statuses):
        return p_unavail
    n_projected = sum(1 for s in statuses if s != "confirmed")
    if n_projected >= 2:
        return p_both
    if n_projected == 1:
        return p_one
    return 0.0


def _bullpen_confidence_penalty(home_bp, away_bp, config: dict) -> float:
    """Confidence points to subtract when the bullpen-availability feed is
    genuinely unavailable for a team.

    Distinct from the case where the feed says "3/3 available" (no penalty;
    that's a real read of "everyone's rested"). Only fires when the feed
    didn't return usable data -- per Step 2 design we lower confidence in
    that case rather than fabricate signal.

    Counts one penalty unit per team whose BullpenStatus is None or has
    available=False.
    """
    cfg = config.get("bullpen_fatigue", {})
    per_team = float(cfg.get("unavailable_confidence_penalty_per_team", 2.0))
    n = 0
    for bp in (home_bp, away_bp):
        if bp is None or not bp.available:
            n += 1
    return per_team * n


def _build_lineup_status_provider(
    mlb_client: MLBClient,
    season: int,
    config: dict,
):
    """Factory: per-slate cached LineupStatus provider.

    One slate-wide pull of per-player season hitting stats (for the key-bats
    list), then per-team lineup lookups are memoized -- and the underlying
    feed-live call is itself game-keyed so each game's lineup is fetched
    once for both sides. Returns None (disabled) if the pre-fetch is empty
    or the feature is config-disabled.
    """
    from datetime import datetime, timezone
    from mlb_value_bot.data.lineup_status import LineupStatus, get_lineup_status

    if not config.get("lineup", {}).get("enabled", True):
        return None

    try:
        per_player = mlb_client.get_per_player_hitting(season)
    except Exception as exc:  # noqa: BLE001
        log.warning("lineup: per-player hitting pull failed (%s); disabling", exc)
        return None
    if not per_player:
        log.info("lineup: no per-player hitting data; disabling")
        return None

    lineup_cache: dict[int, dict[str, list[int]]] = {}    # game_pk -> {home/away: [pids]}
    status_cache: dict[tuple, LineupStatus] = {}          # (team, game_pk, side) -> status
    now_utc = datetime.now(timezone.utc)

    def provider(team_name: str, game_pk: int, side: str, first_pitch_iso: str | None) -> LineupStatus | None:
        if not team_name or not game_pk or side not in ("home", "away"):
            return None
        key = (team_name, int(game_pk), side)
        if key in status_cache:
            return status_cache[key]
        status = get_lineup_status(
            team_name=team_name, game_pk=int(game_pk), side=side,
            first_pitch_iso=first_pitch_iso, mlb=mlb_client,
            per_player_hitting=per_player, config=config,
            now_utc=now_utc, _lineup_cache=lineup_cache,
        )
        status_cache[key] = status
        return status

    return provider


def _build_bullpen_status_provider(
    mlb_client: MLBClient,
    season: int,
    as_of: date_cls,
    config: dict,
):
    """Factory: per-slate cached BullpenStatus provider.

    The per-pitcher reliever stats are fetched ONCE per slate run (one API
    call). Per-team status is then computed on demand and memoized so the
    same team's bullpen isn't re-derived for its second matchup of a
    doubleheader. Returns None (disabled) if the pre-fetch comes back empty
    or if the feature is config-disabled -- in which case the model
    component just contributes 0 with "data unavailable".
    """
    from mlb_value_bot.data.bullpen_status import BullpenStatus, get_bullpen_status

    if not config.get("bullpen_fatigue", {}).get("enabled", True):
        return None

    try:
        per_pitcher = mlb_client.get_per_pitcher_reliever_stats(season)
    except Exception as exc:  # noqa: BLE001
        log.warning("bullpen fatigue: per-pitcher pull failed (%s); disabling", exc)
        return None
    if not per_pitcher:
        log.info("bullpen fatigue: no per-pitcher reliever data; disabling")
        return None

    cache: dict[tuple[str, int], BullpenStatus] = {}

    def provider(team_name: str, team_id: int) -> BullpenStatus | None:
        if not team_name:
            return None
        key = (team_name, team_id)
        if key in cache:
            return cache[key]
        status = get_bullpen_status(
            team_name=team_name, team_id=team_id, as_of=as_of,
            mlb=mlb_client, per_pitcher_relievers=per_pitcher, config=config,
        )
        cache[key] = status
        return status

    return provider


def analyze_slate(
    game_date: str,
    odds_client: OddsClient | None = None,
    mlb_client: MLBClient | None = None,
    config: dict | None = None,
) -> list[GameAnalysis]:
    """Analyze every game on `game_date`, sorted by best-side EV descending."""
    config = config or load_config()
    odds_client = odds_client or OddsClient(config=config)
    mlb_client = mlb_client or MLBClient(config=config)

    schedule = mlb_client.get_schedule(game_date)
    odds = odds_client.get_odds()
    season = int(game_date[:4])
    as_of = date_cls.fromisoformat(game_date)
    provider = TeamMetricsProvider(season=season, config=config, mlb_client=mlb_client)

    # Bullpen fatigue (optional): one slate-wide pull of per-pitcher reliever
    # stats; per-team boxscore lookups are memoized within this run so two
    # teams sharing the same recent matchup don't re-fetch the same boxscore.
    # Disable via config.bullpen_fatigue.enabled=false (e.g. for fast backtests).
    bp_provider = _build_bullpen_status_provider(mlb_client, season, as_of, config)
    # Lineup status: one slate-wide pull of per-player hitting; per-game
    # feed-live calls memoized so both sides share one fetch.
    lu_provider = _build_lineup_status_provider(mlb_client, season, config)

    analyses: list[GameAnalysis] = []
    for sched in schedule:
        matched = _match_odds(sched, odds)
        analyses.append(evaluate_game(
            sched, matched, provider, season, as_of, config,
            bullpen_status_provider=bp_provider,
            lineup_status_provider=lu_provider,
        ))

    # Rank: evaluable games by EV desc, skipped games last.
    def _sort_key(a: GameAnalysis) -> float:
        be = a.best_eval
        return be.ev_pct if be is not None else -999.0

    analyses.sort(key=_sort_key, reverse=True)
    return analyses


def save_slate(analyses: list[GameAnalysis], threshold: float, game_date: str) -> tuple[int, int]:
    """Persist the FULL evaluable slate (upsert). Returns (total, n_value).

    Every analysis with a `best_eval` and a `wp` (i.e. evaluable -- has odds +
    a runnable model) is persisted. Picks that clear `threshold` AND have a
    positive Kelly stake are marked is_value=True (real bets); the rest are
    persisted as is_value=False analyses so the site can render the whole
    slate even on quiet days. Skipped games (no odds / postponed / no model)
    are dropped, since they have nothing meaningful to display.

    Upsert is keyed on (date, game_id): one row per game per date. Re-running
    refreshes prices/CLV on bets and overwrites analyses with the latest.
    """
    from mlb_value_bot.tracking.recommendations import (
        RecommendationRecord,
        upsert_recommendation,
    )

    total = 0
    n_value = 0
    for a in analyses:
        be = a.best_eval
        if be is None or a.wp is None:
            continue
        is_value = be.ev_pct >= threshold and be.kelly_stake > 0
        rec = RecommendationRecord(
            date=game_date,
            game_id=a.game_id,
            home_team=a.home_team,
            away_team=a.away_team,
            recommended_side=a.best_side or "",
            model_prob=be.model_prob,
            market_prob_devigged=be.market_prob_devigged,
            american_odds=be.american_odds,
            decimal_odds=be.decimal_odds,
            ev_pct=be.ev_pct,
            kelly_stake=be.kelly_stake,
            confidence=a.confidence,
            reasoning=a.reasoning(),
            is_value=is_value,
        )
        upsert_recommendation(rec)
        total += 1
        if is_value:
            n_value += 1
    return total, n_value


def save_value_bets(value_bets: list[GameAnalysis], game_date: str) -> int:
    """Backward-compat wrapper: persist a pre-filtered list of +EV bets.

    Prefer `save_slate(analyses, threshold, date)` directly so the full slate
    (including passes) lands in the DB and the public site can show it. This
    wrapper marks every input as is_value=True, matching the old behavior.
    """
    from mlb_value_bot.tracking.recommendations import (
        RecommendationRecord,
        upsert_recommendation,
    )

    count = 0
    for a in value_bets:
        be = a.best_eval
        if be is None or a.wp is None:
            continue
        rec = RecommendationRecord(
            date=game_date,
            game_id=a.game_id,
            home_team=a.home_team,
            away_team=a.away_team,
            recommended_side=a.best_side or "",
            model_prob=be.model_prob,
            market_prob_devigged=be.market_prob_devigged,
            american_odds=be.american_odds,
            decimal_odds=be.decimal_odds,
            ev_pct=be.ev_pct,
            kelly_stake=be.kelly_stake,
            confidence=a.confidence,
            reasoning=a.reasoning(),
            is_value=True,
        )
        upsert_recommendation(rec)
        count += 1
    return count
