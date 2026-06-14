"""GriffBet orchestration — a fork of BiffBet's pipeline glue.

Reuses every PURE, model-agnostic step of BiffBet by import (odds matching, EV
math, de-vig, market intel, stability, tier classification, the data clients and
metric providers). Forks ONLY what must diverge:

  * calls GriffBet's neutralized-base model (griffbet.win_probability),
  * logs a RAW-model pick stream alongside the blended pick (CLV split, §D),
  * uses a GriffBet adjusted-EV that OMITS the dead "-2pp fade past 5pp" haircut,
  * surfaces an EV-vs-data confidence breakdown for Strong picks (§H),
  * applies slate-level discipline (correlation haircut + exposure cap, §C),
  * persists to GriffBet's own store (griffbet.tracking).

Importing BiffBet's private pipeline helpers couples GriffBet to BiffBet
internals, but it does NOT modify BiffBet (the hard freeze), and it keeps each
computational step locked to the champion's behavior rather than drifting in a
copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_cls

from mlb_value_bot.analysis.ev_calculator import evaluate_sides, devigged_market_probs
from mlb_value_bot.analysis.pitcher_metrics import build_pitcher_profile
from mlb_value_bot.analysis.stability import classify_edge_stability
from mlb_value_bot.analysis.team_metrics import TeamMetricsProvider
from mlb_value_bot.analysis.win_probability import (
    compute_confidence,
    compute_data_confidence,
    resolve_market_blend,
)
from mlb_value_bot.data.market_intel import compute_market_intel
from mlb_value_bot.data.mlb_client import MLBClient
from mlb_value_bot.data.odds_client import GameOdds, OddsClient
from mlb_value_bot.analysis.run_environment import projected_score as _proj_score
# BiffBet pure helpers reused verbatim (read-only import; BiffBet is unchanged).
from mlb_value_bot.pipeline import (
    GameAnalysis,
    _bullpen_confidence_penalty,
    _build_bullpen_status_provider,
    _build_lineup_status_provider,
    _classify_bet_tier,
    _kelly_cap_for_tier,
    _lineup_confidence_penalty,
    _match_odds,
    _odds_by_team,
)
from mlb_value_bot.griffbet.win_probability import compute_win_probability_griff
from mlb_value_bot.griffbet.sharp_close import sharp_line_from_odds, price_for_side
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("griffbet.pipeline")


# ---------------------------------------------------------------------------
# GriffAnalysis: BiffBet's GameAnalysis + GriffBet's richer fields.
# ---------------------------------------------------------------------------
@dataclass
class GriffAnalysis(GameAnalysis):
    # Raw-model pick stream (CLV split): the side/price/prob the RAW (pre-blend)
    # model would bet, which can differ from the blended pick actually committed.
    raw_pick_side: str | None = None
    raw_pick_price: int | None = None
    raw_pick_prob: float | None = None
    raw_pick_ev: float | None = None
    # Starter-neutralized base info (None when the toggle is OFF).
    neutralization: dict | None = None
    # Sharp closing line (Pinnacle-preferred) captured this run; on the
    # near-first-pitch run this IS the sharp close CLV is graded against.
    sharp_line: "object | None" = None
    # Best-available price per side this run (for the "obtainable" close).
    best_home_line: int | None = None
    best_away_line: int | None = None
    # EV-vs-data confidence breakdown (for Strong-pick manual review).
    confidence_breakdown: dict | None = None
    # Discipline audit (set by the slate-level passes in analyze_slate_griff).
    stake_before_discipline: float | None = None
    discipline_reasons: list[str] = field(default_factory=list)

    def reasoning(self) -> dict:
        data = super().reasoning()
        be = self.best_eval
        data["raw_pick"] = {
            "side": self.raw_pick_side,
            "price": self.raw_pick_price,
            "prob": round(self.raw_pick_prob, 4) if self.raw_pick_prob is not None else None,
            "ev_pct": round(self.raw_pick_ev, 4) if self.raw_pick_ev is not None else None,
        }
        data["blended_pick"] = {
            "side": self.best_side,
            "price": be.american_odds if be is not None else None,
            "prob": round(be.model_prob, 4) if be is not None else None,
            "ev_pct": round(be.ev_pct, 4) if be is not None else None,
            "market_devig_prob": round(self.market_home_prob, 4) if self.market_home_prob is not None else None,
        }
        if self.neutralization is not None:
            data["neutralization"] = self.neutralization
        if self.confidence_breakdown is not None:
            data["confidence_breakdown"] = self.confidence_breakdown
        if self.stake_before_discipline is not None or self.discipline_reasons:
            data["discipline"] = {
                "stake_before": round(self.stake_before_discipline, 6)
                if self.stake_before_discipline is not None else None,
                "stake_after": round(be.kelly_stake, 6) if be is not None else None,
                "reasons": list(self.discipline_reasons),
            }
        return data


# ---------------------------------------------------------------------------
# Adjusted EV — GriffBet variant that OMITS the dead "-2pp fade past 5pp" branch.
# ---------------------------------------------------------------------------
def _compute_adjusted_ev_griff(
    raw_ev: float,
    sharp_fade_pp: float | None,
    fragile: bool,
    config: dict,
) -> tuple[float, list[str]]:
    """Raw EV -> sizing-grade Adjusted EV, GriffBet rules.

    Identical to BiffBet's EXCEPT the large-sharp-fade reduction branch is gone
    entirely: above sanity.max_sharp_disagreement_pp the game is SKIPPED, so the
    large-fade band never fires -- it was dead code in BiffBet. Sharp support and
    mild sharp fade are mutually exclusive; fragile stacks on top.
    """
    cfg = config.get("adjusted_ev", {})
    adj = raw_ev
    reasons: list[str] = []

    support_pp = float(cfg.get("sharp_support_pp", 3.0))
    fade_pp = float(cfg.get("sharp_fade_pp", 3.0))
    if sharp_fade_pp is not None:
        fade = sharp_fade_pp * 100.0  # to pp
        if fade <= -support_pp:
            boost = float(cfg.get("sharp_support_boost", 0.010))
            adj += boost
            reasons.append(f"+{boost * 100:.1f}pp sharps support pick ({-fade:.1f}pp)")
        elif fade >= fade_pp:
            cut = float(cfg.get("sharp_fade_reduction", 0.010))
            adj -= cut
            reasons.append(f"-{cut * 100:.1f}pp sharp fade ({fade:.1f}pp)")

    if fragile:
        cut = float(cfg.get("fragile_reduction", 0.010))
        adj -= cut
        reasons.append(f"-{cut * 100:.1f}pp fragile edge")

    return round(adj, 6), reasons


def _confidence_breakdown(raw_ev: float, data_conf: float, config: dict) -> dict:
    """Split the confidence score into its EV-magnitude vs data-quality halves
    so a Strong-pick reviewer isn't anchored on the headline number (§H).

    EV sub-score and data sub-score are each 0..100; their config weights
    (edge_magnitude vs the other three) say how much each drives the blend.
    """
    cw = config.get("confidence", {})
    weights = cw.get("weights", {})
    ev_weight = float(weights.get("edge_magnitude", 0.25))
    data_weight = max(0.0, 1.0 - ev_weight)
    edge_full = float(cw.get("edge_full_confidence", 0.10))
    ev_sub = min(abs(raw_ev) / edge_full, 1.0) * 100.0 if edge_full > 0 else 0.0
    return {
        "ev_subscore": round(ev_sub, 1),
        "ev_weight": round(ev_weight, 3),
        "data_subscore": round(data_conf, 1),
        "data_weight": round(data_weight, 3),
    }


# ---------------------------------------------------------------------------
# evaluate_game_griff — structural parallel of BiffBet.evaluate_game.
# ---------------------------------------------------------------------------
def evaluate_game_griff(
    scheduled,
    game_odds: GameOdds | None,
    team_provider: TeamMetricsProvider,
    season: int,
    as_of: date_cls,
    config: dict,
    bullpen_status_provider=None,
    lineup_status_provider=None,
) -> GriffAnalysis:
    analysis = GriffAnalysis(
        game_id=scheduled.game_id,
        game_date=scheduled.game_date,
        home_team=scheduled.home_team,
        away_team=scheduled.away_team,
        status=scheduled.status,
        home_pitcher=scheduled.home_pitcher.name if scheduled.home_pitcher else None,
        away_pitcher=scheduled.away_pitcher.name if scheduled.away_pitcher else None,
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

    max_abs_odds = float(config.get("sanity", {}).get("max_abs_odds", 800))
    if abs(home_odds) > max_abs_odds or abs(away_odds) > max_abs_odds:
        analysis.skipped_reason = (
            f"implausible odds ({home_odds:+d}/{away_odds:+d}) - likely bad market data"
        )
        return analysis
    analysis.home_odds = int(home_odds)
    analysis.away_odds = int(away_odds)
    analysis.best_home_line = int(home_odds)
    analysis.best_away_line = int(away_odds)
    # Capture the sharp (Pinnacle-preferred) line for this run -- on the
    # near-first-pitch run it's the sharp close GriffBet grades CLV against.
    analysis.sharp_line = sharp_line_from_odds(
        game_odds,
        priority=config.get("sharp_close", {}).get("priority", []),
        devig_method=config["ev"].get("devig_method", "power"),
    )

    home_pp = build_pitcher_profile(scheduled.home_pitcher.player_id, scheduled.home_pitcher.name, season, as_of)
    away_pp = build_pitcher_profile(scheduled.away_pitcher.player_id, scheduled.away_pitcher.name, season, as_of)
    home_tp = team_provider.build_team_profile(scheduled.home_team, is_home=True)
    away_tp = team_provider.build_team_profile(scheduled.away_team, is_home=False)

    home_bp = away_bp = None
    if bullpen_status_provider is not None:
        try:
            home_bp = bullpen_status_provider(scheduled.home_team, scheduled.home_team_id)
            away_bp = bullpen_status_provider(scheduled.away_team, scheduled.away_team_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("bullpen status provider failed for game %s (%s)", scheduled.game_id, exc)

    home_lu = away_lu = None
    if lineup_status_provider is not None:
        try:
            home_lu = lineup_status_provider(scheduled.home_team, scheduled.game_id, "home", scheduled.game_datetime)
            away_lu = lineup_status_provider(scheduled.away_team, scheduled.game_id, "away", scheduled.game_datetime)
        except Exception as exc:  # noqa: BLE001
            log.warning("lineup status provider failed for game %s (%s)", scheduled.game_id, exc)

    # GriffBet model (neutralized-base wrapper over BiffBet's model).
    wp, neutralization = compute_win_probability_griff(
        home_tp, away_tp, home_pp, away_pp, config, season,
        home_bullpen_status=home_bp, away_bullpen_status=away_bp,
        home_lineup_status=home_lu, away_lineup_status=away_lu,
    )
    analysis.neutralization = neutralization
    proj_score = _proj_score(home_tp, away_tp, home_pp, away_pp, config)

    devig_method = config["ev"].get("devig_method", "power")
    market_home, _market_away = devigged_market_probs(home_odds, away_odds, devig_method)

    market_intel = compute_market_intel(
        game_odds,
        sharp_books=config.get("odds_api", {}).get("sharp_bookmakers") or [],
        square_books=config.get("odds_api", {}).get("square_bookmakers") or [],
        devig_method=devig_method,
    )

    lineup_penalty = _lineup_confidence_penalty(home_lu, away_lu, config)
    bullpen_penalty = _bullpen_confidence_penalty(home_bp, away_bp, config)
    data_conf = compute_data_confidence(
        wp, home_pp, away_pp, home_tp, away_tp, config,
        lineup_penalty=lineup_penalty, bullpen_penalty=bullpen_penalty,
    )
    blend, blend_tier = resolve_market_blend(data_conf, config["model"])
    blended_home = blend * wp.home_win_prob + (1.0 - blend) * market_home

    # Divergence sanity guard (same as BiffBet).
    max_div = float(config.get("sanity", {}).get("max_model_market_divergence", 0.15))
    divergence = abs(wp.home_win_prob - market_home)
    if divergence > max_div:
        analysis.skipped_reason = (
            f"raw model ({wp.home_win_prob:.3f}) vs market ({market_home:.3f}) "
            f"diverge by {divergence:.3f} > {max_div:.2f} - market likely on news the model doesn't see"
        )
        analysis.market_intel = market_intel
        return analysis

    # RAW-model pick stream (CLV split): the side the raw (pre-blend) model bets.
    raw_evals = evaluate_sides(
        wp.home_win_prob, home_odds, away_odds,
        devig_method=devig_method,
        kelly_multiplier=config["kelly"]["fraction"],
        kelly_cap=config["kelly"]["max_bankroll_fraction"],
    )
    raw_best = max(raw_evals, key=lambda s: raw_evals[s].ev_pct)
    analysis.raw_pick_side = raw_best
    analysis.raw_pick_price = raw_evals[raw_best].american_odds
    analysis.raw_pick_prob = raw_evals[raw_best].model_prob
    analysis.raw_pick_ev = raw_evals[raw_best].ev_pct

    # Blended evals (the actual bet).
    evals = evaluate_sides(
        blended_home, home_odds, away_odds,
        devig_method=devig_method,
        kelly_multiplier=config["kelly"]["fraction"],
        kelly_cap=config["kelly"]["max_bankroll_fraction"],
    )
    best_side = max(evals, key=lambda s: evals[s].ev_pct)

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

    our_pick_home_prob = blended_home if best_side == "home" else 1.0 - blended_home
    sharp_fade_pp = None
    if market_intel.available:
        gap_home = market_intel.disagreement_with(blended_home)
        if gap_home is not None:
            sharp_fade_pp = gap_home if best_side == "home" else -gap_home

    max_sharp_fade = float(config.get("sanity", {}).get("max_sharp_disagreement_pp", 4.0))
    if sharp_fade_pp is not None and sharp_fade_pp * 100 > max_sharp_fade:
        analysis.skipped_reason = (
            f"fading sharp consensus by {sharp_fade_pp * 100:.1f}pp on {best_side} "
            f"(blended {our_pick_home_prob:.3f} vs sharps {market_intel.sharp_devig_home:.3f}) "
            f"> {max_sharp_fade:.1f}pp"
        )
        analysis.market_intel = market_intel
        return analysis

    stability = classify_edge_stability(
        components=wp.components, best_side=best_side, sharp_fade_pp=sharp_fade_pp,
        home_lineup_status=home_lu, away_lineup_status=away_lu, config=config,
    )

    raw_ev = evals[best_side].ev_pct
    adjusted_ev, adjusted_ev_reasons = _compute_adjusted_ev_griff(
        raw_ev, sharp_fade_pp, stability.label == "fragile", config
    )

    ev_threshold = float(config.get("ev", {}).get("threshold", 0.03))
    is_raw_pick = raw_ev >= ev_threshold
    tier, tier_reasons = _classify_bet_tier(
        adjusted_ev, confidence, stability.label, config, is_raw_pick=is_raw_pick
    )
    kelly_cap = _kelly_cap_for_tier(tier, config)
    original_kelly = evals[best_side].kelly_stake
    capped_kelly = round(min(original_kelly, kelly_cap), 6)
    if capped_kelly != original_kelly:
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
    analysis.confidence_breakdown = _confidence_breakdown(raw_ev, data_conf, config)
    return analysis


# ---------------------------------------------------------------------------
# Slate-level discipline (GriffBet only). Applied AFTER per-bet Kelly + tier
# caps, in order: correlation haircut, then slate exposure cap.
# ---------------------------------------------------------------------------
def _committed(analyses: list[GriffAnalysis], threshold: float) -> list[GriffAnalysis]:
    return [a for a in analyses
            if a.best_eval is not None and a.is_value(threshold) and a.best_eval.kelly_stake > 0]


def apply_discipline(analyses: list[GriffAnalysis], threshold: float, config: dict) -> None:
    """Mutate committed-bet stakes in place per the discipline config.

    1. Correlation haircut: when the committed-bet count exceeds the threshold,
       shave a flat fraction off every stake (a crowded slate is more correlated
       than independent, so per-bet Kelly over-bets it).
    2. Slate exposure cap: if the total committed stake still exceeds the cap,
       scale ALL stakes down proportionally to fit.
    Each adjustment is recorded on the analysis for the audit trail.
    """
    disc = config.get("discipline", {})
    committed = _committed(analyses, threshold)
    for a in committed:
        a.stake_before_discipline = a.best_eval.kelly_stake

    # (1) Correlation haircut.
    corr = disc.get("correlation", {})
    bet_threshold = int(corr.get("bet_count_threshold", 4))
    reduction = float(corr.get("per_bet_reduction", 0.10))
    if reduction > 0 and len(committed) > bet_threshold:
        for a in committed:
            new_stake = round(a.best_eval.kelly_stake * (1.0 - reduction), 6)
            a.best_eval.kelly_stake = new_stake
            a.discipline_reasons.append(
                f"-{reduction * 100:.0f}% correlation haircut "
                f"({len(committed)} bets > {bet_threshold})"
            )

    # (2) Slate exposure cap.
    cap = float(disc.get("max_slate_exposure", 0.0))
    if cap > 0:
        total = sum(a.best_eval.kelly_stake for a in committed)
        if total > cap:
            scale = cap / total
            for a in committed:
                new_stake = round(a.best_eval.kelly_stake * scale, 6)
                a.best_eval.kelly_stake = new_stake
                a.discipline_reasons.append(
                    f"x{scale:.3f} slate exposure cap "
                    f"(slate {total * 100:.1f}% > {cap * 100:.1f}%)"
                )


# ---------------------------------------------------------------------------
# analyze_slate_griff
# ---------------------------------------------------------------------------
def analyze_slate_griff(
    game_date: str,
    odds_client: OddsClient | None = None,
    mlb_client: MLBClient | None = None,
    config: dict | None = None,
) -> list[GriffAnalysis]:
    """Analyze every game on `game_date` with GriffBet's model + discipline."""
    config = config or load_config()
    odds_client = odds_client or OddsClient(config=config)
    mlb_client = mlb_client or MLBClient(config=config)

    schedule = mlb_client.get_schedule(game_date)
    odds = odds_client.get_odds()
    season = int(game_date[:4])
    as_of = date_cls.fromisoformat(game_date)
    provider = TeamMetricsProvider(season=season, config=config, mlb_client=mlb_client)
    bp_provider = _build_bullpen_status_provider(mlb_client, season, as_of, config)
    lu_provider = _build_lineup_status_provider(mlb_client, season, config)

    analyses: list[GriffAnalysis] = []
    for sched in schedule:
        matched = _match_odds(sched, odds)
        analyses.append(evaluate_game_griff(
            sched, matched, provider, season, as_of, config,
            bullpen_status_provider=bp_provider,
            lineup_status_provider=lu_provider,
        ))

    threshold = float(config.get("ev", {}).get("threshold", 0.03))
    apply_discipline(analyses, threshold, config)

    def _sort_key(a: GriffAnalysis) -> float:
        be = a.best_eval
        return be.ev_pct if be is not None else -999.0

    analyses.sort(key=_sort_key, reverse=True)
    return analyses


# ---------------------------------------------------------------------------
# Persistence glue
# ---------------------------------------------------------------------------
def _sharp_fields(a: GriffAnalysis) -> dict:
    sl = a.sharp_line
    if sl is None:
        return {"sharp_close_book": None, "sharp_close_home_line": None, "sharp_close_away_line": None}
    return {
        "sharp_close_book": sl.book,
        "sharp_close_home_line": sl.home_line,
        "sharp_close_away_line": sl.away_line,
    }


def save_slate_griff(analyses: list[GriffAnalysis], threshold: float, game_date: str) -> tuple[int, int]:
    """Persist the full evaluable GriffBet slate. Returns (total, n_value)."""
    from mlb_value_bot.griffbet.tracking import GriffRecord, upsert_recommendation

    total = n_value = 0
    for a in analyses:
        be = a.best_eval
        if be is None or a.wp is None:
            continue
        is_value = be.ev_pct >= threshold and be.kelly_stake > 0
        rec = GriffRecord(
            date=game_date, game_id=a.game_id, home_team=a.home_team, away_team=a.away_team,
            recommended_side=a.best_side or "",
            model_prob=be.model_prob, market_prob_devigged=be.market_prob_devigged,
            american_odds=be.american_odds, decimal_odds=be.decimal_odds,
            ev_pct=be.ev_pct, kelly_stake=be.kelly_stake, confidence=a.confidence,
            reasoning=a.reasoning(),
            raw_model_prob=a.wp.home_win_prob, blended_prob=a.blended_home_prob,
            raw_pick_side=a.raw_pick_side, raw_pick_open=a.raw_pick_price,
            best_home_line=a.best_home_line, best_away_line=a.best_away_line,
            is_value=is_value, **_sharp_fields(a),
        )
        upsert_recommendation(rec)
        total += 1
        n_value += int(is_value)
    return total, n_value


def refresh_skipped_closing_lines_griff(analyses: list[GriffAnalysis], game_date: str) -> int:
    """Refresh closing + sharp CLV on committed bets whose game was SKIPPED this
    run (so a skip never freezes GriffBet's CLV). Only analyses that carry
    trustworthy prices (cleared the implausible-odds guard) participate."""
    from mlb_value_bot.griffbet.tracking import refresh_closing_lines

    n = 0
    for a in analyses:
        if a.best_eval is not None:
            continue
        if a.best_home_line is None or a.best_away_line is None:
            continue
        sf = _sharp_fields(a)
        if refresh_closing_lines(
            game_date, a.game_id,
            best_home_line=a.best_home_line, best_away_line=a.best_away_line,
            raw_pick_side=a.raw_pick_side, **sf,
        ):
            n += 1
    return n
