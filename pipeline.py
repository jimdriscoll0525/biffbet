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
    tier_multiplier: float = 0.0
    tier_reasons: list[str] = field(default_factory=list)
    # Lineup snapshots used to populate reasoning["lineup"] (UI chips + breakdown).
    home_lineup_status: "object | None" = None
    away_lineup_status: "object | None" = None

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
            "multiplier": self.tier_multiplier,
            "reasons": self.tier_reasons,
            # Raw Kelly before the tier multiplier was applied (only set when scaled).
            "raw_kelly": getattr(self, "_tier_original_kelly", None),
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

    # Market anchoring: blend the raw model toward the de-vigged market so EV
    # reflects a *bounded tilt* off the sharp consensus, not raw model
    # overconfidence (a standalone heuristic otherwise "finds" edges everywhere).
    devig_method = config["ev"].get("devig_method", "power")
    market_home, _market_away = devigged_market_probs(home_odds, away_odds, devig_method)

    # Dynamic blend: the model earns more weight when the underlying DATA is
    # trustworthy (good pitcher samples, team data complete, components in
    # agreement). EV is deliberately NOT part of this -- letting EV drive the
    # blend would create a feedback loop where the model talks itself into
    # bigger edges. Falls back to a fixed blend if config.model.market_blend is
    # still a scalar.
    #
    # `lineup_penalty` shaves confidence when either team's lineup is still
    # projected at run time. We're betting on a roster guess in that case, so
    # the model should defer more to the market -- which is exactly what a
    # lower data confidence achieves through the blend tier table.
    lineup_penalty = _lineup_confidence_penalty(home_lu, away_lu, config)
    data_conf = compute_data_confidence(
        wp, home_pp, away_pp, home_tp, away_tp, config, lineup_penalty=lineup_penalty,
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
        lineup_penalty=lineup_penalty,
    )

    # Bet sizing tiers: classify the pick into Pass/Small/Standard/Strong and
    # apply a stake multiplier to the raw Kelly stake. This bakes our
    # "reduce Kelly when confidence is low / edge is modest" guardrails into
    # the persisted stake instead of leaving them to the user. The unscaled
    # Kelly is preserved in reasoning_json for transparency.
    tier, tier_mult, tier_reasons = _classify_bet_tier(
        evals[best_side].ev_pct, confidence, config
    )
    if tier_mult != 1.0:
        # Mutate in place: the consumer only ever uses evals[best_side].
        original_kelly = evals[best_side].kelly_stake
        evals[best_side].kelly_stake = round(original_kelly * tier_mult, 6)
        analysis._tier_original_kelly = original_kelly  # type: ignore[attr-defined]

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
    analysis.tier_multiplier = tier_mult
    analysis.tier_reasons = tier_reasons
    analysis.home_lineup_status = home_lu
    analysis.away_lineup_status = away_lu
    return analysis


# --- Bet sizing tiers --------------------------------------------------------
def _classify_bet_tier(
    ev_pct: float, confidence: float, config: dict
) -> tuple[str, float, list[str]]:
    """Classify a recommendation into Pass / Small / Standard / Strong.

    Returns (tier_name, kelly_multiplier, reasons). The multiplier scales the
    raw Kelly stake -- always <= 1.0 today (we only ever reduce, never grow).

    Rules (intentionally simple and readable):
      * EV below threshold              -> "pass"     0.0x
      * EV below 5% or confidence < 60  -> "small"    0.5x  (lean, not action)
      * EV >= 10% AND confidence >= 75  -> "strong"   1.0x  (full quarter-Kelly)
      * else                            -> "standard" 1.0x

    "Strong" and "Standard" use the same multiplier today -- the distinction
    is informational so the UI can show a separate badge. We may dial Strong
    UP (e.g. 1.25x within the cap) once CLV evidence supports it; we will not
    dial it up speculatively.
    """
    ev_cfg = config.get("ev", {})
    sizing = config.get("bet_sizing", {})
    threshold = float(ev_cfg.get("threshold", 0.03))
    strong_ev = float(sizing.get("strong_ev", 0.10))
    standard_ev = float(sizing.get("standard_ev", 0.05))
    strong_conf = float(sizing.get("strong_confidence", 75.0))
    min_conf = float(sizing.get("min_standard_confidence", 60.0))
    small_mult = float(sizing.get("small_multiplier", 0.5))

    reasons: list[str] = []
    if ev_pct < threshold:
        reasons.append(f"EV {ev_pct * 100:.1f}% below {threshold * 100:.1f}% threshold")
        return "pass", 0.0, reasons

    if ev_pct < standard_ev:
        reasons.append(f"EV {ev_pct * 100:.1f}% in lean range (< {standard_ev * 100:.1f}%)")
        return "small", small_mult, reasons
    if confidence < min_conf:
        reasons.append(f"confidence {confidence:.0f} below {min_conf:.0f}")
        return "small", small_mult, reasons

    if ev_pct >= strong_ev and confidence >= strong_conf:
        reasons.append(f"EV {ev_pct * 100:.1f}% & conf {confidence:.0f} both clear strong thresholds")
        return "strong", 1.0, reasons

    return "standard", 1.0, reasons


def _lineup_confidence_penalty(home_lu, away_lu, config: dict) -> float:
    """Confidence points to subtract for projected lineups.

    Counts one penalty unit per team whose lineup is projected (or
    unavailable). Confirmed on both sides = 0 penalty.
    """
    cfg = config.get("lineup", {})
    per_team = float(cfg.get("projected_confidence_penalty_per_team", 3.0))
    n = 0
    for lu in (home_lu, away_lu):
        if lu is None or lu.status != "confirmed":
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
