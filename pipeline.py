"""Orchestration: turn a date's slate into ranked, evaluated game analyses.

This is the glue layer the `today` command (and the backtester) use so the
"pull odds -> match schedule -> build metrics -> run model -> compute EV" flow
lives in one reusable place rather than being copy-pasted into the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_cls

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
    compute_win_probability,
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
    wp: WinProbabilityResult | None = None
    evals: dict[str, SideEvaluation] | None = None
    best_side: str | None = None
    confidence: float = 0.0
    skipped_reason: str | None = None
    # Market-anchoring: the EV uses `blended_home_prob`, not the raw model prob.
    market_home_prob: float | None = None   # de-vigged "fair" market prob (home)
    blend: float = 1.0                       # market_blend weight applied
    blended_home_prob: float | None = None   # blend*model + (1-blend)*market

    @property
    def best_eval(self) -> SideEvaluation | None:
        if not self.evals or not self.best_side:
            return None
        return self.evals[self.best_side]

    def is_value(self, threshold: float) -> bool:
        be = self.best_eval
        return be is not None and be.ev_pct >= threshold and be.kelly_stake > 0

    def reasoning(self) -> dict:
        """Full JSON-able breakdown (model components + market-blend) for the DB."""
        data = self.wp.reasoning() if self.wp else {}
        data["market_anchor"] = {
            "raw_model_home_prob": round(self.wp.home_win_prob, 4) if self.wp else None,
            "market_devig_home_prob": round(self.market_home_prob, 4) if self.market_home_prob is not None else None,
            "blend_weight": self.blend,
            "blended_home_prob": round(self.blended_home_prob, 4) if self.blended_home_prob is not None else None,
        }
        return data


def _odds_by_team(game: GameOdds) -> dict[str, int]:
    out: dict[str, int] = {}
    if game.home:
        out[game.home.team] = game.home.american_odds
    if game.away:
        out[game.away.team] = game.away.american_odds
    return out


def _match_odds(scheduled: ScheduledGame, odds: list[GameOdds]) -> GameOdds | None:
    """Match a scheduled game to its odds by the unordered team pair."""
    target = frozenset({scheduled.home_team, scheduled.away_team})
    for g in odds:
        if frozenset({g.home_team, g.away_team}) == target:
            return g
    return None


def evaluate_game(
    scheduled: ScheduledGame,
    game_odds: GameOdds | None,
    team_provider: TeamMetricsProvider,
    season: int,
    as_of: date_cls,
    config: dict | None = None,
) -> GameAnalysis:
    """Run the full model + EV evaluation for a single game."""
    config = config or load_config()
    analysis = GameAnalysis(
        game_id=scheduled.game_id,
        game_date=scheduled.game_date,
        home_team=scheduled.home_team,
        away_team=scheduled.away_team,
        status=scheduled.status,
        home_pitcher=scheduled.home_pitcher.name,
        away_pitcher=scheduled.away_pitcher.name,
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

    wp = compute_win_probability(home_tp, away_tp, home_pp, away_pp, config)

    # Market anchoring: blend the raw model toward the de-vigged market so EV
    # reflects a *bounded tilt* off the sharp consensus, not raw model
    # overconfidence (a standalone heuristic otherwise "finds" edges everywhere).
    devig_method = config["ev"].get("devig_method", "power")
    market_home, _market_away = devigged_market_probs(home_odds, away_odds, devig_method)
    blend = float(config["model"].get("market_blend", 0.35))
    blended_home = blend * wp.home_win_prob + (1.0 - blend) * market_home

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
    # Catches anything that slips past the odds-band check. Tunable in config.sanity.
    max_ev = float(config.get("sanity", {}).get("max_ev", 0.30))
    if evals[best_side].ev_pct > max_ev:
        analysis.skipped_reason = (
            f"implausible EV ({evals[best_side].ev_pct * 100:.0f}%) - likely bad market data"
        )
        return analysis

    confidence = compute_confidence(
        wp, home_pp, away_pp, home_tp, away_tp, evals[best_side].ev_pct, config
    )

    analysis.wp = wp
    analysis.evals = evals
    analysis.best_side = best_side
    analysis.confidence = confidence
    analysis.market_home_prob = market_home
    analysis.blend = blend
    analysis.blended_home_prob = blended_home
    return analysis


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

    analyses: list[GameAnalysis] = []
    for sched in schedule:
        matched = _match_odds(sched, odds)
        analyses.append(evaluate_game(sched, matched, provider, season, as_of, config))

    # Rank: evaluable games by EV desc, skipped games last.
    def _sort_key(a: GameAnalysis) -> float:
        be = a.best_eval
        return be.ev_pct if be is not None else -999.0

    analyses.sort(key=_sort_key, reverse=True)
    return analyses


def save_value_bets(value_bets: list[GameAnalysis], game_date: str) -> int:
    """Persist +EV analyses as recommendations (upsert). Shared by CLI + web.

    `value_bets` should already be filtered to +EV picks. Returns the count
    saved/updated. Re-saving the same game updates its closing line + CLV.
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
        )
        upsert_recommendation(rec)
        count += 1
    return count
