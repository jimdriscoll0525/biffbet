"""Football pipeline — orchestration glue (the football twin of pipeline.py).

M2 scope (this file grows with the milestones):
  build_league_context — assemble one league's unit stats (prior-blended),
  percentiles (SP+-adjusted for CFB), OL inputs, and schedule; score_game —
  matchup edges + OL layer for one game. The `matchups` CLI debug board and
  the (M3) pick pipeline both run through these, so the model the debug board
  shows IS the model that prices bets.

Layering: this module calls football/data/* for I/O and football/analysis/*
for math. It owns no formulas itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mlb_value_bot.football import season_for_date  # noqa: F401  (used by CLI callers)
from mlb_value_bot.football.analysis import percentiles as pctl
from mlb_value_bot.football.analysis import unit_stats as ustats
from mlb_value_bot.football.analysis.matchup import GameMatchup, game_matchup
from mlb_value_bot.football.analysis.ol_layer import OLGrade, nfl_ol_continuity, ol_grade
from mlb_value_bot.utils import get_logger

log = get_logger("football.pipeline")


@dataclass
class LeagueContext:
    league: str                      # "nfl" | "cfb"
    season: int
    week: int
    config: dict
    unit_stats: pd.DataFrame         # prior-blended, index = canonical team
    percentiles: pd.DataFrame        # unit + stat percentiles, same index
    games: pd.DataFrame              # schedule/games frame (canonical team cols)
    snap_counts: pd.DataFrame | None = None       # NFL only
    g5_teams: set[str] = field(default_factory=set)  # CFB only
    cfb_normalizer: object = None    # CFB only: Odds-API name -> school

    def week_games(self, week: int) -> pd.DataFrame:
        if self.games.empty or "week" not in self.games.columns:
            return pd.DataFrame(columns=["home_team", "away_team"])
        return self.games[self.games["week"] == week]


@dataclass
class ScoredGame:
    home: str
    away: str
    matchup: GameMatchup
    ol_home: OLGrade
    ol_away: OLGrade
    home_units: dict
    away_units: dict


def _nfl_context(season: int, week: int, config: dict) -> LeagueContext:
    from mlb_value_bot.football.data import nfl_client

    current = ustats.nfl_unit_stats(nfl_client.pbp(season, config))
    prior = ustats.nfl_unit_stats(nfl_client.pbp(season - 1, config)) \
        if pctl.prior_weight(week, config) > 0 else pd.DataFrame()
    blended = pctl.blend_with_prior(current, prior, week, config)
    if blended.empty and not prior.empty:
        blended = prior   # pre-season board: run purely on last season

    sched = nfl_client.schedules(season, config)
    games = pd.DataFrame(columns=["home_team", "away_team", "week"])
    if not sched.empty:
        games = sched.rename(columns={"gameday": "date"})

    return LeagueContext(
        league="nfl", season=season, week=week, config=config,
        unit_stats=blended, percentiles=pctl.unit_percentiles(blended),
        games=games, snap_counts=nfl_client.snap_counts(season, config),
    )


def _cfb_context(season: int, week: int, config: dict) -> LeagueContext:
    from mlb_value_bot.football.data.cfbd_client import CfbdClient
    from mlb_value_bot.football.data.teams import build_cfb_matcher

    client = CfbdClient(config)
    if not client.configured:
        log.warning("CFBD not configured; CFB context is empty")
        return LeagueContext("cfb", season, week, config,
                             pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    teams = client.fbs_teams(season)
    current = ustats.cfb_unit_stats(client.season_stats(season), client.ppa_teams(season))
    prior = ustats.cfb_unit_stats(client.season_stats(season - 1), client.ppa_teams(season - 1)) \
        if pctl.prior_weight(week, config) > 0 else pd.DataFrame()
    blended = pctl.blend_with_prior(current, prior, week, config)
    if blended.empty and not prior.empty:
        blended = prior

    # Percentile pool = FBS only (never let stray FCS rows join the pool).
    if not teams.empty and "school" in teams.columns and not blended.empty:
        blended = blended[blended.index.isin(set(teams["school"]))]
    pcts = pctl.unit_percentiles(blended)
    pcts = _apply_sp_adjustment(pcts, client.sp_ratings(season), config)

    games = client.games(season)
    rename = {c: t for c, t in
              (("homeTeam", "home_team"), ("awayTeam", "away_team"),
               ("homePoints", "home_score"), ("awayPoints", "away_score"),
               ("startDate", "start_date"), ("venueId", "venue_id"))
              if c in games.columns}
    games = games.rename(columns=rename) if not games.empty else games

    g5 = set()
    normalizer = None
    if not teams.empty and "school" in teams.columns:
        power = set(config.get("college", {}).get(
            "power_conferences", ["SEC", "Big Ten", "Big 12", "ACC", "FBS Independents"]))
        if "conference" in teams.columns:
            g5 = set(teams.loc[~teams["conference"].isin(power), "school"])
        mascots = list(zip(teams["school"],
                           teams["mascot"] if "mascot" in teams.columns else [None] * len(teams)))
        normalizer = build_cfb_matcher(mascots)

    return LeagueContext("cfb", season, week, config, blended, pcts, games,
                         g5_teams=g5, cfb_normalizer=normalizer)


def _apply_sp_adjustment(pcts: pd.DataFrame, sp: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Blend CFB unit percentiles toward the school's overall SP+ percentile —
    the competition-level correction (stats padded on a G5 schedule get pulled
    toward the team's schedule-adjusted quality)."""
    blend = float(config.get("college", {}).get("sp_blend", 0.30))
    if pcts.empty or sp is None or sp.empty or blend <= 0:
        return pcts
    team_col = "team" if "team" in sp.columns else None
    rating_col = "rating" if "rating" in sp.columns else None
    if not team_col or not rating_col:
        return pcts
    sp_pct = pctl.percentile_series(
        pd.to_numeric(sp.set_index(team_col)[rating_col], errors="coerce"), True)
    sp_pct = sp_pct.reindex(pcts.index)
    out = pcts.copy()
    unit_cols = [c for c in out.columns if c.endswith("_pct")]
    for col in unit_cols:
        adjusted = (1 - blend) * out[col] + blend * sp_pct
        out[col] = adjusted.where(sp_pct.notna(), out[col])
    return out


def build_league_context(league: str, season: int, week: int, config: dict) -> LeagueContext:
    if league == "nfl":
        return _nfl_context(season, week, config)
    if league == "cfb":
        return _cfb_context(season, week, config)
    raise ValueError(f"Unknown league '{league}'")


@dataclass
class FootballPick:
    """One priced market on one game (a bet when is_value, else an analysis)."""
    market: str                  # "spread" | "total"
    side: str                    # home | away | over | under
    line: float
    american_odds: int
    model_prob: float            # blended P(side) — what EV is computed from
    market_prob: float           # de-vigged P(side) at the bet book
    p_push: float
    raw_ev: float
    adjusted_ev: float
    adjustments: list[str]
    confidence: float
    tier: str                    # "standard" | "strong" | "pass"
    stake_pct: float             # flat display stake (0 when pass)
    stability_label: str
    is_value: bool
    hold_reason: str | None      # divergence guard / weather hold / below threshold
    reasoning: dict


@dataclass
class FootballGameAnalysis:
    league: str
    date: str                    # kickoff date (ET-ish, from commence_time)
    week: int | None
    game_id: str
    home: str
    away: str
    commence_time: str
    picks: list[FootballPick]

    @property
    def bets(self) -> list[FootballPick]:
        return [p for p in self.picks if p.is_value]


def score_game(ctx: LeagueContext, home: str, away: str,
               script_lean: float = 0.0) -> ScoredGame | None:
    """Matchup edges + OL layer for one game; None when either team is
    outside the percentile pool (FCS opponent, expansion gap)."""
    if ctx.percentiles.empty or home not in ctx.percentiles.index \
            or away not in ctx.percentiles.index:
        return None

    home_units = ctx.percentiles.loc[home].to_dict()
    away_units = ctx.percentiles.loc[away].to_dict()
    home_raw = ctx.unit_stats.loc[home].to_dict() if home in ctx.unit_stats.index else {}
    away_raw = ctx.unit_stats.loc[away].to_dict() if away in ctx.unit_stats.index else {}

    cont_h = cont_a = None
    if ctx.league == "nfl" and ctx.snap_counts is not None:
        cont_h = nfl_ol_continuity(ctx.snap_counts, home, ctx.week)
        cont_a = nfl_ol_continuity(ctx.snap_counts, away, ctx.week)
    ol_h = ol_grade(home_raw, ctx.league, ctx.config, continuity=cont_h)
    ol_a = ol_grade(away_raw, ctx.league, ctx.config, continuity=cont_a)

    # The OL modifier lands on the team's OFFENSIVE unit percentiles (edge =
    # O - D, so +points on O is +points on both of that team's phase edges),
    # clamped to the 0-100 scale.
    def bump(units: dict, pts: float) -> dict:
        adj = dict(units)
        for key in ("pass_off_pct", "rush_off_pct"):
            val = adj.get(key)
            if val is not None and val == val:
                adj[key] = max(0.0, min(100.0, val + pts))
        return adj

    matchup = game_matchup(bump(home_units, ol_h.points),
                           bump(away_units, ol_a.points),
                           ctx.config, script_lean=script_lean)
    return ScoredGame(home=home, away=away, matchup=matchup,
                      ol_home=ol_h, ol_away=ol_a,
                      home_units=home_units, away_units=away_units)


# =============================================================================
# M3 — market evaluation (projections -> blend -> EV -> adjusted EV -> picks)
# =============================================================================

def _compute_adjusted_ev(raw_ev: float, sharp_gap_side_pp: float | None,
                         fragile: bool, config: dict) -> tuple[float, list[str]]:
    """THE single home for sharp-signal EV magnitude (mirrors pipeline.py's
    _compute_adjusted_ev contract). sharp_gap_side_pp is the sharp-vs-book
    de-vigged probability gap FOR OUR SIDE in probability points; stability
    only ever hands over the measurement, never applies it."""
    cfg = config.get("adjusted_ev", {})
    adj = raw_ev
    notes: list[str] = []
    thresh = float(cfg.get("sharp_support_pp", 2.0))
    if sharp_gap_side_pp is not None:
        if sharp_gap_side_pp >= thresh:
            boost = float(cfg.get("sharp_support_boost", 0.010))
            adj += boost
            notes.append(f"sharp support {sharp_gap_side_pp:+.1f}pp: +{boost * 100:.1f}%")
        elif sharp_gap_side_pp <= -thresh:
            cut = float(cfg.get("sharp_fade_reduction", 0.015))
            adj -= cut
            notes.append(f"sharp fade {sharp_gap_side_pp:+.1f}pp: -{cut * 100:.1f}%")
    if fragile:
        cut = float(cfg.get("fragile_reduction", 0.010))
        adj -= cut
        notes.append(f"fragile edge: -{cut * 100:.1f}%")
    return adj, notes


def _evaluate_market(ctx: LeagueContext, scored: ScoredGame, view, projection,
                     weather, g5_involved: bool, explosive_involved: bool,
                     games_min: float | None) -> FootballPick | None:
    """Price one market (spread or total) into a FootballPick, or None when
    the market can't be evaluated at all."""
    from mlb_value_bot.football.analysis import football_stability as stab
    from mlb_value_bot.football.analysis.football_confidence import confidence_for_pick
    from mlb_value_bot.football.analysis.football_ev import blend_probability, ev_with_push
    from mlb_value_bot.football.analysis.projections import (
        cover_probabilities,
        total_probabilities,
    )

    config = ctx.config
    pcfg = config.get("projections", {})
    league = ctx.league
    is_total = view.market == "total"
    sigma = float(pcfg.get(f"{league}_total_sigma" if is_total else f"{league}_margin_sigma",
                           10.0 if is_total else 13.2))

    if is_total:
        p_a_model, p_push, _ = total_probabilities(projection.total, view.line, sigma)
        model_mu, divergence_guard = projection.total, float(pcfg.get("max_total_divergence_pts", 6.0))
    else:
        p_a_model, p_push, _ = cover_probabilities(projection.margin, view.line, sigma)
        model_mu, divergence_guard = projection.margin, float(pcfg.get("max_spread_divergence_pts", 4.5))

    hold_reason = None
    divergence = abs(model_mu - view.anchor_mean) if view.anchor_mean is not None else None
    if divergence is not None and divergence > divergence_guard:
        hold_reason = (f"divergence guard: model {model_mu:+.1f} vs market mean "
                       f"{view.anchor_mean:+.1f} ({divergence:.1f} > {divergence_guard})")

    outdoor_total = is_total and not weather.indoor
    if is_total and outdoor_total and not weather.available \
            and config.get("weather", {}).get("require_for_bet", True):
        hold_reason = hold_reason or "outdoor total without weather -> analysis-only"

    epa_available = any(v is not None for v in
                        (projection.home_detail.epa_pass, projection.home_detail.epa_rush))
    ol_proxy_only = True   # v1: OL always rides the proxy — keep the label honest
    stability = stab.assess(view, games_min=games_min, epa_available=epa_available,
                            weather_available=weather.available,
                            outdoor_total=outdoor_total,
                            ol_proxy_only=ol_proxy_only, config=config)

    # Blend model with market on side A (home/over), then pick the better side.
    p_a = blend_probability(p_a_model, view.devig_p_a, config)
    p_b = max(0.0, 1.0 - p_a - p_push)
    ev_a = ev_with_push(p_a, p_push, view.price_a)
    ev_b = ev_with_push(p_b, p_push, view.price_b)

    if ev_a >= ev_b:
        side = "over" if is_total else "home"
        p_side, price, raw_ev, market_p = p_a, view.price_a, ev_a, view.devig_p_a
        gap_side = stability.sharp_gap_pp
    else:
        side = "under" if is_total else "away"
        p_side, price, raw_ev, market_p = p_b, view.price_b, ev_b, 1.0 - view.devig_p_a
        gap_side = -stability.sharp_gap_pp if stability.sharp_gap_pp is not None else None

    adjusted_ev, adjustments = _compute_adjusted_ev(
        raw_ev, gap_side, stability.label == stab.LABEL_FRAGILE, config)
    if stability.rlm_note:
        adjustments.append(f"noted: {stability.rlm_note}")

    completeness = sum((
        1.0 if epa_available else 0.0,
        1.0 if weather.available or not is_total else 0.0,
        1.0 if view.n_sharp_books > 0 else 0.0,
        1.0 if (games_min or 0) >= 4 else 0.0,
    )) / 4.0
    conf = confidence_for_pick(
        edge_abs=abs(scored.matchup.home_edge), completeness=completeness,
        league=league, g5_involved=g5_involved, market=view.market,
        explosive_involved=explosive_involved, stability_label=stability.label,
        config=config)

    threshold = float(config.get("ev", {}).get("threshold", 0.03))
    bcfg = config.get("betting", {})
    is_value = hold_reason is None and adjusted_ev > threshold
    if is_value:
        strong = (league == "nfl" and scored.matchup.dual_edge_side is not None
                  and adjusted_ev >= 2 * threshold
                  and stability.label != stab.LABEL_FRAGILE)
        tier = "strong" if strong else "standard"
        stake = float(bcfg.get("flat_stake_pct_strong", 0.02)) if strong \
            else float(bcfg.get("flat_stake_pct", 0.01))
        stake *= conf.stake_mult
    else:
        tier, stake = "pass", 0.0
        if hold_reason is None and adjusted_ev <= threshold:
            hold_reason = f"adjusted EV {adjusted_ev * 100:+.1f}% below threshold"

    reasoning = {
        "model_tag": config.get("model_tag", "matchup_v1"),
        "matchup": {
            "home_edge": scored.matchup.home_edge,
            "archetype": scored.matchup.archetype,
            "dual_edge_side": scored.matchup.dual_edge_side,
            "turnover_flag_side": scored.matchup.turnover_flag_side,
            "pass_weight": scored.matchup.pass_weight,
            "rush_weight": scored.matchup.rush_weight,
            "notes": scored.matchup.notes,
            "edges": [{
                "offense_side": e.offense_side, "phase": e.phase,
                "o_pct": e.o_pct, "d_pct": e.d_pct, "edge": e.edge,
                "archetype": e.archetype, "note": e.note,
            } for e in scored.matchup.edges],
        },
        "units": {"home": _clean_units(scored.home_units),
                  "away": _clean_units(scored.away_units)},
        "ol": {
            "home": {"grade": scored.ol_home.grade, "points": scored.ol_home.points,
                     "continuity": scored.ol_home.continuity, "notes": scored.ol_home.notes},
            "away": {"grade": scored.ol_away.grade, "points": scored.ol_away.points,
                     "continuity": scored.ol_away.continuity, "notes": scored.ol_away.notes},
        },
        "projection": {
            "home_pts": projection.home_pts, "away_pts": projection.away_pts,
            "margin": projection.margin, "total_raw": projection.total_raw,
            "total": projection.total, "weather_mult": projection.weather_mult,
            "notes": projection.home_detail.notes + projection.away_detail.notes,
        },
        "market": {
            "book": view.book, "line": view.line,
            "price_a": view.price_a, "price_b": view.price_b,
            "devig_p_a": round(view.devig_p_a, 4),
            "sharp_line": view.sharp_line,
            "sharp_devig_p_a": round(view.sharp_devig_p_a, 4) if view.sharp_devig_p_a is not None else None,
            "anchor_mean": view.anchor_mean,
            "divergence": round(divergence, 2) if divergence is not None else None,
            "n_sharp_books": view.n_sharp_books,
        },
        "blend": {"market_blend": float(config.get("projections", {}).get("market_blend", 0.35)),
                  "p_model_a": round(p_a_model, 4), "p_market_a": round(view.devig_p_a, 4),
                  "p_final_a": round(p_a, 4), "p_push": round(p_push, 4)},
        "adjusted_ev": {"raw_ev_pct": round(raw_ev, 4),
                        "adjusted_ev_pct": round(adjusted_ev, 4),
                        "adjustments": adjustments},
        "confidence": conf.components,
        "stability": {"label": stability.label, "flags": stability.flags,
                      "rlm_note": stability.rlm_note,
                      "sharp_gap_pp": stability.sharp_gap_pp},
        "weather": {"multiplier": weather.multiplier, "available": weather.available,
                    "indoor": weather.indoor, "temp_f": weather.temp_f,
                    "wind_mph": weather.wind_mph, "note": weather.note},
        "hold_reason": hold_reason,
    }

    return FootballPick(
        market=view.market, side=side, line=view.line, american_odds=price,
        model_prob=round(p_side, 4), market_prob=round(market_p, 4),
        p_push=round(p_push, 4), raw_ev=round(raw_ev, 4),
        adjusted_ev=round(adjusted_ev, 4), adjustments=adjustments,
        confidence=conf.value, tier=tier, stake_pct=round(stake, 4),
        stability_label=stability.label, is_value=is_value,
        hold_reason=hold_reason, reasoning=reasoning,
    )


def _clean_units(units: dict) -> dict:
    return {k: (round(v, 1) if isinstance(v, float) and v == v else None)
            for k, v in units.items() if k.endswith("_pct")}


def _match_game_row(ctx: LeagueContext, home: str, away: str) -> pd.Series | None:
    g = ctx.games
    if g.empty or "home_team" not in g.columns:
        return None
    hit = g[(g["home_team"] == home) & (g["away_team"] == away)]
    if hit.empty:
        return None
    # Nearest upcoming when a matchup repeats (CFB championship rematches).
    return hit.iloc[-1]


def evaluate_league_slate(league: str, date_iso: str, config: dict,
                          odds_games=None) -> list[FootballGameAnalysis]:
    """Analyze one league's current board: odds -> canonical teams -> matchup
    -> projection -> per-market picks. `odds_games` injectable for tests."""
    from mlb_value_bot.football.analysis.football_ev import market_view
    from mlb_value_bot.football.analysis.percentiles import percentile_series
    from mlb_value_bot.football.data import football_odds, stadiums
    from mlb_value_bot.football.data.football_weather import game_weather
    from mlb_value_bot.football.data.teams import normalize_nfl

    season = season_for_date(date_iso)
    if odds_games is None:
        odds_games = football_odds.fetch_league_odds(league, config)
    if not odds_games:
        return []

    # Week for prior-blend/continuity: the earliest week among matched games
    # would need the context first — bootstrap with a coarse schedule guess.
    ctx = build_league_context(league, season, _infer_week(config, league, season, date_iso), config)
    if ctx.percentiles.empty:
        log.warning("%s: no unit stats for season %d; slate skipped", league, season)
        return []

    pcfg = config.get("projections", {})
    pool_rz = float(ctx.unit_stats["rz_td_rate"].mean()) \
        if "rz_td_rate" in ctx.unit_stats.columns else None
    explosive_cut = float(config.get("variance", {}).get("explosive_epa_pctile", 80))
    q4_pct = percentile_series(ctx.unit_stats["q4_epa"], True) \
        if "q4_epa" in ctx.unit_stats.columns else None

    out: list[FootballGameAnalysis] = []
    for game in odds_games:
        if league == "nfl":
            home = normalize_nfl(game.home_name_raw)
            away = normalize_nfl(game.away_name_raw)
        else:
            home = ctx.cfb_normalizer(game.home_name_raw) if ctx.cfb_normalizer else None
            away = ctx.cfb_normalizer(game.away_name_raw) if ctx.cfb_normalizer else None
        if not home or not away:
            log.info("%s: unmatched slate names %s @ %s", league,
                     game.away_name_raw, game.home_name_raw)
            continue

        row = _match_game_row(ctx, home, away)
        week = int(row["week"]) if row is not None and "week" in row and pd.notna(row["week"]) else None
        game_id = str(row["game_id"]) if row is not None and "game_id" in row and pd.notna(row.get("game_id")) \
            else (str(row["id"]) if row is not None and "id" in row and pd.notna(row.get("id"))
                  else game.event_id)

        # Game-script lean from the sharp spread magnitude (blowouts run).
        sigma_m = float(pcfg.get(f"{league}_margin_sigma", 13.2))
        spread_view = market_view(game, "spread", config, sigma_m)
        lean = 0.0
        if spread_view is not None:
            ref_line = spread_view.sharp_line if spread_view.sharp_line is not None else spread_view.line
            lean = min(1.0, abs(ref_line) / float(pcfg.get("script_lean_full_spread", 21.0)))

        scored = score_game(ctx, home, away, script_lean=lean)
        if scored is None:
            log.info("%s: %s @ %s outside percentile pool", league, away, home)
            continue

        # Weather (kickoff-hour, suppress-only).
        if league == "nfl":
            roof = row.get("roof") if row is not None else None
            indoor = stadiums.is_indoor(roof, home)
            coords = stadiums.nfl_coords(home)
        else:
            venue = None
            if row is not None and "venue_id" in row and pd.notna(row.get("venue_id")):
                from mlb_value_bot.football.data.cfbd_client import CfbdClient
                client = CfbdClient(config)
                venues = client.venues() if client.configured else pd.DataFrame()
                venue = stadiums.cfb_venue(venues, int(row["venue_id"]))
            indoor = bool(venue["dome"]) if venue else False
            coords = (venue["lat"], venue["lon"]) if venue else None
        weather = game_weather(coords[0] if coords else None,
                               coords[1] if coords else None,
                               game.commence_time, indoor, config)

        projection = project_game_for(ctx, scored, weather, pool_rz, lean)

        g5_involved = ctx.league == "cfb" and bool({home, away} & ctx.g5_teams)
        explosive_involved = False
        if q4_pct is not None:
            explosive_involved = any(
                t in q4_pct.index and q4_pct[t] == q4_pct[t] and q4_pct[t] >= explosive_cut
                for t in (home, away))
        games_min = None
        if "games" in ctx.unit_stats.columns:
            gm = [ctx.unit_stats.at[t, "games"] for t in (home, away)
                  if t in ctx.unit_stats.index]
            games_min = float(min(gm)) if gm else None

        picks: list[FootballPick] = []
        sigma_t = float(pcfg.get(f"{league}_total_sigma", 10.0))
        for market_name, sigma in (("spread", sigma_m), ("total", sigma_t)):
            view = spread_view if market_name == "spread" \
                else market_view(game, "total", config, sigma_t)
            if view is None:
                continue
            pick = _evaluate_market(ctx, scored, view, projection, weather,
                                    g5_involved, explosive_involved, games_min)
            if pick is not None:
                picks.append(pick)

        max_picks = int(config.get("ev", {}).get("max_picks_per_game", 3))
        bets = [p for p in picks if p.is_value]
        if len(bets) > max_picks:
            keep = set(id(p) for p in sorted(bets, key=lambda p: -p.adjusted_ev)[:max_picks])
            for p in bets:
                if id(p) not in keep:
                    p.is_value, p.tier, p.stake_pct = False, "pass", 0.0
                    p.hold_reason = "capped: better markets on this game"

        out.append(FootballGameAnalysis(
            league=league, date=game.commence_time[:10], week=week, game_id=game_id,
            home=home, away=away, commence_time=game.commence_time, picks=picks))
    return out


def project_game_for(ctx: LeagueContext, scored: ScoredGame, weather,
                     pool_rz: float | None, script_lean: float):
    """Assemble raw unit rows + phase weights into a GameProjection."""
    from mlb_value_bot.football.analysis.matchup import phase_weights
    from mlb_value_bot.football.analysis.projections import project_game

    w_pass, w_rush = phase_weights(ctx.config, script_lean)
    home_raw = ctx.unit_stats.loc[scored.home].to_dict() if scored.home in ctx.unit_stats.index else {}
    away_raw = ctx.unit_stats.loc[scored.away].to_dict() if scored.away in ctx.unit_stats.index else {}
    return project_game(home_raw, away_raw, ctx.league, ctx.config,
                        w_pass, w_rush, weather_mult=weather.multiplier,
                        pool_rz_avg=pool_rz)


def _infer_week(config: dict, league: str, season: int, date_iso: str) -> int:
    """Coarse in-season week estimate (only drives prior-blend weight and
    OL-continuity lookback): weeks since Sep 1 of the season year, clamped."""
    from datetime import date as _date

    start = _date(season, 9, 1)
    d = _date(int(date_iso[:4]), int(date_iso[5:7]), int(date_iso[8:10]))
    return max(1, min(20, (d - start).days // 7 + 1))
