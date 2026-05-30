"""Core correctness tests: EV math, the model breakdown, and the tracking DB.

These avoid all network / pybaseball so they run anywhere. Use real pytest
(`pytest`) or the built-in runner (`python -m mlb_value_bot.tests.test_core`).
"""
from __future__ import annotations

import math

from mlb_value_bot.analysis import ev_calculator as ev
from mlb_value_bot.analysis.pitcher_metrics import PitcherProfile
from mlb_value_bot.analysis.team_metrics import TeamProfile
from mlb_value_bot.analysis.win_probability import (
    compute_confidence,
    compute_win_probability,
    log5,
    regress_winpct,
)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# --- Odds conversions --------------------------------------------------------
def test_american_to_decimal():
    assert approx(ev.american_to_decimal(150), 2.5)
    assert approx(ev.american_to_decimal(-120), 1.0 + 100 / 120)
    assert approx(ev.american_to_decimal(100), 2.0)


def test_decimal_to_american_roundtrip():
    for a in (-250, -110, 100, 145, 320):
        assert ev.decimal_to_american(ev.american_to_decimal(a)) == a


def test_implied_prob():
    assert approx(ev.american_to_implied(100), 0.5)
    assert approx(ev.american_to_implied(-110), 110 / 210, tol=1e-9)


# --- De-vig ------------------------------------------------------------------
def test_devig_sums_to_one():
    raw = [ev.american_to_implied(-110), ev.american_to_implied(-110)]
    for method in ("proportional", "power"):
        fair = ev.devig(raw, method=method)
        assert approx(sum(fair), 1.0, tol=1e-6), method
    # Symmetric market -> 50/50 either way.
    fair = ev.devig(raw, method="power")
    assert approx(fair[0], 0.5, tol=1e-4)


def test_devig_power_preserves_order():
    # Favorite/longshot: -200 / +170. Fair probs keep the favorite > underdog.
    raw = [ev.american_to_implied(-200), ev.american_to_implied(170)]
    fair = ev.devig(raw, method="power")
    assert approx(sum(fair), 1.0, tol=1e-6)
    assert fair[0] > fair[1]


# --- EV + Kelly --------------------------------------------------------------
def test_ev_pct():
    # 55% model prob at +100 (decimal 2.0) -> EV = 0.55*2 - 1 = 0.10
    assert approx(ev.ev_pct(0.55, 2.0), 0.10)


def test_kelly_no_edge_is_zero():
    # Fair coin at +100 has zero edge -> zero stake.
    assert ev.kelly_fraction(0.5, 2.0) == 0.0


def test_kelly_capped():
    # Big edge gets capped at 2% by default.
    f = ev.kelly_fraction(0.90, 2.0, kelly_multiplier=1.0, cap=0.02)
    assert approx(f, 0.02)


def test_evaluate_sides_consistency():
    res = ev.evaluate_sides(0.55, home_american=-120, away_american=110)
    assert approx(res["home"].model_prob + res["away"].model_prob, 1.0)
    assert approx(res["home"].market_prob_devigged + res["away"].market_prob_devigged, 1.0, tol=1e-6)


# --- Model -------------------------------------------------------------------
def test_log5_symmetry():
    assert approx(log5(0.6, 0.6), 0.5)
    assert approx(log5(0.6, 0.4) + log5(0.4, 0.6), 1.0, tol=1e-9)


def test_regression_pulls_to_500():
    # Small sample 1.000 win% regresses well below 1.0 with k=30.
    assert regress_winpct(1.0, games=2, k=30) < 0.6
    # Large sample barely moves.
    assert regress_winpct(0.6, games=600, k=30) > 0.59


def _mk_team(team, winpct, games, wrc, bp_fip, pf=100.0):
    return TeamProfile(team=team, raw_winpct=winpct, games=games, wins=winpct * games,
                       losses=(1 - winpct) * games, offense_wrc_plus=wrc, bullpen_fip=bp_fip, park_factor=pf)


def _mk_pitcher(name, xfip, ip=80.0):
    return PitcherProfile(player_id=1, name=name, ip=ip, xfip=xfip, k_bb_pct=0.18,
                          csw_pct=0.30, recent_xwoba_con=0.330, recent_starts=5,
                          has_season_stats=True, has_statcast=True)


def test_model_favors_better_pitcher_and_breaks_down():
    home_t = _mk_team("Home City Homers", 0.52, 60, 105, 3.8, pf=100)
    away_t = _mk_team("Away Town Aways", 0.50, 60, 100, 4.2, pf=100)
    ace = _mk_pitcher("Ace", 3.10)        # home ace
    scrub = _mk_pitcher("Scrub", 4.80)    # away weak starter
    res = compute_win_probability(home_t, away_t, ace, scrub)

    # Components present and named. `bullpen_fatigue` and `lineup` were
    # added 2026-05-28; both contribute 0 when no status is supplied (this
    # test's path), so the count grows but no math drifts.
    names = {c.name for c in res.components}
    assert names == {"starter", "bullpen", "bullpen_fatigue", "lineup",
                     "park", "home_field", "form"}
    # Big pitching edge + HFA -> home clearly favored.
    assert res.home_win_prob > res.base_prob
    assert res.home_win_prob > 0.55
    # Starter component must be positive (favors home) and dominate.
    starter = next(c for c in res.components if c.name == "starter")
    assert starter.weighted_delta > 0
    # Probabilities are valid and complementary.
    assert approx(res.home_win_prob + res.away_win_prob, 1.0)


def test_golden_win_probability_locks_current_math():
    """Golden snapshot of compute_win_probability on one fixed game, so any change
    to the model formulas surfaces as an explicit diff here. When a weight/formula
    change is INTENTIONAL, update these expected numbers deliberately (and note why).

    Locked 2026-05-27, before the prediction-engine improvements (blend, park HFA,
    form regression, pitcher-volatility tuning).
    """
    ht = TeamProfile(team="H", raw_winpct=0.55, games=80, wins=44, losses=36,
                     offense_wrc_plus=110, bullpen_fip=3.80, park_factor=104)
    at = TeamProfile(team="A", raw_winpct=0.48, games=80, wins=38.4, losses=41.6,
                     offense_wrc_plus=95, bullpen_fip=4.30, park_factor=100)
    hp = PitcherProfile(player_id=1, name="Home SP", ip=90, xfip=3.40, k_bb_pct=0.18,
                        csw_pct=0.30, recent_xwoba_con=0.320, recent_starts=5,
                        has_season_stats=True, has_statcast=True)
    ap = PitcherProfile(player_id=2, name="Away SP", ip=85, xfip=4.20, k_bb_pct=0.14,
                        csw_pct=0.28, recent_xwoba_con=0.360, recent_starts=5,
                        has_season_stats=True, has_statcast=True)
    res = compute_win_probability(ht, at, hp, ap)

    assert approx(res.base_prob, 0.550802, tol=1e-5), res.base_prob
    # home_win_prob shifted by -0.005 on 2026-05-30 when the form normal cap
    # tightened from +/-5% to +/-1.5% (Step 1). Raw form diff is still 0.020
    # (legacy single-window path with recent=0.360 vs 0.320), but it now
    # clamps to 0.015. Everything else is unchanged.
    assert approx(res.home_win_prob, 0.654446, tol=1e-5), res.home_win_prob
    deltas = {c.name: c.weighted_delta for c in res.components}
    assert approx(deltas["starter"], 0.051895, tol=1e-5), deltas["starter"]
    assert approx(deltas["bullpen"], 0.011000, tol=1e-5), deltas["bullpen"]
    assert approx(deltas["park"], 0.000750, tol=1e-5), deltas["park"]
    # home_field is now the park-specific DEFAULT (team "H" isn't in park_hfa);
    # changed 0.035 -> 0.025 on 2026-05-27 with park-specific HFA.
    assert approx(deltas["home_field"], 0.025000, tol=1e-5), deltas["home_field"]
    # Form: legacy path produces diff 0.020, clamped to new 1.5% cap.
    assert approx(deltas["form"], 0.015000, tol=1e-5), deltas["form"]
    # Form is fragile under the legacy path (no per-window validation possible).
    form_comp = next(c for c in res.components if c.name == "form")
    assert form_comp.fragile is True


def test_park_specific_home_field_advantage():
    """Home-field is park-specific: Coors (Rockies) > default; unlisted -> default."""
    at = TeamProfile(team="A", raw_winpct=0.50, games=80, wins=40, losses=40,
                     offense_wrc_plus=100, bullpen_fip=4.00, park_factor=100)
    hp = PitcherProfile(player_id=1, name="H SP", ip=80, xfip=4.0, k_bb_pct=0.16,
                        csw_pct=0.29, recent_xwoba_con=0.330, recent_starts=5,
                        has_season_stats=True, has_statcast=True)
    ap = PitcherProfile(player_id=2, name="A SP", ip=80, xfip=4.0, k_bb_pct=0.16,
                        csw_pct=0.29, recent_xwoba_con=0.330, recent_starts=5,
                        has_season_stats=True, has_statcast=True)

    def hfa_for(home_name: str) -> float:
        ht = TeamProfile(team=home_name, raw_winpct=0.50, games=80, wins=40, losses=40,
                         offense_wrc_plus=100, bullpen_fip=4.00, park_factor=100)
        res = compute_win_probability(ht, at, hp, ap)
        return next(c.weighted_delta for c in res.components if c.name == "home_field")

    assert approx(hfa_for("Colorado Rockies"), 0.045, tol=1e-9)   # Coors
    assert approx(hfa_for("Some Unlisted Team"), 0.025, tol=1e-9)  # default


def test_recent_form_regresses_toward_season():
    """Recent xwOBAcon is regressed toward the season value (0.6/0.4), damping
    the delta vs using raw recent form."""
    from mlb_value_bot.analysis.win_probability import _form_delta

    home = PitcherProfile(player_id=1, name="H", recent_xwoba_con=0.300, xwoba_con=0.340, recent_starts=5)
    away = PitcherProfile(player_id=2, name="A", recent_xwoba_con=0.360, xwoba_con=0.340, recent_starts=5)
    # 0.6 recent / 0.4 season -> H=0.316, A=0.352, diff 0.036, *0.5 scale = 0.018.
    # New cap (2026-05-30) is +/-1.5% on the legacy path -> clamped to 0.015.
    # Legacy path always sets fragile=True (no per-window validation possible).
    delta, _, avail, fragile = _form_delta(home, away, scale=0.5, recent_weight=0.6)
    assert avail and fragile
    assert approx(delta, 0.015, tol=1e-6), delta
    # Same cap applies regardless of recent_weight (it's a hard ceiling now,
    # not a value that scales with the input).
    raw, _, _, _ = _form_delta(home, away, scale=0.5, recent_weight=1.0)
    assert approx(raw, 0.015, tol=1e-6)
    # No season value -> falls back to recent only -> still clamped to 0.015.
    h2 = PitcherProfile(player_id=3, name="H2", recent_xwoba_con=0.300, recent_starts=5)
    a2 = PitcherProfile(player_id=4, name="A2", recent_xwoba_con=0.360, recent_starts=5)
    d2, _, ok2, frag2 = _form_delta(h2, a2, scale=0.5, recent_weight=0.6)
    assert ok2 and approx(d2, 0.015, tol=1e-6) and frag2


def test_starter_delta_is_clamped():
    """The starter component (previously unbounded) is capped at +/- starter_clamp."""
    def starter(hx: float, ax: float) -> float:
        ht = TeamProfile(team="H", raw_winpct=0.5, games=80, wins=40, losses=40,
                         offense_wrc_plus=100, bullpen_fip=4.0, park_factor=100)
        at = TeamProfile(team="A", raw_winpct=0.5, games=80, wins=40, losses=40,
                         offense_wrc_plus=100, bullpen_fip=4.0, park_factor=100)
        hp = PitcherProfile(player_id=1, name="H", xfip=hx, recent_starts=5, has_season_stats=True)
        ap = PitcherProfile(player_id=2, name="A", xfip=ax, recent_starts=5, has_season_stats=True)
        res = compute_win_probability(ht, at, hp, ap)
        return next(c.weighted_delta for c in res.components if c.name == "starter")

    assert approx(starter(2.0, 7.0), 0.15, tol=1e-6)    # elite vs awful -> capped
    assert approx(starter(7.0, 2.0), -0.15, tol=1e-6)   # symmetric
    assert abs(starter(3.4, 4.2)) < 0.15                # normal matchup untouched


def test_confidence_drops_with_missing_data():
    home_t = _mk_team("H", 0.5, 60, 100, 4.0)
    away_t = _mk_team("A", 0.5, 60, 100, 4.0)
    full_h = _mk_pitcher("FullH", 3.5)
    full_a = _mk_pitcher("FullA", 4.5)
    res_full = compute_win_probability(home_t, away_t, full_h, full_a)
    conf_full = compute_confidence(res_full, full_h, full_a, home_t, away_t, recommended_ev=0.08)

    empty_h = PitcherProfile(player_id=None, name=None)
    empty_a = PitcherProfile(player_id=None, name=None)
    res_empty = compute_win_probability(home_t, away_t, empty_h, empty_a)
    conf_empty = compute_confidence(res_empty, empty_h, empty_a, home_t, away_t, recommended_ev=0.08)

    assert conf_full > conf_empty
    assert 0 <= conf_empty <= 100 and 0 <= conf_full <= 100


# --- Tracking DB (uses a temp DB via monkeypatching the module path) ---------
def test_tracking_roundtrip(tmp_path=None):
    import importlib
    import mlb_value_bot.utils as utils
    from pathlib import Path
    import tempfile

    # Redirect the DB to a throwaway file so we never touch the real one.
    tmpdir = Path(tempfile.mkdtemp())
    orig_db = utils.DB_PATH
    utils.DB_PATH = tmpdir / "test.db"
    recs = importlib.reload(importlib.import_module("mlb_value_bot.tracking.recommendations"))
    try:
        rec = recs.RecommendationRecord(
            date="2024-04-01", game_id=999, home_team="Home City Homers", away_team="Away Town Aways",
            recommended_side="home", model_prob=0.58, market_prob_devigged=0.52,
            american_odds=-110, decimal_odds=ev.american_to_decimal(-110), ev_pct=0.06,
            kelly_stake=0.012, confidence=72.0, reasoning={"x": 1},
        )
        rid = recs.upsert_recommendation(rec)
        assert rid > 0
        # Re-upsert with a different price -> sets closing line + CLV.
        rec2 = recs.RecommendationRecord(**{**rec.__dict__})
        rec2.american_odds = -130  # line shortened toward our side
        recs.upsert_recommendation(rec2)
        rows = recs.get_for_date("2024-04-01")
        assert len(rows) == 1
        assert rows[0]["closing_line"] == -130
        assert rows[0]["clv_pct"] is not None and rows[0]["clv_pct"] > 0  # we beat the close
    finally:
        utils.DB_PATH = orig_db
        importlib.reload(recs)


# --- Odds <-> schedule matching (date-aware, series-safe) --------------------
def test_match_odds_picks_correct_day_in_a_series():
    from types import SimpleNamespace
    from mlb_value_bot.pipeline import _match_odds
    from mlb_value_bot.data.odds_client import GameOdds

    sched = SimpleNamespace(
        home_team="San Diego Padres", away_team="Philadelphia Phillies", game_date="2026-05-27"
    )
    # Same matchup on consecutive days (a series). Padres are PT, so a 5/27 night
    # game starts early on 5/28 UTC — the matcher must still bucket it to 5/27.
    tonight = GameOdds(event_id="tonight", commence_time="2026-05-28T01:40:00Z",
                       home_team="San Diego Padres", away_team="Philadelphia Phillies")
    tomorrow = GameOdds(event_id="tomorrow", commence_time="2026-05-29T01:40:00Z",
                        home_team="San Diego Padres", away_team="Philadelphia Phillies")

    # Order-independent: always returns the game on the analyzed date.
    assert _match_odds(sched, [tomorrow, tonight]).event_id == "tonight"
    assert _match_odds(sched, [tonight, tomorrow]).event_id == "tonight"
    # If only a different day's odds exist, skip rather than show a wrong line.
    assert _match_odds(sched, [tomorrow]) is None
    # No matchup at all -> None.
    assert _match_odds(sched, []) is None


# --- Projected score (run-environment display, 2026-05-28) -----------------
def test_projected_score_basic():
    """Ace pitcher reduces opposing team's projected runs."""
    from mlb_value_bot.analysis.run_environment import projected_score

    ht = _mk_team("H", 0.5, 60, 100, 4.0, pf=100)   # avg offense, avg bullpen, avg park
    at = _mk_team("A", 0.5, 60, 100, 4.0, pf=100)   # same
    cfg = {"league": {"runs_per_game": 4.5, "avg_xfip": 4.0}, "model": {}}

    # Both pitchers league average -> ~4.5 each (regression to mean).
    even_h = _mk_pitcher("H_avg", 4.0)
    even_a = _mk_pitcher("A_avg", 4.0)
    ps = projected_score(ht, at, even_h, even_a, cfg)
    assert ps.available
    assert approx(ps.home_runs, 4.5, tol=0.5)
    assert approx(ps.away_runs, 4.5, tol=0.5)

    # Home faces an ace (away SP rate 3.0) -> home RS drops.
    ace_a = _mk_pitcher("A_ace", 3.0)
    ps_vs_ace = projected_score(ht, at, even_h, ace_a, cfg)
    assert ps_vs_ace.home_runs < ps.home_runs
    # Away faces an average SP (home), so its RS unchanged.
    assert approx(ps_vs_ace.away_runs, ps.away_runs, tol=1e-6)


def test_projected_score_park_factor_lifts_both_teams():
    """Coors-like park (factor 112) bumps RS for both teams."""
    from mlb_value_bot.analysis.run_environment import projected_score

    neutral_h = _mk_team("H", 0.5, 60, 100, 4.0, pf=100)
    coors_h = _mk_team("H", 0.5, 60, 100, 4.0, pf=112)
    at = _mk_team("A", 0.5, 60, 100, 4.0, pf=100)
    even_h = _mk_pitcher("H_avg", 4.0)
    even_a = _mk_pitcher("A_avg", 4.0)
    cfg = {"league": {"runs_per_game": 4.5, "avg_xfip": 4.0}, "model": {}}

    neutral = projected_score(neutral_h, at, even_h, even_a, cfg)
    coors = projected_score(coors_h, at, even_h, even_a, cfg)
    assert coors.home_runs > neutral.home_runs
    assert coors.away_runs > neutral.away_runs


def test_projected_score_degrades_when_inputs_missing():
    """Missing offense / pitcher rate -> available=False, no crash."""
    from mlb_value_bot.analysis.run_environment import projected_score
    from mlb_value_bot.analysis.team_metrics import TeamProfile

    no_offense = TeamProfile(team="H", raw_winpct=0.5, games=60,
                             offense_wrc_plus=None, bullpen_fip=4.0, park_factor=100)
    at = _mk_team("A", 0.5, 60, 100, 4.0, pf=100)
    even_h = _mk_pitcher("H_avg", 4.0)
    even_a = _mk_pitcher("A_avg", 4.0)
    cfg = {"league": {"runs_per_game": 4.5, "avg_xfip": 4.0}, "model": {}}
    ps = projected_score(no_offense, at, even_h, even_a, cfg)
    assert not ps.available
    assert ps.home_runs is None and ps.away_runs is None


# --- Sharp/square market intel (2026-05-28) ---------------------------------
def test_market_intel_sharp_minus_square_signed():
    """sharp_minus_square is + when sharps' home prob > squares' home prob."""
    from mlb_value_bot.data.market_intel import compute_market_intel
    from mlb_value_bot.data.odds_client import GameOdds
    # Square books price home as ~63% (DK -175 / +144 fair ~63%). Sharps
    # price home at ~62% (-162 / +149). So sharp_minus_square should be NEG.
    odds = GameOdds(
        event_id="e1", commence_time="2026-05-28T22:41:00Z",
        home_team="Pittsburgh Pirates", away_team="Chicago Cubs",
        all_books=[
            {"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Pittsburgh Pirates", "price": -175},
                {"name": "Chicago Cubs", "price": 144},
            ]}]},
            {"key": "fanduel", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Pittsburgh Pirates", "price": -174},
                {"name": "Chicago Cubs", "price": 146},
            ]}]},
            {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Pittsburgh Pirates", "price": -162},
                {"name": "Chicago Cubs", "price": 149},
            ]}]},
            {"key": "lowvig", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Pittsburgh Pirates", "price": -163},
                {"name": "Chicago Cubs", "price": 147},
            ]}]},
        ],
    )
    mi = compute_market_intel(
        odds,
        sharp_books=["pinnacle", "lowvig"],
        square_books=["draftkings", "fanduel"],
    )
    assert mi.available
    assert mi.n_sharp_books == 2 and mi.n_square_books == 2 and mi.n_total_books == 4
    # Sharps less bullish on home -> sharp - square is negative.
    assert mi.sharp_minus_square is not None and mi.sharp_minus_square < 0
    # Sharp consensus around 0.61 (Pinnacle -162), square around 0.62 (DK -175).
    # Tight bounds without locking the exact devig math.
    assert 0.58 < mi.sharp_devig_home < 0.64
    assert 0.60 < mi.square_devig_home < 0.66
    assert mi.square_devig_home > mi.sharp_devig_home  # squares more bullish here
    assert mi.dispersion_pp is not None and mi.dispersion_pp > 0


def test_market_intel_unavailable_when_no_sharp_books():
    """No sharp books quoted -> available=False, model degrades cleanly."""
    from mlb_value_bot.data.market_intel import compute_market_intel
    from mlb_value_bot.data.odds_client import GameOdds
    odds = GameOdds(
        event_id="e1", commence_time="2026-05-28T22:41:00Z",
        home_team="Pirates", away_team="Cubs",
        all_books=[{"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Pirates", "price": -175}, {"name": "Cubs", "price": 144},
        ]}]}],
    )
    mi = compute_market_intel(odds, sharp_books=["pinnacle"], square_books=["draftkings"])
    assert not mi.available
    assert mi.sharp_devig_home is None
    assert mi.disagreement_with(0.5) is None


def test_market_intel_disagreement_signed():
    """disagreement_with returns positive when our blended is HIGHER than sharps."""
    from mlb_value_bot.data.market_intel import MarketIntelligence
    mi = MarketIntelligence(
        sharp_devig_home=0.55, square_devig_home=0.58,
        sharp_minus_square=-0.03, dispersion_pp=0.5,
        n_sharp_books=2, n_square_books=2, n_total_books=4,
    )
    # We say home 0.65, sharps say 0.55 -> we're fading by +0.10.
    assert approx(mi.disagreement_with(0.65), 0.10, tol=1e-9)


# --- Edge stability classification (Step 3, 2026-05-30) ---------------------
def _mk_component(name, weighted, available=True, fragile=False):
    """Test-only constructor: mirrors win_probability._mk shape."""
    from mlb_value_bot.analysis.win_probability import Component
    return Component(name=name, raw_delta=weighted, weight=1.0,
                     weighted_delta=weighted, note="", available=available, fragile=fragile)


def test_stability_stable_when_pitcher_and_bullpen_drive():
    """Pick driven mostly by starter + bullpen (stable components) -> STABLE."""
    from mlb_value_bot.analysis.stability import classify_edge_stability
    from mlb_value_bot.data.lineup_status import LineupStatus, STATUS_CONFIRMED
    components = [
        _mk_component("starter", 0.040),      # pushes home, stable
        _mk_component("bullpen", 0.020),      # pushes home, stable
        _mk_component("home_field", 0.025),   # neutral
        _mk_component("form", 0.005),         # tiny push, not fragile
        _mk_component("park", 0.000),
    ]
    confirmed = LineupStatus(team="X", status=STATUS_CONFIRMED, key_bats_total=3, key_bats_present=3)
    s = classify_edge_stability(components, "home", sharp_fade_pp=None,
                                home_lineup_status=confirmed, away_lineup_status=confirmed)
    assert s.label == "stable"
    # starter + bullpen / total positive drive >= 60%.
    assert s.stable_share >= 0.60
    assert s.fragile_share < 0.10
    assert not s.hard_fragile_signals


def test_stability_fragile_when_form_fragile_dominates():
    """Pick driven mostly by a fragile-flagged form component -> FRAGILE."""
    from mlb_value_bot.analysis.stability import classify_edge_stability
    from mlb_value_bot.data.lineup_status import LineupStatus, STATUS_CONFIRMED
    components = [
        _mk_component("starter", 0.002),                # tiny stable contribution
        _mk_component("form", 0.030, fragile=True),     # fragile-flagged, ~71% of drive
        _mk_component("home_field", 0.010),             # neutral
    ]
    confirmed = LineupStatus(team="X", status=STATUS_CONFIRMED, key_bats_total=3, key_bats_present=3)
    s = classify_edge_stability(components, "home", sharp_fade_pp=None,
                                home_lineup_status=confirmed, away_lineup_status=confirmed)
    # fragile_share = 0.030 / 0.042 ~= 0.71 >= 0.50 threshold.
    assert s.label == "fragile"
    assert s.fragile_share >= 0.50


def test_stability_fragile_on_hard_signal_projected_lineup():
    """Even a stable-driver pick is FRAGILE when a lineup is projected."""
    from mlb_value_bot.analysis.stability import classify_edge_stability
    from mlb_value_bot.data.lineup_status import LineupStatus, STATUS_CONFIRMED, STATUS_PROJECTED
    components = [
        _mk_component("starter", 0.040),
        _mk_component("bullpen", 0.020),
        _mk_component("home_field", 0.025),
    ]
    confirmed = LineupStatus(team="H", status=STATUS_CONFIRMED, key_bats_total=3, key_bats_present=3)
    projected = LineupStatus(team="A", status=STATUS_PROJECTED, key_bats_total=3)
    s = classify_edge_stability(components, "home", sharp_fade_pp=None,
                                home_lineup_status=confirmed, away_lineup_status=projected)
    assert s.label == "fragile"
    assert any("projected" in sig for sig in s.hard_fragile_signals)


def test_stability_fragile_on_sharp_fade_hard_signal():
    """A 2pp+ sharp fade on the pick side flags fragile regardless of drivers."""
    from mlb_value_bot.analysis.stability import classify_edge_stability
    from mlb_value_bot.data.lineup_status import LineupStatus, STATUS_CONFIRMED
    components = [
        _mk_component("starter", 0.040),
        _mk_component("bullpen", 0.020),
    ]
    confirmed = LineupStatus(team="X", status=STATUS_CONFIRMED, key_bats_total=3, key_bats_present=3)
    s = classify_edge_stability(components, "home", sharp_fade_pp=0.030,  # 3pp fade
                                home_lineup_status=confirmed, away_lineup_status=confirmed)
    assert s.label == "fragile"
    assert any("sharps" in sig for sig in s.hard_fragile_signals)


def test_stability_moderate_when_neither_dominant():
    """Mixed stable + neutral drive, no fragile signals, but stable_share < 60%."""
    from mlb_value_bot.analysis.stability import classify_edge_stability
    from mlb_value_bot.data.lineup_status import LineupStatus, STATUS_CONFIRMED
    # Stable drivers only contribute 35% of positive drive; rest is home_field
    # (neutral). No fragile signals -> MODERATE.
    components = [
        _mk_component("starter", 0.010),
        _mk_component("bullpen", 0.005),
        _mk_component("home_field", 0.040),    # neutral driver
        _mk_component("park", 0.010),          # neutral
    ]
    confirmed = LineupStatus(team="X", status=STATUS_CONFIRMED, key_bats_total=3, key_bats_present=3)
    s = classify_edge_stability(components, "home", sharp_fade_pp=None,
                                home_lineup_status=confirmed, away_lineup_status=confirmed)
    assert s.label == "moderate"
    assert s.stable_share < 0.60
    assert not s.hard_fragile_signals


def test_stability_drivers_sorted_for_ui():
    """The `drivers` field is sorted by absolute aligned contribution desc."""
    from mlb_value_bot.analysis.stability import classify_edge_stability
    from mlb_value_bot.data.lineup_status import LineupStatus, STATUS_CONFIRMED
    components = [
        _mk_component("starter", 0.030),
        _mk_component("home_field", 0.025),
        _mk_component("form", 0.010),
    ]
    confirmed = LineupStatus(team="X", status=STATUS_CONFIRMED, key_bats_total=3, key_bats_present=3)
    s = classify_edge_stability(components, "home", sharp_fade_pp=None,
                                home_lineup_status=confirmed, away_lineup_status=confirmed)
    names = [d["name"] for d in s.drivers]
    assert names == ["starter", "home_field", "form"]


# --- In-progress games are not bettable (2026-05-28) -----------------------
def test_in_progress_games_are_not_playable():
    """A game whose detailedState is 'In Progress' (or Final, etc.) must NOT
    be treated as playable -- live moneyline prices reflect the in-game
    score, not the pre-game matchup, and the model is pre-game only.
    """
    from mlb_value_bot.data.mlb_client import ScheduledGame, ProbablePitcher

    def _mk(status):
        return ScheduledGame(
            game_id=1, game_date="2026-05-28", status=status,
            home_team="H", away_team="A",
            home_pitcher=ProbablePitcher(player_id=None, name=None),
            away_pitcher=ProbablePitcher(player_id=None, name=None),
        )

    # Bettable: hasn't started yet.
    assert _mk("Scheduled").is_playable
    assert _mk("Pre-Game").is_playable
    assert _mk("Warmup").is_playable
    # NOT bettable: in progress -- live prices reflect score, not matchup.
    assert not _mk("In Progress").is_playable
    assert not _mk("Manager Challenge").is_playable
    assert not _mk("Delayed").is_playable
    # NOT bettable: already finished.
    assert not _mk("Final").is_playable
    assert not _mk("Game Over").is_playable
    assert not _mk("Completed Early").is_playable
    # NOT bettable: won't happen today.
    assert not _mk("Postponed").is_playable
    assert not _mk("Cancelled").is_playable
    assert not _mk("Suspended").is_playable


# --- Model/market divergence guard (2026-05-28) -----------------------------
def test_evaluate_game_skips_on_extreme_model_market_divergence():
    """A raw model vs market gap > max_model_market_divergence -> skip.

    Catches the "late-scratch fake +EV" failure mode we hit in production:
    market moved from -149 to +295 on a bullpen-game scratch the model didn't
    know about; the mid blend produced +12.6% fake EV. The new guard refuses
    to bet whenever the model and market disagree more than the configured
    threshold, on the principle that the market knows something we don't.
    """
    from types import SimpleNamespace
    from mlb_value_bot.pipeline import evaluate_game
    from mlb_value_bot.data.mlb_client import ScheduledGame, ProbablePitcher
    from mlb_value_bot.data.odds_client import GameOdds, SidePrice
    from mlb_value_bot.analysis.team_metrics import TeamMetricsProvider
    from datetime import date

    # Build a config where the model produces a HIGH home win prob (good
    # pitcher, good lineup) but the market prices the home team as a heavy
    # underdog -- the divergence guard should fire.
    sched = ScheduledGame(
        game_id=1, game_date="2026-05-28", status="Scheduled",
        home_team="Home", away_team="Away",
        home_pitcher=ProbablePitcher(player_id=None, name=None),
        away_pitcher=ProbablePitcher(player_id=None, name=None),
        game_datetime=None,
    )
    odds = GameOdds(
        event_id="e1", commence_time="2026-05-28T20:00:00Z",
        home_team="Home", away_team="Away",
        home=SidePrice(team="Home", american_odds=400, bookmaker="dk"),   # market: home is heavy underdog
        away=SidePrice(team="Away", american_odds=-500, bookmaker="dk"),
    )

    # Stub the team metrics so the model produces a roughly 50/50 prob.
    class _Stub:
        def build_team_profile(self, name, is_home):
            from mlb_value_bot.analysis.team_metrics import TeamProfile
            return TeamProfile(team=name, raw_winpct=0.55, games=60, wins=33, losses=27,
                              offense_wrc_plus=100, bullpen_fip=4.0, park_factor=100)

    result = evaluate_game(sched, odds, _Stub(), 2026, date(2026, 5, 28))
    assert result.skipped_reason is not None
    assert "diverge" in result.skipped_reason


# --- Multi-window recent form (2026-05-28) ----------------------------------
def test_blended_form_uses_all_windows_when_present():
    """All three windows present + above sample-size floor -> weighted blend."""
    from mlb_value_bot.analysis.win_probability import _blended_form_xwoba
    p = PitcherProfile(player_id=1, name="P",
                       recent_xwoba_con_14d=0.300, recent_bip_14d=40,
                       recent_xwoba_con_30d=0.320, recent_bip_30d=80,
                       xwoba_con=0.340)
    cfg = {"model": {"form_windows": {"d14": 0.5, "d30": 0.3, "season": 0.2},
                     "form_min_bip": {"d14": 25, "d30": 60}}}
    fb = _blended_form_xwoba(p, cfg)
    # 0.5*0.300 + 0.3*0.320 + 0.2*0.340 = 0.150 + 0.096 + 0.068 = 0.314
    assert approx(fb.blended, 0.314, tol=1e-6)
    assert "14d=" in fb.note and "30d=" in fb.note and "season=" in fb.note
    # All three windows present; 14d carries exactly its configured 50% share.
    assert approx(fb.w14_share, 0.5, tol=1e-6)


def test_blended_form_drops_undersample_windows_and_renormalizes():
    """A window below its BIP floor is dropped; remaining weights renormalize."""
    from mlb_value_bot.analysis.win_probability import _blended_form_xwoba
    p = PitcherProfile(player_id=1, name="P",
                       recent_xwoba_con_14d=0.300, recent_bip_14d=5,   # below 25 floor
                       recent_xwoba_con_30d=0.320, recent_bip_30d=80,
                       xwoba_con=0.340)
    cfg = {"model": {"form_windows": {"d14": 0.5, "d30": 0.3, "season": 0.2},
                     "form_min_bip": {"d14": 25, "d30": 60}}}
    fb = _blended_form_xwoba(p, cfg)
    # 14d dropped; 30d (0.3) + season (0.2) renormalize to (0.6, 0.4):
    # 0.6*0.320 + 0.4*0.340 = 0.192 + 0.136 = 0.328
    assert approx(fb.blended, 0.328, tol=1e-6)
    # 14d was dropped under its sample floor -> 0% share.
    assert fb.w14_share == 0.0
    assert "14d" not in fb.note  # 14d not in label since it was dropped


def test_blended_form_returns_none_when_nothing_available():
    from mlb_value_bot.analysis.win_probability import _blended_form_xwoba
    p = PitcherProfile(player_id=1, name="P")  # all None
    cfg = {"model": {"form_windows": {"d14": 0.5, "d30": 0.3, "season": 0.2},
                     "form_min_bip": {"d14": 25, "d30": 60}}}
    assert _blended_form_xwoba(p, cfg) is None


def test_form_delta_prefers_multi_window_when_available():
    """When the new per-window fields are present, _form_delta uses them
    instead of the legacy recent_xwoba_con + season blend."""
    from mlb_value_bot.analysis.win_probability import _form_delta
    home = PitcherProfile(player_id=1, name="H",
                          recent_xwoba_con=0.999,             # legacy field (would dominate)
                          recent_xwoba_con_14d=0.290, recent_bip_14d=40,
                          recent_xwoba_con_30d=0.310, recent_bip_30d=80,
                          xwoba_con=0.330)
    away = PitcherProfile(player_id=2, name="A",
                          recent_xwoba_con=0.001,             # legacy field
                          recent_xwoba_con_14d=0.360, recent_bip_14d=40,
                          recent_xwoba_con_30d=0.345, recent_bip_30d=80,
                          xwoba_con=0.340)
    cfg = {"model": {"form_windows": {"d14": 0.5, "d30": 0.3, "season": 0.2},
                     "form_min_bip": {"d14": 25, "d30": 60}}}
    delta, note, avail, _fragile = _form_delta(home, away, scale=0.5, config=cfg)
    # Multi-window blend favors HOME (lower xwOBA). Legacy fields would flip it.
    assert avail
    assert delta > 0
    assert "xwOBAcon H" in note and "14d=0.290" in note


def test_form_normal_cap_is_15bp():
    """Without 14d+30d directional agreement OR meaningful sample, the form
    delta is clamped to +/- 1.5% even when the raw diff would be much larger."""
    from mlb_value_bot.analysis.win_probability import _form_delta
    # Inputs that pre-2026-05-30 would have produced ~3% delta (well above the
    # new 1.5% normal cap but under the old 5% cap). Use under-sample 30d so
    # the extreme cap is NOT licensed -- forces normal cap.
    home = PitcherProfile(player_id=1, name="H",
                          recent_xwoba_con_14d=0.250, recent_bip_14d=40,
                          recent_xwoba_con_30d=0.260, recent_bip_30d=50,  # below 70 floor
                          xwoba_con=0.310)
    away = PitcherProfile(player_id=2, name="A",
                          recent_xwoba_con_14d=0.400, recent_bip_14d=40,
                          recent_xwoba_con_30d=0.390, recent_bip_30d=50,  # below 70 floor
                          xwoba_con=0.350)
    cfg = {"model": {"form_windows": {"d14": 0.5, "d30": 0.3, "season": 0.2},
                     "form_min_bip": {"d14": 25, "d30": 30},  # let 30d into blend
                     "form_normal_cap": 0.015, "form_extreme_cap": 0.025,
                     "form_extreme_min_bip_14d": 30, "form_extreme_min_bip_30d": 70,
                     "form_fragile_w14_share": 0.60}}
    delta, note, avail, _ = _form_delta(home, away, scale=0.5, config=cfg)
    assert avail
    assert approx(delta, 0.015, tol=1e-6), delta   # hit normal cap
    assert "normal cap" in note


def test_form_extreme_cap_requires_agreement_and_sample():
    """Extreme cap (+/-2.5%) is allowed only when both 14d and 30d agree
    directionally with the blended diff AND BIP samples are meaningful."""
    from mlb_value_bot.analysis.win_probability import _form_delta
    # Strong + clean signal: home is much better on both windows AND season,
    # both pitchers well over the sample floor.
    home = PitcherProfile(player_id=1, name="H",
                          recent_xwoba_con_14d=0.230, recent_bip_14d=60,
                          recent_xwoba_con_30d=0.240, recent_bip_30d=120,
                          xwoba_con=0.250)
    away = PitcherProfile(player_id=2, name="A",
                          recent_xwoba_con_14d=0.420, recent_bip_14d=60,
                          recent_xwoba_con_30d=0.410, recent_bip_30d=120,
                          xwoba_con=0.400)
    cfg = {"model": {"form_windows": {"d14": 0.5, "d30": 0.3, "season": 0.2},
                     "form_min_bip": {"d14": 25, "d30": 60},
                     "form_normal_cap": 0.015, "form_extreme_cap": 0.025,
                     "form_extreme_min_bip_14d": 30, "form_extreme_min_bip_30d": 70,
                     "form_fragile_w14_share": 0.60}}
    delta, note, avail, _ = _form_delta(home, away, scale=0.5, config=cfg)
    assert avail
    # Raw diff > extreme cap, so clamps to +0.025 (not the +0.015 normal cap).
    assert approx(delta, 0.025, tol=1e-6), delta
    assert "extreme cap" in note


def test_form_fragile_when_14d_dominates():
    """When 30d underweights below its sample floor, 14d carries >= 60% of the
    blend on both pitchers -> the form component is tagged fragile=True."""
    from mlb_value_bot.analysis.win_probability import _form_delta
    home = PitcherProfile(player_id=1, name="H",
                          recent_xwoba_con_14d=0.300, recent_bip_14d=40,
                          recent_xwoba_con_30d=None,  recent_bip_30d=0,  # missing 30d
                          xwoba_con=0.330)
    away = PitcherProfile(player_id=2, name="A",
                          recent_xwoba_con_14d=0.360, recent_bip_14d=40,
                          recent_xwoba_con_30d=None,  recent_bip_30d=0,
                          xwoba_con=0.340)
    cfg = {"model": {"form_windows": {"d14": 0.5, "d30": 0.3, "season": 0.2},
                     "form_min_bip": {"d14": 25, "d30": 60},
                     "form_normal_cap": 0.015, "form_extreme_cap": 0.025,
                     "form_extreme_min_bip_14d": 30, "form_extreme_min_bip_30d": 70,
                     "form_fragile_w14_share": 0.60}}
    _delta, _note, _avail, fragile = _form_delta(home, away, scale=0.5, config=cfg)
    # Only 14d (w=0.5) and season (w=0.2) made it -> 14d is 0.5/0.7 = 71% > 60%.
    assert fragile is True


# --- Lineup confirmation component (2026-05-28) -----------------------------
def test_lineup_status_short_label():
    """short_label() reflects status + key bats present."""
    from mlb_value_bot.data.lineup_status import (
        LineupStatus, STATUS_CONFIRMED, STATUS_PROJECTED, STATUS_UNAVAILABLE,
    )
    s = LineupStatus(team="X", status=STATUS_UNAVAILABLE)
    assert "unavailable" in s.short_label()
    s = LineupStatus(team="X", status=STATUS_PROJECTED)
    assert "projected" in s.short_label()
    s = LineupStatus(team="X", status=STATUS_CONFIRMED,
                     key_bats_total=3, key_bats_present=2)
    assert "2/3" in s.short_label()


def test_lineup_delta_only_when_both_confirmed():
    """The lineup component is signed by missing-bats diff and clamped tight.

    + favors home when AWAY has more key bats missing.
    """
    from mlb_value_bot.analysis.win_probability import compute_win_probability
    from mlb_value_bot.data.lineup_status import LineupStatus, STATUS_CONFIRMED, STATUS_PROJECTED
    ht = _mk_team("H", 0.5, 60, 100, 4.0)
    at = _mk_team("A", 0.5, 60, 100, 4.0)
    hp = _mk_pitcher("H SP", 4.0)
    ap = _mk_pitcher("A SP", 4.0)

    # Home all in (0 missing); Away 2 of 3 key bats missing -> tilts toward home.
    home_lu = LineupStatus(team="H", status=STATUS_CONFIRMED,
                           key_bats_total=3, key_bats_present=3)
    away_lu = LineupStatus(team="A", status=STATUS_CONFIRMED,
                           key_bats_total=3, key_bats_present=1,
                           missing_key_bats=["Star1", "Star2"])
    res = compute_win_probability(ht, at, hp, ap,
                                  home_lineup_status=home_lu,
                                  away_lineup_status=away_lu)
    lu = next(c for c in res.components if c.name == "lineup")
    assert lu.available
    # (2 - 0) * 0.005 = 0.010, within +/- 0.02 clamp.
    assert approx(lu.raw_delta, 0.010, tol=1e-6)

    # Projected on either side -> component contributes 0.
    proj_lu = LineupStatus(team="A", status=STATUS_PROJECTED, key_bats_total=3)
    res2 = compute_win_probability(ht, at, hp, ap,
                                   home_lineup_status=home_lu,
                                   away_lineup_status=proj_lu)
    lu2 = next(c for c in res2.components if c.name == "lineup")
    assert not lu2.available
    assert lu2.raw_delta == 0.0


def test_lineup_penalty_drops_data_confidence():
    """A non-zero lineup_penalty subtracts confidence points (data + full)."""
    from mlb_value_bot.analysis.win_probability import (
        compute_data_confidence, compute_confidence,
    )
    ht = _mk_team("H", 0.5, 60, 100, 4.0)
    at = _mk_team("A", 0.5, 60, 100, 4.0)
    hp = _mk_pitcher("H SP", 4.0)
    ap = _mk_pitcher("A SP", 4.0)
    res = compute_win_probability(ht, at, hp, ap)

    base = compute_data_confidence(res, hp, ap, ht, at)
    penalized = compute_data_confidence(res, hp, ap, ht, at, lineup_penalty=8.0)
    assert penalized == round(max(0.0, base - 8.0), 1)

    base_full = compute_confidence(res, hp, ap, ht, at, recommended_ev=0.05)
    pen_full = compute_confidence(res, hp, ap, ht, at, recommended_ev=0.05, lineup_penalty=8.0)
    assert pen_full == round(max(0.0, base_full - 8.0), 1)


# --- Bullpen fatigue component (2026-05-28) ---------------------------------
def test_reliever_usage_unavailable_rules():
    """Each fatigue rule independently flags a reliever as unavailable."""
    from mlb_value_bot.data.bullpen_status import RelieverUsage
    # Heavy pitch count yesterday alone -> unavailable.
    r = RelieverUsage(player_id=1, name="A", pitches_by_day=[40, 0, 0])
    assert r.is_unavailable(pitch_threshold=35, appearance_threshold=3)
    # Back-to-back (yesterday + day before) alone -> unavailable.
    r = RelieverUsage(player_id=2, name="B", pitches_by_day=[15, 12, 0], consecutive_days=2)
    assert r.is_unavailable(35, 3)
    # 3 appearances in 3 days alone -> unavailable.
    r = RelieverUsage(player_id=3, name="C", pitches_by_day=[10, 10, 10], appearances_3d=3)
    assert r.is_unavailable(35, 3)
    # Single light outing two days ago -> available.
    r = RelieverUsage(player_id=4, name="D", pitches_by_day=[0, 12, 0], appearances_3d=1)
    assert not r.is_unavailable(35, 3)


def test_bullpen_status_short_label_and_score():
    """Status surfaces leverage counts and a clean UI label."""
    from mlb_value_bot.data.bullpen_status import BullpenStatus, RelieverUsage
    relievers = [
        RelieverUsage(player_id=1, name="Closer", is_leverage=True, pitches_by_day=[40, 0, 0]),
        RelieverUsage(player_id=2, name="Setup", is_leverage=True),
        RelieverUsage(player_id=3, name="HighLev", is_leverage=True),
        RelieverUsage(player_id=4, name="Mop-up", is_leverage=False),
    ]
    leverage_total = sum(1 for r in relievers if r.is_leverage)
    leverage_unavailable = sum(
        1 for r in relievers
        if r.is_leverage and r.is_unavailable(35, 3)
    )
    s = BullpenStatus(team="X", available=True, relievers=relievers,
                     leverage_total=leverage_total,
                     leverage_unavailable=leverage_unavailable)
    assert s.leverage_total == 3
    assert s.leverage_available == 2
    assert s.fatigue_score == 1
    assert "2/3" in s.short_label()


def test_bullpen_fatigue_tiered_schedule():
    """Per-team penalty uses the explicit non-linear schedule
    [0, -0.5%, -1.5%, -3.0%], net delta = home_pen - away_pen."""
    from mlb_value_bot.analysis.win_probability import compute_win_probability
    from mlb_value_bot.data.bullpen_status import BullpenStatus
    ht = _mk_team("H", 0.5, 60, 100, 4.0)
    at = _mk_team("A", 0.5, 60, 100, 4.0)
    hp = _mk_pitcher("H SP", 4.0)
    ap = _mk_pitcher("A SP", 4.0)
    # Home 0 down (0%), away 2 down (-1.5%) -> net = 0 - (-0.015) = +0.015.
    home_bp = BullpenStatus(team="H", available=True, leverage_total=3, leverage_unavailable=0)
    away_bp = BullpenStatus(team="A", available=True, leverage_total=3, leverage_unavailable=2)
    res = compute_win_probability(ht, at, hp, ap,
                                  home_bullpen_status=home_bp,
                                  away_bullpen_status=away_bp)
    bf = next(c for c in res.components if c.name == "bullpen_fatigue")
    assert bf.available
    assert approx(bf.raw_delta, 0.015, tol=1e-6), bf.raw_delta
    # Sign check: swap roles -> -0.015.
    res2 = compute_win_probability(ht, at, hp, ap,
                                   home_bullpen_status=away_bp,
                                   away_bullpen_status=home_bp)
    bf2 = next(c for c in res2.components if c.name == "bullpen_fatigue")
    assert approx(bf2.raw_delta, -0.015, tol=1e-6)


def test_bullpen_fatigue_max_penalty_3_down():
    """0/3 leverage available on home vs 3/3 on away -> -3.0% on home."""
    from mlb_value_bot.analysis.win_probability import compute_win_probability
    from mlb_value_bot.data.bullpen_status import BullpenStatus
    ht = _mk_team("H", 0.5, 60, 100, 4.0); at = _mk_team("A", 0.5, 60, 100, 4.0)
    hp = _mk_pitcher("H SP", 4.0); ap = _mk_pitcher("A SP", 4.0)
    home_bp = BullpenStatus(team="H", available=True, leverage_total=3, leverage_unavailable=3)
    away_bp = BullpenStatus(team="A", available=True, leverage_total=3, leverage_unavailable=0)
    res = compute_win_probability(ht, at, hp, ap,
                                  home_bullpen_status=home_bp,
                                  away_bullpen_status=away_bp)
    bf = next(c for c in res.components if c.name == "bullpen_fatigue")
    # home_pen = -0.030, away_pen = 0 -> net = -0.030.
    assert approx(bf.raw_delta, -0.030, tol=1e-6)
    # Note format: both per-team values must appear and the right signs.
    assert "-3.0%" in bf.note and "+0.0%" in bf.note


def test_bullpen_fatigue_penalty_for_team_no_leverage_data():
    """leverage_total < 3 -> per-team penalty 0 (schedule needs full 3-arm core)."""
    from mlb_value_bot.analysis.win_probability import _bullpen_penalty_for_team
    from mlb_value_bot.data.bullpen_status import BullpenStatus
    bp = BullpenStatus(team="X", available=True, leverage_total=1, leverage_unavailable=1)
    cfg = {"bullpen_fatigue": {"penalty_by_unavailable": [0.0, -0.005, -0.015, -0.030]}}
    assert _bullpen_penalty_for_team(bp, cfg) == 0.0


def test_bullpen_confidence_penalty_when_feed_unavailable():
    """Per-team confidence reduction when bullpen status is None or
    unavailable. NOT triggered by '3/3 available' (that's real data)."""
    from mlb_value_bot.pipeline import _bullpen_confidence_penalty
    from mlb_value_bot.data.bullpen_status import BullpenStatus
    cfg = {"bullpen_fatigue": {"unavailable_confidence_penalty_per_team": 2.0}}
    real_bp = BullpenStatus(team="H", available=True, leverage_total=3, leverage_unavailable=0)
    unavail_bp = BullpenStatus(team="A", available=False, leverage_total=0)
    # Both real -> 0 penalty.
    assert _bullpen_confidence_penalty(real_bp, real_bp, cfg) == 0.0
    # One unavailable -> 2.0 penalty.
    assert _bullpen_confidence_penalty(real_bp, unavail_bp, cfg) == 2.0
    # Both unavailable -> 4.0 penalty.
    assert _bullpen_confidence_penalty(None, unavail_bp, cfg) == 4.0


def test_bullpen_fatigue_missing_status_degrades_cleanly():
    """No bullpen status -> component contributes 0 with 'unavailable' note."""
    from mlb_value_bot.analysis.win_probability import compute_win_probability
    ht = _mk_team("H", 0.5, 60, 100, 4.0)
    at = _mk_team("A", 0.5, 60, 100, 4.0)
    hp = _mk_pitcher("H SP", 4.0)
    ap = _mk_pitcher("A SP", 4.0)
    res = compute_win_probability(ht, at, hp, ap)  # no bullpen status passed
    bf = next(c for c in res.components if c.name == "bullpen_fatigue")
    assert not bf.available
    assert bf.raw_delta == 0.0
    assert "unavailable" in bf.note


# --- Dynamic blend + bet tiers (2026-05-28) ---------------------------------
def test_resolve_market_blend_tiered():
    """Tiered config picks high/mid/low blend based on data confidence."""
    from mlb_value_bot.analysis.win_probability import resolve_market_blend
    tiered = {
        "market_blend": {
            "high_conf": 0.45, "mid_conf": 0.35, "low_conf": 0.25,
            "high_threshold": 85.0, "mid_threshold": 70.0,
        }
    }
    assert resolve_market_blend(90.0, tiered) == (0.45, "high")
    assert resolve_market_blend(85.0, tiered) == (0.45, "high")
    assert resolve_market_blend(80.0, tiered) == (0.35, "mid")
    assert resolve_market_blend(70.0, tiered) == (0.35, "mid")
    assert resolve_market_blend(50.0, tiered) == (0.25, "low")


def test_resolve_market_blend_scalar_legacy():
    """Scalar config still works -> fixed blend for every game (back-compat)."""
    from mlb_value_bot.analysis.win_probability import resolve_market_blend
    fixed = {"market_blend": 0.35}
    assert resolve_market_blend(95.0, fixed) == (0.35, "fixed")
    assert resolve_market_blend(40.0, fixed) == (0.35, "fixed")


def test_data_confidence_excludes_ev():
    """Data confidence is the non-EV portion of the full confidence score.

    It should be deterministic given the inputs (no EV dependency) and stay
    in [0, 100].
    """
    from mlb_value_bot.analysis.win_probability import compute_data_confidence
    ht = _mk_team("H", 0.5, 60, 100, 4.0)
    at = _mk_team("A", 0.5, 60, 100, 4.0)
    hp = _mk_pitcher("H SP", 4.0)
    ap = _mk_pitcher("A SP", 4.0)
    res = compute_win_probability(ht, at, hp, ap)
    dc1 = compute_data_confidence(res, hp, ap, ht, at)
    # Same inputs -> same output. EV doesn't enter the function signature at all.
    dc2 = compute_data_confidence(res, hp, ap, ht, at)
    assert dc1 == dc2
    assert 0.0 <= dc1 <= 100.0


def test_classify_bet_tier():
    """Pass / Small / Standard / Strong assigned by EV + confidence."""
    from mlb_value_bot.pipeline import _classify_bet_tier
    from mlb_value_bot.utils import load_config
    cfg = load_config()

    # Below EV threshold -> Pass, 0x.
    tier, mult, _ = _classify_bet_tier(0.01, 80.0, cfg)
    assert tier == "pass" and mult == 0.0
    # EV in [threshold, standard_ev) -> Small (halved).
    tier, mult, _ = _classify_bet_tier(0.035, 80.0, cfg)
    assert tier == "small" and mult == 0.5
    # EV high but confidence < min_standard -> Small.
    tier, mult, _ = _classify_bet_tier(0.08, 55.0, cfg)
    assert tier == "small" and mult == 0.5
    # EV in [standard, strong) with adequate confidence -> Standard, 1x.
    tier, mult, _ = _classify_bet_tier(0.06, 70.0, cfg)
    assert tier == "standard" and mult == 1.0
    # EV >= strong AND confidence >= strong -> Strong, 1x (today).
    tier, mult, _ = _classify_bet_tier(0.12, 80.0, cfg)
    assert tier == "strong" and mult == 1.0


# --- Manual runner -----------------------------------------------------------
def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} tests passed.")
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)
