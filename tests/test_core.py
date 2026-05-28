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
    assert approx(res.home_win_prob, 0.659446, tol=1e-5), res.home_win_prob
    deltas = {c.name: c.weighted_delta for c in res.components}
    assert approx(deltas["starter"], 0.051895, tol=1e-5), deltas["starter"]
    assert approx(deltas["bullpen"], 0.011000, tol=1e-5), deltas["bullpen"]
    assert approx(deltas["park"], 0.000750, tol=1e-5), deltas["park"]
    # home_field is now the park-specific DEFAULT (team "H" isn't in park_hfa);
    # changed 0.035 -> 0.025 on 2026-05-27 with park-specific HFA.
    assert approx(deltas["home_field"], 0.025000, tol=1e-5), deltas["home_field"]
    assert approx(deltas["form"], 0.020000, tol=1e-5), deltas["form"]


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
    delta, _, avail = _form_delta(home, away, scale=0.5, recent_weight=0.6)
    assert avail
    assert approx(delta, 0.018, tol=1e-6), delta
    # Raw recent (weight 1.0) would be 0.030 — regression dampens it.
    raw, _, _ = _form_delta(home, away, scale=0.5, recent_weight=1.0)
    assert approx(raw, 0.030, tol=1e-6) and delta < raw
    # No season value -> falls back to recent only (no crash).
    h2 = PitcherProfile(player_id=3, name="H2", recent_xwoba_con=0.300, recent_starts=5)
    a2 = PitcherProfile(player_id=4, name="A2", recent_xwoba_con=0.360, recent_starts=5)
    d2, _, ok2 = _form_delta(h2, a2, scale=0.5, recent_weight=0.6)
    assert ok2 and approx(d2, 0.030, tol=1e-6)


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


def test_bullpen_fatigue_delta_clamped_and_signed():
    """The fatigue delta favors the team whose opponent has more arms down."""
    from mlb_value_bot.analysis.win_probability import compute_win_probability
    from mlb_value_bot.data.bullpen_status import BullpenStatus
    ht = _mk_team("H", 0.5, 60, 100, 4.0)
    at = _mk_team("A", 0.5, 60, 100, 4.0)
    hp = _mk_pitcher("H SP", 4.0)
    ap = _mk_pitcher("A SP", 4.0)
    # Home has 0 leverage down; away has 2 down -> + favors home.
    home_bp = BullpenStatus(team="H", available=True, leverage_total=3, leverage_unavailable=0)
    away_bp = BullpenStatus(team="A", available=True, leverage_total=3, leverage_unavailable=2)
    res = compute_win_probability(ht, at, hp, ap,
                                  home_bullpen_status=home_bp,
                                  away_bullpen_status=away_bp)
    bf = next(c for c in res.components if c.name == "bullpen_fatigue")
    assert bf.available
    # (2 - 0) * 0.012 = 0.024, within +/- 0.03 clamp.
    assert approx(bf.raw_delta, 0.024, tol=1e-6)
    # Sign check: swap roles -> negative.
    res2 = compute_win_probability(ht, at, hp, ap,
                                   home_bullpen_status=away_bp,
                                   away_bullpen_status=home_bp)
    bf2 = next(c for c in res2.components if c.name == "bullpen_fatigue")
    assert approx(bf2.raw_delta, -0.024, tol=1e-6)


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
