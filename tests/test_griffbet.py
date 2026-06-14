"""GriffBet correctness tests — no network, no pybaseball.

The headline guarantee is the GOLDEN EQUIVALENCE: with the neutralization flag
OFF and an identical config, GriffBet's win probability is bit-identical to
BiffBet's. That locks the structural-copy claim so every GriffBet divergence is
attributable to its config/toggle/discipline, not accidental drift.

Run: `python -m mlb_value_bot.tests.test_griffbet` or pytest.
"""
from __future__ import annotations

import importlib
import math
import tempfile
from pathlib import Path

import pandas as pd

from mlb_value_bot.analysis import ev_calculator as ev
from mlb_value_bot.analysis.pitcher_metrics import PitcherProfile
from mlb_value_bot.analysis.team_metrics import TeamProfile
from mlb_value_bot.analysis.win_probability import compute_win_probability
from mlb_value_bot.griffbet import load_griff_config
from mlb_value_bot.griffbet.win_probability import compute_win_probability_griff


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def _mk_team(team, winpct, games, wrc, bp_fip, pf=100.0):
    return TeamProfile(team=team, raw_winpct=winpct, games=games, wins=winpct * games,
                       losses=(1 - winpct) * games, offense_wrc_plus=wrc,
                       bullpen_fip=bp_fip, park_factor=pf)


def _mk_pitcher(name, xfip, ip=80.0):
    return PitcherProfile(player_id=1, name=name, ip=ip, xfip=xfip, k_bb_pct=0.18,
                          csw_pct=0.30, recent_xwoba_con=0.330, recent_starts=5,
                          has_season_stats=True, has_statcast=True)


# --- Golden equivalence ------------------------------------------------------
def test_golden_equivalence_neutralization_off():
    """Flag OFF == BiffBet exactly, across several matchups. Forces the flag OFF
    in a copied config so the invariant holds regardless of the shipped default
    (the golden guarantee is about the OFF code path, not the config value)."""
    cfg = load_griff_config()
    cfg = {**cfg, "model": {**cfg["model"], "starter_neutralized_base": False}}
    cases = [
        (_mk_team("H", 0.60, 50, 105, 3.9, pf=108), _mk_team("A", 0.50, 50, 100, 4.1),
         _mk_pitcher("Ace", 3.2), _mk_pitcher("Mid", 4.3)),
        (_mk_team("H2", 0.45, 40, 95, 4.4), _mk_team("A2", 0.55, 40, 110, 3.7),
         _mk_pitcher("Bad", 5.1), _mk_pitcher("Good", 3.0)),
    ]
    for ht, at, hp, ap in cases:
        ref = compute_win_probability(ht, at, hp, ap, cfg)
        res, info = compute_win_probability_griff(ht, at, hp, ap, cfg, 2026)
        assert info is None
        assert res.home_win_prob == ref.home_win_prob
        assert res.base_prob == ref.base_prob


def test_neutralization_on_shifts_base_when_rotation_known(monkeypatch):
    """With the flag ON and a known rotation rate, base is stripped of the
    team's season rotation strength and the total prob shifts accordingly."""
    cfg = load_griff_config()
    cfg = {**cfg, "model": {**cfg["model"], "starter_neutralized_base": True}}
    ht = _mk_team("Aces", 0.60, 50, 100, 4.0)     # strong record...
    at = _mk_team("Avg", 0.50, 50, 100, 4.0)
    hp = _mk_pitcher("TodaySP", 4.0)
    ap = _mk_pitcher("OppSP", 4.0)

    import mlb_value_bot.griffbet.win_probability as gwp
    # Home rotation a full run better than league avg -> strip ~+6.5pp win% out
    # of its base; away rotation league-average -> no strip.
    def fake_inc(team, season, config):
        return 0.065 if team == "Aces" else 0.0
    monkeypatch.setattr(gwp, "rotation_winpct_increment", fake_inc)

    off = compute_win_probability(ht, at, hp, ap, cfg)
    on, info = gwp.compute_win_probability_griff(ht, at, hp, ap, cfg, 2026)
    assert info["applied"] and not info["degraded"]
    assert info["base_shift"] < 0                      # home's base lowered
    assert on.home_win_prob < off.home_win_prob        # less credit double-counted
    # Away contributes 0 increment -> only home base moved.
    assert info["away_rotation_increment"] == 0.0


def test_rotation_increment_centers_on_league_mean(monkeypatch):
    """The increment is (league_mean_rate - team_rate)*run_to_wp, so it's
    scale-agnostic (works for xFIP or the ERA proxy) and sums to ~0 across the
    league. Unknown teams degrade to None."""
    import mlb_value_bot.griffbet.team_extras as te
    rates = {"Aces": 3.50, "Avg": 4.00, "Scrubs": 4.50}  # mean 4.00
    monkeypatch.setattr(te, "team_rotation_rates", lambda season, config=None: rates)
    cfg = load_griff_config()
    rtw = cfg["model"]["pitcher_run_to_winpct"]
    assert approx(te.rotation_winpct_increment("Aces", 2026, cfg), 0.5 * rtw)
    assert approx(te.rotation_winpct_increment("Scrubs", 2026, cfg), -0.5 * rtw)
    assert approx(te.rotation_winpct_increment("Avg", 2026, cfg), 0.0)
    assert te.rotation_winpct_increment("Unknown", 2026, cfg) is None


def test_neutralization_degrades_when_rotation_unknown(monkeypatch):
    cfg = load_griff_config()
    cfg = {**cfg, "model": {**cfg["model"], "starter_neutralized_base": True}}
    import mlb_value_bot.griffbet.win_probability as gwp
    monkeypatch.setattr(gwp, "rotation_winpct_increment", lambda *a, **k: None)
    ht = _mk_team("H", 0.55, 50, 100, 4.0)
    at = _mk_team("A", 0.50, 50, 100, 4.0)
    hp, ap = _mk_pitcher("X", 3.8), _mk_pitcher("Y", 4.2)
    off = compute_win_probability(ht, at, hp, ap, cfg)
    on, info = gwp.compute_win_probability_griff(ht, at, hp, ap, cfg, 2026)
    assert info["degraded"]
    assert approx(on.home_win_prob, off.home_win_prob)   # falls back to BiffBet base


# --- Adjusted EV omits the dead -2pp haircut ---------------------------------
def test_griff_adjusted_ev_omits_large_fade():
    from mlb_value_bot.griffbet.pipeline import _compute_adjusted_ev_griff
    cfg = load_griff_config()
    # A 6pp fade (would be -2pp under BiffBet's dead branch) is only -1pp here.
    adj, reasons = _compute_adjusted_ev_griff(0.05, 0.06, False, cfg)
    assert approx(adj, 0.04, tol=1e-9)
    assert any("sharp fade" in r for r in reasons)
    assert not any("large" in r for r in reasons)
    # Support still boosts; fragile still stacks.
    adj_s, _ = _compute_adjusted_ev_griff(0.05, -0.04, False, cfg)
    assert approx(adj_s, 0.06, tol=1e-9)
    adj_f, _ = _compute_adjusted_ev_griff(0.05, 0.04, True, cfg)
    assert approx(adj_f, 0.05 - 0.01 - 0.01, tol=1e-9)


# --- Discipline (correlation haircut + slate cap) ----------------------------
class _FakeEval:
    def __init__(self, ev_pct, kelly):
        self.ev_pct = ev_pct
        self.kelly_stake = kelly
        self.american_odds = -110


class _FakeAnalysis:
    """Minimal stand-in exposing best_eval + is_value for apply_discipline."""
    def __init__(self, kelly, ev_pct=0.05):
        self._eval = _FakeEval(ev_pct, kelly)
        self.discipline_reasons = []
        self.stake_before_discipline = None

    @property
    def best_eval(self):
        return self._eval

    def is_value(self, threshold):
        return self._eval.ev_pct >= threshold and self._eval.kelly_stake > 0


def test_correlation_haircut_fires_only_when_crowded():
    from mlb_value_bot.griffbet.pipeline import apply_discipline
    cfg = load_griff_config()
    # 3 bets (<= threshold 4) -> no haircut, total 0.015 < cap 0.05.
    small = [_FakeAnalysis(0.005) for _ in range(3)]
    apply_discipline(small, 0.03, cfg)
    assert all(a.best_eval.kelly_stake == 0.005 for a in small)
    assert all(not a.discipline_reasons for a in small)

    # 6 bets (> 4) -> 10% haircut each. Total after haircut 6*0.0045=0.027 < cap.
    crowded = [_FakeAnalysis(0.005) for _ in range(6)]
    apply_discipline(crowded, 0.03, cfg)
    assert all(approx(a.best_eval.kelly_stake, 0.0045, tol=1e-9) for a in crowded)
    assert all(any("correlation" in r for r in a.discipline_reasons) for a in crowded)


def test_slate_exposure_cap_scales_proportionally():
    from mlb_value_bot.griffbet.pipeline import apply_discipline
    cfg = load_griff_config()
    # 3 big bets summing to 0.09 (> cap 0.05), below correlation threshold so
    # only the cap fires. Each 0.03 -> scaled by 0.05/0.09.
    bets = [_FakeAnalysis(0.03) for _ in range(3)]
    apply_discipline(bets, 0.03, cfg)
    total = sum(a.best_eval.kelly_stake for a in bets)
    assert approx(total, 0.05, tol=1e-4)
    assert all(any("exposure cap" in r for r in a.discipline_reasons) for a in bets)
    assert all(a.stake_before_discipline == 0.03 for a in bets)


# --- Sharp-close extraction --------------------------------------------------
def test_sharp_line_prefers_pinnacle():
    from mlb_value_bot.griffbet.sharp_close import sharp_line_from_odds
    from types import SimpleNamespace
    go = SimpleNamespace(
        home_team="Boston Red Sox", away_team="New York Yankees", event_id="e1",
        all_books=[
            {"key": "betonlineag", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Boston Red Sox", "price": -120}, {"name": "New York Yankees", "price": 105}]}]},
            {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                {"name": "Boston Red Sox", "price": -125}, {"name": "New York Yankees", "price": 110}]}]},
        ],
    )
    sl = sharp_line_from_odds(go, priority=["pinnacle", "betonlineag", "lowvig"])
    assert sl is not None and sl.book == "pinnacle"
    assert sl.home_line == -125 and sl.away_line == 110
    # Falls through to next priority book when Pinnacle absent.
    go.all_books = [go.all_books[0]]
    sl2 = sharp_line_from_odds(go, priority=["pinnacle", "betonlineag", "lowvig"])
    assert sl2.book == "betonlineag"
    # No priority book -> None.
    go.all_books = [{"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [
        {"name": "Boston Red Sox", "price": -120}, {"name": "New York Yankees", "price": 105}]}]}]
    assert sharp_line_from_odds(go, priority=["pinnacle"]) is None


# --- Persistence: dual CLV streams + committed-bet freeze ---------------------
def test_griff_tracking_dual_clv_and_commit_freeze():
    import mlb_value_bot.griffbet.tracking as gt
    gt = importlib.reload(gt)
    gt.GRIFF_DB_PATH = Path(tempfile.mkdtemp()) / "griff.db"

    rec = gt.GriffRecord(
        date="2026-06-14", game_id=1, home_team="H", away_team="A",
        recommended_side="home", model_prob=0.55, market_prob_devigged=0.52,
        american_odds=-110, decimal_odds=ev.american_to_decimal(-110),
        ev_pct=0.05, kelly_stake=0.005, confidence=72.0,
        raw_model_prob=0.50, blended_prob=0.55, raw_pick_side="away", raw_pick_open=100,
        best_home_line=-110, best_away_line=100, is_value=True,
    )
    gt.upsert_recommendation(rec)
    # Re-price: best home shortens to -130; sharp Pinnacle home -125 / away +105.
    rec2 = gt.GriffRecord(
        date="2026-06-14", game_id=1, home_team="H", away_team="A",
        recommended_side="home", model_prob=0.55, market_prob_devigged=0.52,
        american_odds=-130, decimal_odds=ev.american_to_decimal(-130),
        ev_pct=0.03, kelly_stake=0.005, confidence=72.0,
        raw_model_prob=0.50, blended_prob=0.55, raw_pick_side="away", raw_pick_open=100,
        sharp_close_book="pinnacle", sharp_close_home_line=-125, sharp_close_away_line=105,
        best_home_line=-130, best_away_line=110, is_value=True,
    )
    gt.upsert_recommendation(rec2)
    row = dict(gt.get_for_date("2026-06-14")[0])
    assert row["opening_line"] == -110          # committed open never moves
    assert row["closing_line"] == -130
    assert row["clv_pct"] > 0                    # -110 open beat -130 best close
    assert row["clv_blended_vs_sharp"] > 0       # -110 beat sharp home close -125
    assert row["clv_raw_vs_sharp"] < 0           # away +100 worse than sharp +105
    assert row["sharp_close_book"] == "pinnacle"


# --- Referee metrics ---------------------------------------------------------
def test_referee_calibration_and_monotonicity():
    from mlb_value_bot.griffbet import referee as R
    assert R.brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert R.brier_score([0.0, 1.0], [1, 0]) == 1.0
    assert R.log_loss([0.5, 0.5], [1, 0]) == round(math.log(2), 4)
    buckets = R.reliability_buckets([0.55, 0.58, 0.85], [1, 0, 1], [0.0, 0.6, 1.0])
    assert buckets[0]["n"] == 2 and approx(buckets[0]["mean_actual"], 0.5)

    df = pd.DataFrame([
        {"result": "win", "american_odds": 100, "ev_pct": 0.04, "kelly_stake": 0.005, "profit_loss": 0.005},
        {"result": "loss", "american_odds": -110, "ev_pct": 0.04, "kelly_stake": 0.005, "profit_loss": -0.005},
        {"result": "win", "american_odds": 120, "ev_pct": 0.06, "kelly_stake": 0.01, "profit_loss": 0.012},
    ])
    mono = R.ev_monotonicity(df, [-1.0, 0.05, 1.0])
    assert len(mono) == 2
    band_low = next(b for b in mono if b["ev_band"].startswith("-100%"))
    assert band_low["wins"] == 1 and band_low["losses"] == 1


def test_referee_sharp_clv_join_reports_gap():
    from mlb_value_bot.griffbet.referee import biff_sharp_clv_via_join
    # One BiffBet bet matched to a GriffBet sharp close, one with no match (gap).
    biff = pd.DataFrame([
        {"date": "2026-06-14", "game_id": 1, "recommended_side": "home",
         "opening_line": -110, "is_value": 1},
        {"date": "2026-06-14", "game_id": 2, "recommended_side": "away",
         "opening_line": 120, "is_value": 1},
    ])
    griff = pd.DataFrame([
        {"date": "2026-06-14", "game_id": 1, "sharp_close_home_line": -125,
         "sharp_close_away_line": 110},
    ])
    out = biff_sharp_clv_via_join(biff, griff)
    assert out["available"] and out["n_matched"] == 1 and out["n_gap_no_sharp_close"] == 1
    assert out["avg"] > 0      # -110 open beat sharp -125 close


# --- Residual market-error engine --------------------------------------------
def test_residual_predict_equals_market_when_beta_zero():
    """β all-zero (cold start) => p_home is exactly the market devig prob."""
    from mlb_value_bot.griffbet.residual_model import ResidualModel, FEATURES
    m = ResidualModel(FEATURES, [0.0] * len(FEATURES), [0.0] * len(FEATURES),
                      [1.0] * len(FEATURES), l2=1.0, n_train=0)
    for mkt in (0.40, 0.55, 0.73):
        assert approx(m.predict_home_prob({f: 0.5 for f in FEATURES}, mkt), mkt, tol=1e-9)


def test_residual_recovers_signal():
    """With a real signal and light regularization, the model learns a positive
    coefficient and tilts above the market when the feature is high."""
    import numpy as np
    import pandas as pd
    from mlb_value_bot.griffbet.residual_model import FEATURES, fit_residual_model
    rng = np.random.default_rng(0)
    n = 400
    starter = rng.normal(0, 1, n)
    market = np.full(n, 0.5)
    # Home wins with prob sigmoid(1.5*starter) -- market (0.5) is "wrong" in a
    # way the starter feature explains: pure market error.
    p_true = 1 / (1 + np.exp(-1.5 * starter))
    home_won = (rng.uniform(0, 1, n) < p_true).astype(int)
    data = {f: np.zeros(n) for f in FEATURES}
    data["starter"] = starter
    df = pd.DataFrame({**data, "market_devig_home": market, "home_won": home_won,
                       "date": ["2026-01-01"] * n})
    model = fit_residual_model(df, l2=0.5)
    b = dict(zip(model.features, model.beta))
    assert b["starter"] > 0.2                       # learned the signal
    # High starter -> prob well above the 0.5 market; low -> below.
    hi = model.predict_home_prob({**{f: 0.0 for f in FEATURES}, "starter": 2.0}, 0.5)
    lo = model.predict_home_prob({**{f: 0.0 for f in FEATURES}, "starter": -2.0}, 0.5)
    assert hi > 0.55 and lo < 0.45


def test_residual_cold_start_shrinks_to_market():
    """Few rows + heavy L2 => β ~ 0 => predictions hug the market (no
    data-starved overconfidence)."""
    import numpy as np
    import pandas as pd
    from mlb_value_bot.griffbet.residual_model import FEATURES, fit_residual_model
    rng = np.random.default_rng(1)
    n = 12
    df = pd.DataFrame({
        **{f: rng.normal(0, 1, n) for f in FEATURES},
        "market_devig_home": rng.uniform(0.45, 0.55, n),
        "home_won": rng.integers(0, 2, n), "date": ["2026-01-01"] * n,
    })
    model = fit_residual_model(df, l2=500.0)
    assert max(abs(b) for b in model.beta) < 0.02            # β crushed toward 0
    mkt = 0.52
    assert approx(model.predict_home_prob({f: 1.0 for f in FEATURES}, mkt), mkt, tol=0.02)


def test_residual_feature_extraction():
    """feature_vector pulls component weighted-deltas + market context; _home_won
    resolves the home outcome from result + pick side."""
    from mlb_value_bot.griffbet.residual_model import feature_vector, _home_won
    reasoning = {
        "market_anchor": {"market_devig_home_prob": 0.48, "data_confidence": 80.0},
        "market_intel": {"sharp_minus_square_pp": 1.5, "dispersion_pp": 0.6},
        "components": [
            {"name": "starter", "weighted_delta": 0.03},
            {"name": "bullpen", "weighted_delta": -0.01},
            {"name": "home_field", "weighted_delta": 0.025},
        ],
    }
    f = feature_vector(reasoning)
    assert approx(f["starter"], 0.03) and approx(f["bullpen"], -0.01)
    assert approx(f["lineup"], 0.0)                      # missing component -> 0
    assert approx(f["sharp_minus_square"], 0.015) and approx(f["data_confidence"], 0.8)
    assert approx(f["_market_devig_home"], 0.48)
    assert feature_vector({"components": []}) is None    # no market anchor -> None
    # home won iff (home pick & win) or (away pick & loss)
    assert _home_won("win", "home") == 1 and _home_won("loss", "home") == 0
    assert _home_won("win", "away") == 0 and _home_won("loss", "away") == 1
    assert _home_won("pending", "home") is None


def test_engine_switch_residual_vs_warmup():
    """engine='residual' uses the model only past the warmup gate (n_train floor
    AND beats-market OOS); otherwise it falls back to the blend."""
    from mlb_value_bot.griffbet.pipeline import resolve_engine_prob
    from mlb_value_bot.griffbet.residual_model import ResidualModel, FEATURES
    cfg = load_griff_config()
    assert cfg["model"]["engine"] == "residual"
    feat = {f: 0.0 for f in FEATURES}
    feat["starter"] = 2.0
    good_oos = {"sufficient": True, "beats_market_log_loss": True}

    # Below the n_train floor -> warmup -> returns the blend untouched.
    weak = ResidualModel(FEATURES, [0.0] * len(FEATURES), [0.0] * len(FEATURES),
                         [1.0] * len(FEATURES), l2=50, n_train=10, oos=good_oos)
    final, info = resolve_engine_prob(0.40, feat, 0.50, weak, cfg)
    assert info["mode"] == "warmup_blend" and final == 0.40

    # Past the floor + beats market OOS -> residual drives the prob.
    beta = [0.0] * len(FEATURES)
    beta[FEATURES.index("starter")] = 0.5
    ready = ResidualModel(FEATURES, beta, [0.0] * len(FEATURES), [1.0] * len(FEATURES),
                          l2=50, n_train=500, oos=good_oos)
    final2, info2 = resolve_engine_prob(0.40, feat, 0.50, ready, cfg)
    assert info2["mode"] == "residual" and final2 > 0.50   # starter>0, β>0 -> above market

    # Past the floor but OOS says it does NOT beat the market -> held in warmup.
    held = ResidualModel(FEATURES, beta, [0.0] * len(FEATURES), [1.0] * len(FEATURES),
                         l2=50, n_train=500, oos={"sufficient": True, "beats_market_log_loss": False})
    _, info3 = resolve_engine_prob(0.40, feat, 0.50, held, cfg)
    assert info3["mode"] == "warmup_blend"


# --- Stage 4: free training features -----------------------------------------
def test_pitcher_quality_and_lineup_features():
    from types import SimpleNamespace
    from mlb_value_bot.griffbet.features import pitcher_quality_features, lineup_features
    home = SimpleNamespace(whiff_pct=0.30, csw_pct=0.32, hardhit_pct=0.34)
    away = SimpleNamespace(whiff_pct=0.24, csw_pct=0.28, hardhit_pct=0.40)
    pq = pitcher_quality_features(home, away)
    assert approx(pq["whiff_net"], 0.06) and approx(pq["csw_net"], 0.04)
    assert approx(pq["hardhit_net"], 0.06)            # away-home, + favors home (home suppresses contact)
    # Missing metric -> 0.
    assert pitcher_quality_features(SimpleNamespace(whiff_pct=None, csw_pct=None, hardhit_pct=None), away)["whiff_net"] == 0.0
    # Lineup: only counts when BOTH confirmed.
    h_lu = SimpleNamespace(is_confirmed=True, missing_count=1)
    a_lu = SimpleNamespace(is_confirmed=True, missing_count=2)
    lf = lineup_features(h_lu, a_lu)
    assert lf["lineup_confirmed"] == 1.0 and approx(lf["keybats_net"], 1.0)   # away_missing-home_missing
    proj = lineup_features(SimpleNamespace(is_confirmed=False, missing_count=0), a_lu)
    assert proj["lineup_confirmed"] == 0.0 and proj["keybats_net"] == 0.0


def test_weather_degrades_when_disabled_or_unknown():
    from mlb_value_bot.griffbet.features import weather_features
    cfg = {"griff_features": {"weather": {"enabled": False}}}
    assert weather_features("Boston Red Sox", "2026-06-14", cfg) == {"temp": 0.0, "wind": 0.0}
    # Unknown park -> neutral, even when enabled (no network call).
    cfg_on = {"griff_features": {"weather": {"enabled": True}}}
    assert weather_features("Nonexistent Team", "2026-06-14", cfg_on) == {"temp": 0.0, "wind": 0.0}


def test_extractor_reads_griff_features_and_defaults_missing():
    from mlb_value_bot.griffbet.residual_model import feature_vector, FEATURES
    from mlb_value_bot.griffbet.features import GRIFF_FEATURE_KEYS
    # New keys are part of the model's feature schema.
    assert all(k in FEATURES for k in GRIFF_FEATURE_KEYS)
    # Present block is read through.
    r = {"market_anchor": {"market_devig_home_prob": 0.5},
         "components": [], "griff_features": {"whiff_net": 0.05, "temp": 3.0}}
    f = feature_vector(r)
    assert approx(f["whiff_net"], 0.05) and approx(f["temp"], 3.0)
    # Row predating Stage 4 (no block) -> all new features default to 0.
    r2 = {"market_anchor": {"market_devig_home_prob": 0.5}, "components": []}
    f2 = feature_vector(r2)
    assert all(approx(f2[k], 0.0) for k in GRIFF_FEATURE_KEYS)


def test_lineup_features_from_reasoning():
    from mlb_value_bot.griffbet.features import lineup_features_from_reasoning
    both = {"lineup": {"home": {"status": "confirmed", "missing_key_bats": ["X"]},
                       "away": {"status": "confirmed", "missing_key_bats": ["Y", "Z"]}}}
    f = lineup_features_from_reasoning(both)
    assert f["lineup_confirmed"] == 1.0 and approx(f["keybats_net"], 1.0)   # 2 away - 1 home
    proj = {"lineup": {"home": {"status": "projected"}, "away": {"status": "confirmed"}}}
    assert lineup_features_from_reasoning(proj) == {"lineup_confirmed": 0.0, "keybats_net": 0.0}
    assert lineup_features_from_reasoning({}) == {"lineup_confirmed": 0.0, "keybats_net": 0.0}


def test_feature_store_join_in_extractor(monkeypatch):
    """Backfilled features in GriffBet's feature store are joined into training
    rows whose base reasoning lacks them (e.g. BiffBet-only historical games)."""
    import importlib
    import mlb_value_bot.utils as utils
    from pathlib import Path
    import tempfile
    utils.DB_PATH = Path(tempfile.mkdtemp()) / "biff.db"
    griff_tracking = importlib.reload(importlib.import_module("mlb_value_bot.griffbet.tracking"))
    griff_tracking.GRIFF_DB_PATH = Path(tempfile.mkdtemp()) / "griff.db"

    # A graded BiffBet row WITHOUT a griff_features block.
    import mlb_value_bot.tracking.recommendations as biff
    biff = importlib.reload(biff)
    rec = biff.RecommendationRecord(
        date="2026-05-25", game_id=11, home_team="H", away_team="A",
        recommended_side="home", model_prob=0.55, market_prob_devigged=0.52,
        american_odds=-110, decimal_odds=ev.american_to_decimal(-110), ev_pct=0.05,
        kelly_stake=0.005, confidence=70.0,
        reasoning={"market_anchor": {"market_devig_home_prob": 0.52}, "components": []},
    )
    rid = biff.upsert_recommendation(rec)
    biff.update_result(rid, "win", 0.004)

    # Backfill a feature for that game into the GriffBet feature store.
    griff_tracking.upsert_features("2026-05-25", 11, {"whiff_net": 0.07, "temp": 5.0})

    from mlb_value_bot.griffbet.residual_model import extract_training_data
    df = extract_training_data()
    row = df[df["game_id"].astype(int) == 11] if "game_id" in df.columns else df
    # game_id isn't a column; just check the single extracted row picked up the join.
    assert len(df) == 1
    assert approx(float(df.iloc[0]["whiff_net"]), 0.07)
    assert approx(float(df.iloc[0]["temp"]), 5.0)
    assert int(df.iloc[0]["home_won"]) == 1
    importlib.reload(biff)
    importlib.reload(griff_tracking)


def test_matchup_math():
    """compute_matchup nets home vs away expected offense; + favors home."""
    from mlb_value_bot.griffbet.matchup import compute_matchup, _lineup_offense
    # Home pitcher throws 100% fastball; away pitcher 100% slider.
    home_arsenal = {"FF": 1.0}
    away_arsenal = {"SL": 1.0}
    # Home batters crush sliders (.400 vs SL); away batters weak vs fastballs (.250 vs FF).
    home_batters = [{"SL": 0.400}, {"SL": 0.420}]
    away_batters = [{"FF": 0.250}, {"FF": 0.230}]
    net = compute_matchup(home_arsenal, away_arsenal, home_batters, away_batters)
    # home_offense ~0.41 (vs away SL), away_offense ~0.24 (vs home FF) -> + favors home.
    assert net > 0.15
    # Symmetric skill -> ~0.
    sym = compute_matchup({"FF": 1.0}, {"FF": 1.0}, [{"FF": 0.32}], [{"FF": 0.32}])
    assert approx(sym, 0.0, tol=1e-9)
    # Missing data on a side -> degrade to 0.
    assert compute_matchup({"FF": 1.0}, {}, [{"FF": 0.3}], [{"FF": 0.3}]) == 0.0
    # Batter with no overlapping pitch type is skipped, not counted as 0.
    assert _lineup_offense({"FF": 1.0}, [{"SL": 0.5}]) is None


def test_matchup_feature_requires_confirmed_lineups():
    """The live feature is 0 unless BOTH lineups are confirmed (look-ahead-free
    + bounds the heavy pulls to near first pitch)."""
    from types import SimpleNamespace
    from datetime import date
    from mlb_value_bot.griffbet.matchup import matchup_feature
    cfg = load_griff_config()
    pp = SimpleNamespace(player_id=1)
    proj = SimpleNamespace(is_confirmed=False, batting_order_ids=[])
    conf = SimpleNamespace(is_confirmed=True, batting_order_ids=[100, 101])
    assert matchup_feature(pp, pp, proj, conf, 2026, date(2026, 6, 14), cfg) == 0.0
    # Disabled -> 0 even when confirmed (no network).
    off = {"griff_features": {"matchup": {"enabled": False}}}
    assert matchup_feature(pp, pp, conf, conf, 2026, date(2026, 6, 14), off) == 0.0


# --- Self-running harness (mirrors test_core.py) -----------------------------
def _run_all():
    import inspect, sys
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction)
           if n.startswith("test_")]
    passed = 0
    for name, fn in fns:
        params = inspect.signature(fn).parameters
        if "monkeypatch" in params:
            mp, undo = _MiniMonkeypatch(), None
            try:
                fn(mp)
            finally:
                mp.undo()
        else:
            fn()
        print(f"  PASS  {name}")
        passed += 1
    print(f"\n{passed}/{len(fns)} GriffBet tests passed.")


class _MiniMonkeypatch:
    """Tiny setattr-only monkeypatch so the file runs without pytest."""
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value):
        old = getattr(target, name)
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def undo(self):
        for target, name, old in reversed(self._undo):
            setattr(target, name, old)


if __name__ == "__main__":
    _run_all()
