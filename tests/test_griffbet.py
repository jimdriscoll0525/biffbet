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
    """Flag OFF == BiffBet exactly, across several matchups."""
    cfg = load_griff_config()
    assert cfg["model"]["starter_neutralized_base"] is False
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
