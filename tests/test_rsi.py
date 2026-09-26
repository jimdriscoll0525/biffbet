"""RSI analytics tests -- fixtures only, no network, no Supabase.

Run: .venv/Scripts/python.exe -m pytest tests/test_rsi.py -q
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlb_value_bot.rsi import holdout, overlay, reconcile, report, segments, shadow_stats, stats
from mlb_value_bot.rsi import suggest as sg
from mlb_value_bot.rsi import versions
from mlb_value_bot.rsi.import_ledger import build_rows
from mlb_value_bot.rsi.review import review_sport

CFG = {
    "min_sample": {"mlb": 6, "mlb_totals": 6, "football": 4, "griffbet": 6},
    "t_threshold": 2.0, "consecutive_runs_for_pending": 2, "clv_min_tracked": 3,
    "holdout_pct": 20, "shadow_weeks": {"mlb": 4}, "rolling_window": {"mlb": 5, "football": 4},
    "rollback": {"clv_floor": 0.0, "clv_drop_vs_baseline": 0.5}, "holdout_min_n": 2,
    "small_favorites": {"pools": ["ml-bets", "ml-passes"], "dim": "fav_size",
                        "values": ["small fav -101..-119", "fav -120..-149"]},
}


# --- overlay -------------------------------------------------------------------
def test_overlay_apply_validate_diff():
    base = {"filters": {"heavy_favorite_american": -150, "min_model_prob": 0.5},
            "model": {"weights": {"starter": 0.4}, "market_blend": 0.35}, "list": [1, 2]}
    out = overlay.apply_overlay(base, {"filters.min_model_prob": 0.55, "model.weights.form": 0.1,
                                       "filters.heavy_favorite_american": None, "list": [3]})
    assert out["filters"] == {"min_model_prob": 0.55}
    assert out["model"]["weights"] == {"starter": 0.4, "form": 0.1}
    assert out["list"] == [3]
    assert base["filters"]["min_model_prob"] == 0.5          # deep copy, base untouched
    allow = {"mlb": ["filters.", "model.market_blend", "model.weights."]}
    assert overlay.validate_overlay("mlb", {"filters.min_model_prob": 0.5,
                                            "model.market_blend": 0.4}, allow) == []
    problems = overlay.validate_overlay("mlb", {"odds_api.key": "x", "filters.x": {"a": 1}}, allow)
    assert any("not in the mlb allowlist" in p for p in problems)
    assert any("JSON scalar" in p for p in problems)
    assert overlay.validate_overlay("nope", {"filters.a": 1}, allow)
    assert overlay.diff_overlay(base, out) == {"filters.heavy_favorite_american": None,
                                               "filters.min_model_prob": 0.55, "list": [3],
                                               "model.weights.form": 0.1}


# --- holdout -------------------------------------------------------------------
def test_holdout_pinned_and_share():
    for key, gid in (("mlb", 745123), ("mlb_totals", 745123), ("cfb", "401628391"), ("nfl", "2026_01_KC_BAL")):
        expected = int(hashlib.md5(f"{key}:{gid}".encode()).hexdigest()[:7], 16) % 100 < 20
        assert holdout.is_holdout(key, gid) is expected
    # A few concrete values (regression pins; the SQL twin must agree).
    assert holdout.is_holdout("mlb", 1) == (int(hashlib.md5(b"mlb:1").hexdigest()[:7], 16) % 100 < 20)
    share = sum(holdout.is_holdout("mlb", i) for i in range(10_000)) / 10_000
    assert 0.15 <= share <= 0.25
    assert holdout.sport_key_for("football", "cfb", "cfb") == "cfb"
    assert holdout.sport_key_for("mlb_totals") == "mlb_totals"


# --- segments ------------------------------------------------------------------
def _ml_view_row(i, price, side="home", result="win", clv=1.0, open_p=None, close_p=None,
                 blended_home=0.56, decision="pick", pass_reason=None, date="2026-06-01"):
    return {"engine": "mlb", "sport": "mlb", "league": None, "date": date, "game_key": str(i),
            "home_team": "A", "away_team": "B", "market": "moneyline", "pick_side": side,
            "features": {"market_anchor": {"blended_home_prob": blended_home, "blend_tier": "mid"},
                         "market_intel": {"sharp_devig_home": 0.5, "dispersion_pp": 0.3},
                         "stability": {"label": "stable"}, "bet_sizing": {"tier": "small"}},
            "model_version": "biff_v1", "ev_pct": 0.04, "adjusted_ev_pct": 0.03, "confidence": 65,
            "decision": decision, "pass_reason": pass_reason,
            "opening_line": float(open_p if open_p is not None else price),
            "opening_price": open_p if open_p is not None else price, "pick_price": price,
            "pick_line": None, "closing_line": float(close_p if close_p is not None else price),
            "closing_price": close_p if close_p is not None else price, "result": result,
            "clv": clv, "clv_metric": "clv_pct",
            "flat_pl_units": {"win": 100 / -price if price < 0 else price / 100, "loss": -1.0}.get(result),
            "kelly_pl_units": None, "is_holdout": False}


def test_segments_mlb_view_shape():
    df = pd.DataFrame([
        _ml_view_row(1, -105, open_p=-105, close_p=-115),          # moved toward
        _ml_view_row(2, -135, open_p=-135, close_p=-130),          # against, tiny (1pp) -> against
        _ml_view_row(3, -160, open_p=-160, close_p=-162),          # < 0.5pp -> flat
        _ml_view_row(4, 110, side="away", blended_home=0.55),      # away pick prob 0.45
        _ml_view_row(5, 130), _ml_view_row(6, 175), _ml_view_row(7, 250), _ml_view_row(8, -230),
        _ml_view_row(9, -120, decision="pass", pass_reason="filter:heavy_favorite+skip:x",
                     result="pending"),
    ])
    out = segments.prepare_mlb(df)
    assert list(out["fav_size"]) == ["small fav -101..-119", "fav -120..-149", "fav -150..-199",
                                     "small dog +100..+119", "dog +120..+149", "dog +150..+199",
                                     "big dog +200+", "big fav -200+", "fav -120..-149"]
    assert list(out["line_move"][:3]) == ["toward", "against", "flat"]
    assert out["model_prob_bucket"].iloc[3] == "<50%"
    assert out["model_prob_bucket"].iloc[0] == "55-60%"
    assert out["pass_reason"].iloc[8] == "filter:heavy_favorite"
    assert out["sharp_fade"].iloc[0] == "fade 4pp+"
    assert out["stability"].iloc[0] == "stable" and out["bet_tier"].iloc[0] == "small"
    assert out["month"].iloc[0] == "2026-06"
    assert out["settled"].iloc[0] and not out["settled"].iloc[8]
    assert math.isclose(out["flat_pl"].iloc[0], 100 / 105, rel_tol=1e-6)
    for dim in segments.ML_BET_DIMS + segments.ML_PASS_DIMS:
        assert dim in out.columns, dim


def test_segments_totals_rain_and_line_move():
    def row(i, side, precip_prob=None, precip_mm=None, open_line=8.5, close_line=9.0):
        w = {"roof": "open", "wind_out_component": 8.0}
        if precip_prob is not None:
            w["precip_prob"] = precip_prob
        if precip_mm is not None:
            w["precip_mm"] = precip_mm
        return {"engine": "mlb_totals", "sport": "mlb_totals", "date": "2026-07-01", "game_key": str(i),
                "market": "total", "pick_side": side,
                "features": json.dumps({"weather": w, "run_distribution": {"tilt": 0.4},
                                        "pick": {"tier": "small"}, "stability": {"label": "moderate"}}),
                "ev_pct": 0.04, "confidence": 50, "decision": "pick", "pass_reason": None,
                "opening_line": open_line, "opening_price": -110, "pick_price": -110, "pick_line": 8.5,
                "closing_line": close_line, "closing_price": -110, "result": "win", "clv": 0.5,
                "clv_metric": "clv_pp", "flat_pl_units": 0.909}
    df = pd.DataFrame([row(1, "over", precip_prob=60), row(2, "under", precip_mm=0.7),
                       row(3, "over", precip_prob=10), row(4, "under")])
    out = segments.prepare_totals(df)
    assert list(out["rain"]) == ["rain", "rain", "dry", "n/a"]
    assert list(out["line_move"]) == ["toward", "against", "toward", "against"]
    assert out["tier"].iloc[0] == "small" and out["stability"].iloc[0] == "moderate"
    assert out["tilt_bucket"].iloc[0] == "mild over (0.25..0.75)"
    assert out["line_bucket"].iloc[0] == "mid (8-8.5)"
    for dim in segments.TOTALS_BET_DIMS + segments.TOTALS_PASS_DIMS:
        assert dim in out.columns, dim


def test_segments_football_lenses():
    feats = {"stability": {"label": "stable"}, "hold_reason": None,
             "weather": {"indoor": False, "wind_mph": 14, "precip_prob": 60},
             "context": {"home_rest_days": 7, "away_rest_days": 5, "conference_game": True,
                         "division_game": False},
             "matchup": {"archetype": "pass O vs pass D", "dual_edge_side": "home",
                         "edges": [{"edge": 22.0}, {"edge": -8.0}]},
             "ol": {"home": {"grade": 0.5}, "away": {"grade": -0.2}}}
    rows = [
        {"engine": "football", "sport": "cfb", "league": "cfb", "date": "2026-09-05", "game_key": "g1",
         "market": "spread", "pick_side": "home", "features": feats, "ev_pct": 0.05, "confidence": 70,
         "decision": "pick", "pass_reason": None, "opening_line": -6.5, "opening_price": -110,
         "pick_price": -110, "pick_line": -7.5, "closing_line": -7.5, "closing_price": -110,
         "result": "win", "clv": 0.02, "clv_metric": "clv_pp", "flat_pl_units": 0.909},
        {"engine": "football", "sport": "nfl", "league": "nfl", "date": "2026-09-13", "game_key": "g2",
         "market": "spread", "pick_side": "away", "features": {**feats, "context": {"away_rest_days": 15}},
         "ev_pct": 0.03, "confidence": 55, "decision": "pass",
         "pass_reason": "divergence guard: model 4.0 vs market mean 7.5", "opening_line": 7.0,
         "opening_price": -110, "pick_price": -105, "pick_line": 7.5, "closing_line": 6.5,
         "closing_price": -110, "result": "loss", "clv": -0.01, "clv_metric": "clv_pp", "flat_pl_units": -1},
        {"engine": "football", "sport": "nfl", "league": "nfl", "date": "2026-09-13", "game_key": "g3",
         "market": "total", "pick_side": "under", "features": {"hold_reason": "adjusted EV +1.1% below threshold"},
         "ev_pct": 0.02, "confidence": 50, "decision": "pass", "pass_reason": "adjusted EV +1.1% below threshold",
         "opening_line": 47.5, "opening_price": -110, "pick_price": -110, "pick_line": 47.0,
         "closing_line": 46.5, "closing_price": -110, "result": "win", "clv": 0.0, "clv_metric": "clv_pp",
         "flat_pl_units": 0.909},
    ]
    out = segments.prepare_football(pd.DataFrame(rows))
    r0, r1, r2 = out.iloc[0], out.iloc[1], out.iloc[2]
    assert r0["league_market"] == "cfb spread" and r0["spread_band"] == "3.5-7.5"
    assert r0["lean_type"] == "favorite" and r1["lean_type"] == "dog"
    assert r0["venue_side"] == "home" and r1["venue_side"] == "away" and r2["venue_side"] == "n/a"
    assert r0["fav_size"] == "favorite 3.5-7.5" and r1["fav_size"] == "dog 3.5-7.5"
    assert r0["rest_bucket"] == "normal (6-8)" and r1["rest_bucket"] == "bye (14+)" and r2["rest_bucket"] == "n/a"
    assert r0["conference_game"] == "conference" and r0["division_game"] == "non-division"
    assert r2["conference_game"] == "n/a"
    assert r0["wind_bucket"] == "windy (12-18)" and r0["rain"] == "rain"
    assert r0["archetype"] == "pass O vs pass D"
    assert r0["dual_edge"] == "dual edge (pick)" and r1["dual_edge"] == "dual edge (other)"
    assert r0["ol_grade_pick"] == "strong" and r1["ol_grade_pick"] == "neutral"
    assert r0["phase_edge_bucket"] == "20-35"
    assert r0["line_move"] == "toward" and r1["line_move"] == "toward"     # +7 -> +6.5 toward dog
    assert r1["hold_kind"] == "divergence guard" and r2["hold_kind"] == "below EV threshold"
    assert r0["hold_kind"] == "pick"
    for dim in segments.FB_BET_DIMS + segments.FB_HOLD_DIMS:
        assert dim in out.columns, dim


def test_performance_prepare_parity():
    """tracking/performance._prepare delegates to rsi.segments.prepare_mlb and
    keeps its historical columns, labels and categorical dtypes."""
    from mlb_value_bot.tracking import performance as perf

    rj = json.dumps({"stability": {"label": "fragile"}, "bet_sizing": {"tier": "standard"},
                     "market_anchor": {"blend_tier": "high"},
                     "market_intel": {"sharp_devig_home": 0.50}})
    df = pd.DataFrame([
        {"date": "2026-06-01", "game_id": 1, "recommended_side": "home", "model_prob": 0.545,
         "american_odds": -130, "decimal_odds": 1.769, "ev_pct": 0.04, "kelly_stake": 0.006,
         "confidence": 65, "opening_line": -130, "closing_line": -140, "clv_pct": 1.2,
         "result": "win", "profit_loss": 0.004, "is_value": 1, "reasoning_json": rj},
        {"date": "2026-06-02", "game_id": 2, "recommended_side": "away", "model_prob": 0.45,
         "american_odds": 160, "decimal_odds": 2.6, "ev_pct": 0.09, "kelly_stake": 0.012,
         "confidence": 45, "opening_line": 160, "closing_line": 150, "clv_pct": None,
         "result": "loss", "profit_loss": -0.012, "is_value": 1, "reasoning_json": None},
        {"date": "2026-06-03", "game_id": 3, "recommended_side": "home", "model_prob": 0.60,
         "american_odds": -110, "decimal_odds": 1.909, "ev_pct": 0.13, "kelly_stake": 0.02,
         "confidence": 85, "opening_line": -110, "closing_line": None, "clv_pct": -0.5,
         "result": "pending", "profit_loss": None, "is_value": 1, "reasoning_json": "{}"},
    ])
    out = perf._prepare(df)
    assert list(out["settled"]) == [True, True, False]
    assert math.isclose(out["flat_pl"].iloc[0], 100 / 130, rel_tol=1e-6)
    assert out["flat_pl"].iloc[1] == -1.0 and np.isnan(out["flat_pl"].iloc[2])
    assert list(out["confidence_bucket"].astype(str)) == ["60-80", "40-60", "80-100"]
    assert list(out["ev_bucket"].astype(str)) == ["3-5%", "8-12%", "12%+"]
    assert list(out["kelly_bucket"].astype(str)) == ["0.5-1%", "1-1.5%", "1.5%+"]
    assert list(out["side_type"]) == ["favorite", "underdog", "favorite"]
    assert list(out["venue_side"]) == ["home", "road", "home"]
    assert list(out["clv_sign"]) == ["CLV+", "unknown", "CLV-"]
    assert list(out["bet_tier"]) == ["standard", "n/a", "n/a"]
    assert list(out["blend_tier"]) == ["high", "n/a", "n/a"]
    assert list(out["stability"].astype(str)) == ["fragile", "n/a", "n/a"]
    assert list(out["stability"].cat.categories) == ["stable", "moderate", "fragile", "n/a"]
    assert list(out["sharp_fade"].astype(str)) == ["fade 4pp+", "n/a", "n/a"]
    assert list(out["sharp_fade"].cat.categories) == ["sharps agree 3pp+", "neutral", "fade 3-4pp",
                                                       "fade 4pp+", "n/a"]
    assert list(out["odds_bucket"].astype(str)) == ["small fav (-101..-149)", "big dog (+151+)",
                                                     "small fav (-101..-149)"]
    for col in ("confidence_bucket", "ev_bucket", "kelly_bucket", "odds_bucket", "stability", "sharp_fade"):
        assert isinstance(out[col].dtype, pd.CategoricalDtype), col
    # The full report still builds from a prepared frame.
    rep_overall = perf._stats(out)
    assert rep_overall["settled"] == 2 and rep_overall["wins"] == 1
    seg = perf._segment(out, "odds_bucket")
    assert list(seg["odds_bucket"]) == ["small fav (-101..-149)", "big dog (+151+)"]


# --- stats ---------------------------------------------------------------------
def _six_row_frame():
    rows = [_ml_view_row(i, -110, result=r, clv=c) for i, (r, c) in enumerate(
        [("win", 1.0), ("win", 0.5), ("loss", 0.2), ("win", 1.5), ("loss", -0.3), ("push", None)], 1)]
    rows[5]["flat_pl_units"] = 0.0
    return segments.prepare_mlb(pd.DataFrame(rows))


def test_cell_stats_and_bonferroni():
    s = stats.cell_stats(_six_row_frame(), "clv_pct")
    assert s["rows"] == 6 and s["settled"] == 5 and s["wins"] == 3 and s["losses"] == 2 and s["pushes"] == 1
    assert s["hit_rate"] == 0.6 and s["clv_tracked"] == 5 and s["clv_positive"] == 4
    assert s["clv_metric"] == "clv_pct" and math.isclose(s["avg_clv"], 0.58, abs_tol=1e-6)
    assert math.isclose(s["breakeven_hit_rate"], 110 / 210, rel_tol=1e-4)
    assert math.isclose(s["flat_pl_units"], 3 * (100 / 110) - 2, abs_tol=1e-3)
    assert s["t_stat"] is not None and s["p_value"] is not None
    assert s["date_from"] == "2026-06-01" and s["date_to"] == "2026-06-01"
    lo, hi = s["hit_rate_wilson_95"]
    assert lo < 0.6 < hi
    assert stats.bonferroni(0.01, 4) is True and stats.bonferroni(0.01, 6) is False
    assert stats.bonferroni(None, 3) is False
    t, p = stats.t_test(pd.Series([1.0, 1.0, 1.0]))
    assert math.isnan(t) and math.isnan(p)


def test_segment_report_flags_only_strong_cells():
    rows = [_ml_view_row(i, -110 - 5 * i, result="win" if i != 8 else "loss", clv=1.0 + 0.1 * i)
            for i in range(1, 9)]
    rows += [_ml_view_row(i, 150 + 5 * i, result="loss" if i != 16 else "win", clv=-1.0 - 0.1 * i)
             for i in range(9, 17)]
    frame = segments.prepare_mlb(pd.DataFrame(rows))
    cands, tables, tested = stats.segment_report(frame, ["side_type", "month"], "ml-bets", 6, 2.0, 3)
    keys = {c.key: c for c in cands}
    assert "ml-bets|side_type=favorite" in keys and "ml-bets|side_type=underdog" in keys
    assert keys["ml-bets|side_type=favorite"].direction == "positive"
    assert keys["ml-bets|side_type=underdog"].direction == "negative"
    assert keys["ml-bets|side_type=underdog"].clv_agrees is True
    assert tested == 3            # favorite, underdog, one month
    assert tables["side_type"]["favorite"]["settled"] == 8


# --- reconcile -----------------------------------------------------------------
def _finding(key="ml-bets|fav_size=fav -120..-149", settled=80, bonf=False, clv_agrees=True,
             dim=None, direction="negative", engine="mlb", value=None, pool=None):
    pool = pool or key.split("|")[0]
    dim = dim or key.split("|")[1].split("=")[0]
    value = value or key.split("=", 1)[1]
    return {"key": key, "pool": pool, "dimension": dim, "value": value, "direction": direction,
            "engine": engine, "sport": "mlb", "rows": settled + 5, "settled": settled, "wins": 30,
            "losses": settled - 30, "hit_rate": 0.4, "flat_roi": -0.12, "t_stat": -2.4, "p_value": 0.02,
            "avg_clv": -0.6, "clv_metric": "clv_pct", "clv_tracked": settled, "date_from": "2026-05-01",
            "date_to": "2026-09-20", "bonferroni_significant": bonf, "clv_agrees": clv_agrees,
            "cells_tested": 120, "suggestion": {"kind": "overlay", "title": "T", "description": "D",
                                                "suggested_change": "C",
                                                "overlay": {"filters.heavy_favorite_american": -119}}}


def _existing(key, status, hits, misses=0, settled=70, extra=None):
    row = {"id": 7, "engine": "mlb", "sport": "mlb", "finding_key": key, "kind": "overlay",
           "title": "old", "description": "old", "suggested_change": None, "overlay": None,
           "status": status, "confidence": "low", "direction": "negative", "latest_stats": {},
           "evidence": [{"run_date": "2026-09-14", "settled": settled}], "consecutive_hits": hits,
           "consecutive_misses": misses, "first_seen": "2026-09-14", "last_seen": "2026-09-14",
           "snoozed_until": None, "created_at": "x"}
    row.update(extra or {})
    return row


def test_reconcile_new_finding_is_watch():
    ch = reconcile.reconcile([_finding()], [], "2026-09-21", CFG)
    assert len(ch) == 1 and ch[0].is_new and ch[0].status_after == "watch"
    row = ch[0].row
    assert row["consecutive_hits"] == 1 and row["first_seen"] == "2026-09-21"
    assert row["overlay"] == {"filters.heavy_favorite_american": -119} and row["kind"] == "overlay"
    assert row["evidence"][0]["run_date"] == "2026-09-21" and row["evidence"][0]["settled"] == 80
    assert row["latest_stats"]["clv_metric"] == "clv_pct"
    assert [e["event"] for e in ch[0].events] == ["seen"]
    assert "id" not in row


def test_reconcile_second_hit_with_clv_goes_pending():
    key = "ml-bets|fav_size=fav -120..-149"
    ch = reconcile.reconcile([_finding(key)], [_existing(key, "watch", 1)], "2026-09-21", CFG)[0]
    assert ch.status_before == "watch" and ch.status_after == "pending"
    assert ch.row["consecutive_hits"] == 2 and ch.row["confidence"] == "medium"
    assert [e["event"] for e in ch.events] == ["seen", "bar_met", "pending"]
    assert len(ch.row["evidence"]) == 2


def test_reconcile_bar_not_met_stays_watch_and_stale_run_no_hit():
    key = "ml-bets|fav_size=fav -120..-149"
    ch = reconcile.reconcile([_finding(key, clv_agrees=False)], [_existing(key, "watch", 1)],
                             "2026-09-21", CFG)[0]
    assert ch.status_after == "watch" and ch.row["consecutive_hits"] == 2
    # Settled count did not grow -> no hit counted.
    ch = reconcile.reconcile([_finding(key, settled=70)], [_existing(key, "watch", 1)],
                             "2026-09-21", CFG)[0]
    assert ch.status_after == "watch" and ch.row["consecutive_hits"] == 1
    # High confidence when Bonferroni and CLV agree.
    ch = reconcile.reconcile([_finding(key, bonf=True)], [_existing(key, "watch", 1)],
                             "2026-09-21", CFG)[0]
    assert ch.row["confidence"] == "high" and ch.status_after == "pending"


def test_reconcile_absent_twice_is_dropped():
    key = "ml-bets|fav_size=fav -120..-149"
    ch = reconcile.reconcile([], [_existing(key, "watch", 2, misses=0)], "2026-09-21", CFG)[0]
    assert ch.status_after == "watch" and ch.row["consecutive_misses"] == 1 and ch.row["consecutive_hits"] == 0
    ch = reconcile.reconcile([], [_existing(key, "watch", 0, misses=1)], "2026-09-28", CFG)[0]
    assert ch.status_after == "dropped" and [e["event"] for e in ch.events] == ["dropped"]
    # approved / promoted / rejected untouched when absent
    assert reconcile.reconcile([], [_existing(key, "approved", 3)], "2026-09-28", CFG) == []


def test_reconcile_snooze_expiry():
    key = "ml-bets|fav_size=fav -120..-149"
    ex = _existing(key, "snoozed", 2, extra={"snoozed_until": "2026-09-20"})
    ch = reconcile.reconcile([_finding(key)], [ex], "2026-09-21", CFG)[0]
    assert ch.status_after == "pending" and any(e["event"] == "unsnoozed" for e in ch.events)
    ex = _existing(key, "snoozed", 2, extra={"snoozed_until": "2026-10-20"})
    ch = reconcile.reconcile([_finding(key)], [ex], "2026-09-21", CFG)[0]
    assert ch.status_after == "snoozed"


def test_reconcile_calendar_dim_never_pending():
    key = "ml-bets|month=2026-08"
    ch = reconcile.reconcile([_finding(key, bonf=True)], [_existing(key, "watch", 3)], "2026-09-21", CFG)[0]
    assert ch.status_after == "watch" and ch.row["consecutive_hits"] == 4


def test_reconcile_rejected_reappearing_appends_evidence_only():
    key = "ml-bets|fav_size=fav -120..-149"
    for status in ("rejected", "dropped", "promoted", "approved"):
        ex = _existing(key, status, 0, extra={"title": "keep me", "decided_at": "2026-09-01T00:00:00Z"})
        ch = reconcile.reconcile([_finding(key, bonf=True)], [ex], "2026-09-21", CFG)[0]
        assert ch.status_after == status and ch.row["consecutive_hits"] == 0
        assert ch.row["title"] == "keep me" and ch.row["decided_at"] == "2026-09-01T00:00:00Z"
        assert len(ch.row["evidence"]) == 2 and [e["event"] for e in ch.events] == ["seen"]


def test_reconcile_overlap_folding():
    sharp = _finding("ml-bets|fav_size=fav -120..-149")
    mid = _finding("ml-bets|odds_bucket=small fav (-101..-149)")
    broad = _finding("ml-bets|side_type=favorite")
    other = _finding("ml-bets|side_type=underdog", direction="positive")   # not nested: kept
    ch = reconcile.reconcile([broad, sharp, mid, other], [], "2026-09-21", CFG)
    keys = {c.finding_key: c for c in ch}
    assert "ml-bets|fav_size=fav -120..-149" in keys
    assert "ml-bets|odds_bucket=small fav (-101..-149)" not in keys
    assert "ml-bets|side_type=favorite" not in keys
    assert "ml-bets|side_type=underdog" in keys and len(ch) == 2
    kept = keys["ml-bets|fav_size=fav -120..-149"].row
    corr = {c["key"] for c in kept["evidence"][-1]["corroborating_cells"]}
    assert corr == {"ml-bets|odds_bucket=small fav (-101..-149)", "ml-bets|side_type=favorite"}
    # A folded key that already has a proposal gets evidence only.
    ex = _existing("ml-bets|side_type=favorite", "watch", 2)
    ch = reconcile.reconcile([sharp, broad], [ex], "2026-09-21", CFG)
    folded = [c for c in ch if c.finding_key == "ml-bets|side_type=favorite"][0]
    assert folded.note == "folded" and folded.row["consecutive_hits"] == 2
    assert folded.row["evidence"][-1]["folded_into"] == "ml-bets|fav_size=fav -120..-149"


# --- suggest -------------------------------------------------------------------
def test_suggest_known_and_unknown():
    base = {"filters": {"heavy_favorite_american": -150}, "adjusted_ev": {"fragile_reduction": 0.01},
            "ev": {"threshold": 0.03}, "college": {"max_abs_spread": 28.0},
            "projections": {"max_spread_divergence_pts_cfb": 6.0}}
    st = {"settled": 101, "wins": 43, "losses": 58, "hit_rate": 0.426, "breakeven_hit_rate": 0.56,
          "flat_roi": -0.121, "avg_clv": -0.8, "clv_metric": "clv_pct", "clv_tracked": 95,
          "date_from": "2026-05-01", "date_to": "2026-09-20"}
    s = sg.suggest("mlb", "ml-bets", "fav_size", "fav -120..-149", "negative", base, st)
    assert s.kind == "overlay" and s.overlay == {"filters.heavy_favorite_american": -119}
    assert "n=101" in s.description and "-12.1%" in s.description and "-0.80%" in s.description
    assert "2026-05-01 to 2026-09-20" in s.description and "43%" in s.description
    assert sg.suggest("mlb", "ml-bets", "stability", "fragile", "negative", base, st).overlay == \
        {"adjusted_ev.fragile_reduction": 0.02}
    assert sg.suggest("mlb", "ml-bets", "model_prob_bucket", "<50%", "negative", base, st).overlay == \
        {"filters.min_model_prob": 0.5}
    assert sg.suggest("mlb", "ml-bets", "sharp_fade", "fade 4pp+", "negative", base, st).overlay == \
        {"sanity.max_sharp_disagreement_pp": 3.0}
    assert sg.suggest("mlb", "ml-passes", "ev_shortfall", "0-1pp short", "positive", base, st).overlay == \
        {"ev.threshold": 0.02}
    assert sg.suggest("mlb_totals", "totals-bets", "roof", "retractable_assumed_open", "negative", {}, st).overlay == \
        {"totals.weather.require_verified_roof": True}
    fb = sg.suggest("football", "fb-bets", "spread_band", "14-28", "negative", base, st, sport="cfb")
    assert fb.overlay == {"college.max_abs_spread": 14.0}
    assert sg.suggest("football", "fb-spread-holds", "hold_kind", "divergence guard", "positive", base, st).overlay == \
        {"projections.max_spread_divergence_pts_cfb": 7.5}
    # unmapped -> insight
    ins = sg.suggest("football", "fb-bets", "rest_bucket", "short (<6)", "negative", base, st)
    assert ins.kind == "insight" and ins.overlay is None and ins.suggested_change is None
    assert sg.suggest("mlb", "ml-bets", "venue_side", "home", "positive", base, st).kind == "insight"
    # already-filtered band -> insight (no looser overlay)
    assert sg.suggest("mlb", "ml-bets", "fav_size", "big fav -200+", "negative", base, st).kind == "insight"


# --- shadow gate ---------------------------------------------------------------
def _shadow(chal_n=80, chal_clv=1.2, champ_clv=0.8, hold_n=5, hold_chal=1.0, hold_champ=0.9, complete=True):
    return {"engine": "mlb",
            "champion": {"n_settled": 120, "avg_clv": champ_clv, "clv_metric": "clv_pct"},
            "challenger": {"n_settled": chal_n, "avg_clv": chal_clv, "clv_metric": "clv_pct"},
            "holdout": {"champion": {"n_settled": hold_n, "avg_clv": hold_champ},
                        "challenger": {"n_settled": hold_n, "avg_clv": hold_chal}},
            "window_complete": complete}


def test_shadow_gate_four_cases():
    ok = shadow_stats.gate(_shadow(), CFG, "mlb")
    assert ok == {"enabled": True, "reasons": []}
    small = shadow_stats.gate(_shadow(chal_n=3), CFG, "mlb")
    assert not small["enabled"] and any("needs 6" in r for r in small["reasons"])
    worse = shadow_stats.gate(_shadow(chal_clv=0.5), CFG, "mlb")
    assert not worse["enabled"] and any("does not beat" in r for r in worse["reasons"])
    hold = shadow_stats.gate(_shadow(hold_n=1), CFG, "mlb")
    assert not hold["enabled"] and "holdout sample too small" in hold["reasons"]
    hold2 = shadow_stats.gate(_shadow(hold_chal=0.5), CFG, "mlb")
    assert not hold2["enabled"] and any(r.startswith("holdout:") for r in hold2["reasons"])
    window = shadow_stats.gate(_shadow(complete=False), CFG, "mlb")
    assert not window["enabled"] and "shadow window not complete" in window["reasons"]


def test_shadow_compare_frames():
    champ = pd.DataFrame([_ml_view_row(i, -110, result="win" if i % 2 else "loss", clv=0.5) for i in range(1, 7)])
    chal = pd.DataFrame([{"date": "2026-06-01", "game_id": str(i), "market": "moneyline", "sport": "mlb",
                          "league": None, "result": "win", "clv": 1.0, "flat_pl": 0.9} for i in range(4, 10)])
    proposal = {"engine": "mlb", "challenger_tag": "biff_p1", "shadow_started_at": "2026-06-01",
                "shadow_ends_at": "2026-06-29"}
    from datetime import date
    out = shadow_stats.compare_frames(proposal, champ, chal, CFG, today=date(2026, 7, 1))
    assert out["champion"]["n_settled"] == 6 and out["challenger"]["n_settled"] == 6
    assert out["overlap"] == 3 and out["only_champion"] == 3 and out["only_challenger"] == 3
    assert out["window_complete"] is True and out["days_in_shadow"] == 30
    assert out["challenger"]["avg_clv"] == 1.0 and out["champion"]["clv_positive_share"] == 1.0
    assert "champion" in out["holdout"] and "challenger" in out["holdout"]


# --- versions ------------------------------------------------------------------
def test_rolling_check_flags_and_never_mutates_status():
    version = {"id": 1, "engine": "mlb", "tag": "biff_v1", "status": "active", "baseline": {"avg_clv": 1.0}}
    rows = pd.DataFrame([_ml_view_row(i, -110, result="loss", clv=-0.4, date=f"2026-06-{i:02d}")
                         for i in range(1, 9)])
    out = versions.rolling_check(version, CFG, rows=rows, write=False)
    assert out["rollback_flagged"] is True and "below floor" in out["rollback_reason"]
    assert "baseline" in out["rollback_reason"]
    assert out["rolling"]["n"] == 5 and out["rolling"]["window"] == 5      # last `window` settled only
    assert out["status"] == "active" and version["status"] == "active"
    assert "status" not in out["rolling"]
    good = pd.DataFrame([_ml_view_row(i, -110, result="win", clv=1.2, date=f"2026-06-{i:02d}") for i in range(1, 9)])
    out = versions.rolling_check(version, CFG, rows=good, write=False)
    assert out["rollback_flagged"] is False and out["rollback_reason"] is None
    # Baseline drop alone flags even with positive CLV.
    mid = pd.DataFrame([_ml_view_row(i, -110, result="win", clv=0.3, date=f"2026-06-{i:02d}") for i in range(1, 9)])
    out = versions.rolling_check(version, CFG, rows=mid, write=False)
    assert out["rollback_flagged"] is True and "baseline" in out["rollback_reason"]
    # Too few rows -> never flagged.
    tiny = pd.DataFrame([_ml_view_row(1, -110, result="loss", clv=-5.0)])
    assert versions.rolling_check(version, CFG, rows=tiny, write=False)["rollback_flagged"] is False


# --- report / email ------------------------------------------------------------
def test_render_email_content():
    result = SimpleNamespace(
        run_date="2026-09-21", dry_run=False, errors=[],
        sports=[SimpleNamespace(
            sport="mlb", engine="mlb", rows=500, settled=420, cells_tested=130,
            pools={"ml-bets": {"rows": 200, "settled": 180, "flat_roi": 0.01, "avg_clv": 0.4,
                               "clv_metric": "clv_pct"}},
            findings=[1, 2],
            changes=[SimpleNamespace(is_new=True, status_before=None, status_after="watch"),
                     SimpleNamespace(is_new=False, status_before="watch", status_after="pending"),
                     SimpleNamespace(is_new=False, status_before="watch", status_after="dropped")],
            proposals_after=[
                {"id": 3, "engine": "mlb", "sport": "mlb", "finding_key": "ml-bets|fav_size=fav -120..-149",
                 "title": "MLB moneyline bets: fav_size=fav -120..-149 is losing", "kind": "overlay",
                 "confidence": "medium", "status": "pending",
                 "latest_stats": {"settled": 101, "avg_clv": -0.8, "clv_metric": "clv_pct", "clv_tracked": 95,
                                  "flat_roi": -0.121, "hit_rate": 0.43, "date_from": "2026-05-01",
                                  "date_to": "2026-09-20"}},
                {"id": 4, "engine": "mlb", "status": "watch", "title": "not pending", "latest_stats": {}},
            ], error=None)],
        shadows=[{"proposal_id": 9, "title": "Totals roof hold", "engine": "mlb_totals", "challenger_tag": "totals_p9",
                  "champion": {"avg_clv": 0.2, "clv_metric": "clv_pp", "n_settled": 60, "flat_roi": 0.01},
                  "challenger": {"avg_clv": 0.5, "clv_metric": "clv_pp", "n_settled": 40, "flat_roi": 0.03},
                  "gate": {"enabled": False, "reasons": ["shadow window not complete"]}}],
        rollbacks=[{"tag": "biff_v2", "engine": "mlb", "rollback_flagged": True,
                    "rollback_reason": "rolling avg CLV -0.300 below floor +0.00 over last 75 settled picks",
                    "rolling": {"n": 75}},
                   {"tag": "totals_v1", "engine": "mlb_totals", "rollback_flagged": False}])
    summary = report.build_summary(result)
    sp = summary["sports"][0]
    assert (sp["new_watches"], sp["newly_pending"], sp["dropped"]) == (1, 1, 1)
    assert len(summary["pending"]) == 1 and len(summary["rollbacks"]) == 1
    subject, text = report.render_email(summary, "https://biffbet.com/proposals")
    assert "2026-09-21" in subject and "1 pending" in subject and "1 rollback" in subject
    assert "mlb: 500 rows analysed, 420 settled, 130 cells tested, 1 new watch(es), 1 newly pending, 1 dropped" in text
    pending_lines = [ln for ln in text.splitlines() if "fav_size=fav -120..-149" in ln]
    assert len(pending_lines) == 1
    assert "n=101" in pending_lines[0] and "CLV -0.80% (n=95)" in pending_lines[0]
    assert "ROI -12.1%" in pending_lines[0] and "hit 43%" in pending_lines[0]
    assert "2026-05-01 to 2026-09-20" in pending_lines[0]
    assert "not pending" not in text
    assert "Totals roof hold" in text and "CLV +0.50pp" in text and "gate closed: shadow window not complete" in text
    assert "biff_v2" in text and "totals_v1" not in text
    assert "https://biffbet.com/proposals" in text
    assert "python -m mlb_value_bot.rsi review --sport all" in text
    urls = [w for w in text.split() if w.startswith("http")]
    assert urls == ["https://biffbet.com/proposals"]


# --- import ledger -------------------------------------------------------------
def test_import_ledger_mapping(tmp_path):
    ledger = {"updated": "2026-09-15", "abilities": [
        {"key": "ml-bets|model_prob_bucket=<50%", "status": "active", "first_seen": "2026-08-30",
         "activated": "2026-08-30", "consecutive_hits": 1, "consecutive_misses": 0, "pool": "ml-bets",
         "dimension": "model_prob_bucket", "value": "<50%", "direction": "negative",
         "config_keys": ["filters.min_model_prob"], "proposal_doc": "docs/abilities/proposals/x.md",
         "kill_criteria": "retire if ...", "notes": "blended p < 50% loses.",
         "evidence": [{"run_date": "2026-08-30", "rows": 170, "settled": 24, "wins": 6, "losses": 18,
                       "hit_rate": 0.25, "flat_roi": -0.46, "t_stat": -2.36, "p_value": 0.02,
                       "avg_clv_pct": 2.03, "clv_tracked": 24, "bonferroni_significant": False,
                       "clv_agrees": False}]},
        {"key": "ml-passes|side_type=favorite", "status": "watch", "first_seen": "2026-08-10",
         "consecutive_hits": 4, "consecutive_misses": 0, "direction": "negative", "notes": "Passed favorites lose.",
         "evidence": [{"run_date": "2026-09-15", "settled": 300}]},
        {"key": "totals-passes|pick_side=over", "status": "dropped", "first_seen": "2026-08-31",
         "dropped": "2026-09-14", "consecutive_hits": 0, "consecutive_misses": 2, "direction": "negative",
         "notes": "n", "evidence": []},
        {"key": "fb-model|cfb-margin-anchor", "status": "active", "first_seen": "2026-08-31",
         "activated": "2026-08-31", "consecutive_hits": 1, "consecutive_misses": 0, "pool": "fb-bets",
         "config_keys": ["projections.margin_anchor_cfb", "projections.elo_points_per_margin"],
         "notes": "CFB margin anchor.", "evidence": []},
        {"key": "fb-model|nfl-ats-epa-anchor", "status": "rejected", "rejected": "2026-09-01",
         "config_keys": ["projections.margin_anchor_nfl"], "notes": "rejected", "evidence": []},
        {"key": "fb-spread-holds|hold_kind=divergence guard", "status": "retire-recommended",
         "first_seen": "2026-08-31", "consecutive_hits": 1, "consecutive_misses": 0, "notes": "x", "evidence": []},
    ]}
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    (proposals / "x.md").write_text("# Ability: min-model-prob\n\nbody text\n", encoding="utf-8")
    configs = {"mlb": {"filters": {"min_model_prob": 0.5}}, "mlb_totals": {},
               "football": {"projections": {"margin_anchor_cfb": "elo", "elo_points_per_margin": 23.0,
                                            "margin_anchor_nfl": None}}, "griffbet": {}}
    rows = build_rows(ledger, proposals, configs, {"biff_v1": 1, "totals_v1": 2, "matchup_v1": 3})
    by = {r["finding_key"]: r for r in rows}
    a = by["ml-bets|model_prob_bucket=<50%"]
    assert a["engine"] == "mlb" and a["status"] == "promoted" and a["version_id"] == 1
    assert a["overlay"] == {"filters.min_model_prob": 0.5} and a["kind"] == "overlay"
    assert a["title"] == "Ability: min-model-prob" and "body text" in a["description"]
    assert a["consecutive_hits"] == 1 and a["first_seen"] == "2026-08-30" and a["last_seen"] == "2026-08-30"
    assert a["latest_stats"]["settled"] == 24 and a["latest_stats"]["avg_clv"] == 2.03
    assert a["evidence"][0]["run_date"] == "2026-08-30" and a["evidence"][-1]["kill_criteria"] == "retire if ..."
    assert a["decided_at"] == "2026-08-30T00:00:00Z"
    w = by["ml-passes|side_type=favorite"]
    assert w["engine"] == "mlb" and w["status"] == "watch" and w["consecutive_hits"] == 4 and w["kind"] == "insight"
    assert w["last_seen"] == "2026-09-15" and w["title"] == "Passed favorites lose"
    assert by["totals-passes|pick_side=over"]["engine"] == "mlb_totals"
    assert by["totals-passes|pick_side=over"]["status"] == "dropped"
    fb = by["fb-model|cfb-margin-anchor"]
    assert fb["engine"] == "football" and fb["sport"] == "cfb" and fb["status"] == "promoted"
    assert fb["version_id"] == 3 and fb["overlay"] == {"projections.margin_anchor_cfb": "elo",
                                                       "projections.elo_points_per_margin": 23.0}
    rej = by["fb-model|nfl-ats-epa-anchor"]
    assert rej["status"] == "rejected" and rej["sport"] == "nfl" and rej["overlay"] == {"projections.margin_anchor_nfl": None}
    assert by["fb-spread-holds|hold_kind=divergence guard"]["status"] == "pending"
    assert len({(r["engine"], r["finding_key"]) for r in rows}) == len(rows)


# --- review (end to end, no I/O) ----------------------------------------------
def test_review_sport_end_to_end_with_preregistered_test():
    random.seed(1)
    rows = []
    # 12 small favorites that lose (pre-registered cell), 12 big dogs that win.
    for i in range(1, 13):
        rows.append(_ml_view_row(i, -110, result="loss" if i != 12 else "win", clv=-0.9,
                                 date=f"2026-06-{i:02d}"))
    for i in range(13, 25):
        rows.append(_ml_view_row(i, 220, result="win" if i != 24 else "loss", clv=1.1,
                                 date=f"2026-07-{i - 12:02d}"))
    # settled passes (counterfactual) that would have won
    for i in range(25, 33):
        rows.append(_ml_view_row(i, 130, result="win", clv=None, decision="pass",
                                 pass_reason="below_threshold", date=f"2026-08-{i - 24:02d}"))
    # holdout rows must be ignored
    rows.append({**_ml_view_row(99, -110, result="loss"), "is_holdout": True})
    df = pd.DataFrame(rows)
    sr = review_sport("mlb", df, [], CFG, "2026-09-21")
    assert sr.rows == 32 and sr.settled == 32
    keys = {c.key: c for c in sr.findings}
    pre = keys.get("ml-bets|fav_size=small favs -101..-149")
    assert pre is not None and pre.preregistered and pre.direction == "negative"
    assert pre.bonferroni_significant is True          # own p-value, no correction
    assert pre.suggestion["overlay"] == {"filters.heavy_favorite_american": -101}
    assert "ml-bets|side_type=underdog" in keys or "ml-bets|fav_size=big dog +200+" in keys
    # every candidate got deltas vs its pool baseline and a suggestion
    for c in sr.findings:
        assert c.suggestion is not None and c.cells_tested == sr.cells_tested
        if not c.preregistered:
            assert c.delta_roi is not None
    assert "ml-bets" in sr.pools and "ml-passes" in sr.pools
    assert sr.pools["ml-bets"]["settled"] == 24 and sr.pools["ml-passes"]["settled"] == 8
    assert all(ch.is_new and ch.status_after == "watch" for ch in sr.changes)
    # Nested price cells collapsed onto the sharpest key.
    assert not any(ch.finding_key == "ml-bets|side_type=favorite" for ch in sr.changes) or \
        not any(ch.finding_key.startswith("ml-bets|fav_size=") and "fav" in ch.finding_key for ch in sr.changes)
    summary = report.build_summary(SimpleNamespace(run_date="2026-09-21", sports=[sr], dry_run=True,
                                                   shadows=[], rollbacks=[], errors=[]))
    assert summary["sports"][0]["new_watches"] == len(sr.changes)
    subject, text = report.render_email(summary, "https://biffbet.com/proposals")
    assert "[dry run]" in subject


def test_review_sport_empty_frame_is_fine():
    sr = review_sport("football", pd.DataFrame(), [], CFG, "2026-09-21")
    assert sr.rows == 0 and sr.findings == [] and sr.changes == []
