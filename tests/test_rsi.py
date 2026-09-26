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


# =============================================================================
# Runtime half (2026-09-26): state / effective config / shadow picks / grading.
# Fixtures only: every Supabase call goes through _MemSupa (rsi.supa is
# monkeypatched), the MLB client and football finals are stubs.
# =============================================================================
from mlb_value_bot.rsi import config as rsi_config
from mlb_value_bot.rsi import shadow as rsi_shadow
from mlb_value_bot.rsi import shadow_grade
from mlb_value_bot.rsi import state as rsi_state
from mlb_value_bot.rsi import supa as rsi_supa
from mlb_value_bot.rsi.state import RsiState, ShadowProposal


class _MemSupa:
    """In-memory stand-in for rsi.supa (get_rows / upsert_rows / patch_rows)."""

    def __init__(self, tables=None):
        self.tables = {k: [dict(r) for r in v] for k, v in (tables or {}).items()}
        self.upserts: list[list[dict]] = []
        self.patches: list[tuple] = []
        self._next_id = 1000

    @staticmethod
    def _match(row, filters):
        for col, spec in (filters or {}).items():
            op, val = spec if isinstance(spec, tuple) else ("eq", spec)
            v = row.get(col)
            if op == "eq" and v != val:
                return False
            if op == "in" and v not in val:
                return False
            if op == "lt" and not str(v) < str(val):
                return False
            if op == "gte" and not str(v) >= str(val):
                return False
        return True

    def get_rows(self, table, filters=None, select="*", order=None, limit=None):
        rows = [dict(r) for r in self.tables.get(table, []) if self._match(r, filters)]
        return rows[:limit] if limit else rows

    def upsert_rows(self, table, rows, on_conflict):
        keys = on_conflict.split(",")
        store = self.tables.setdefault(table, [])
        self.upserts.append([dict(r) for r in rows])
        for r in rows:
            k = tuple(str(r.get(c)) for c in keys)
            for ex in store:
                if tuple(str(ex.get(c)) for c in keys) == k:
                    ex.update(r)
                    break
            else:
                self._next_id += 1
                store.append({"id": self._next_id, "result": "pending", **r})
        return len(rows)

    def patch_rows(self, table, filters, fields):
        for r in self.tables.get(table, []):
            if self._match(r, filters):
                r.update(fields)
                self.patches.append((r["id"], dict(fields)))

    def install(self, monkeypatch):
        monkeypatch.setattr(rsi_supa, "get_rows", self.get_rows)
        monkeypatch.setattr(rsi_supa, "upsert_rows", self.upsert_rows)
        monkeypatch.setattr(rsi_supa, "patch_rows", self.patch_rows)
        return self

    def row(self, table, **where):
        for r in self.tables.get(table, []):
            if all(str(r.get(k)) == str(v) for k, v in where.items()):
                return r
        return None


def _supa_down(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("supabase unreachable")
    monkeypatch.setattr(rsi_supa, "get_rows", _raise)
    monkeypatch.setattr(rsi_supa, "upsert_rows", _raise)
    monkeypatch.setattr(rsi_supa, "patch_rows", _raise)


def _ga(gid, side="home", odds=-110, ev_pct=0.05, kelly=0.01, filters=(), skipped=None,
        date="2026-06-10"):
    from mlb_value_bot.analysis.ev_calculator import SideEvaluation, american_to_decimal
    from mlb_value_bot.pipeline import GameAnalysis

    se = SideEvaluation(side=side, american_odds=odds, decimal_odds=american_to_decimal(odds),
                        model_prob=0.55, market_prob_raw=0.52, market_prob_devigged=0.50,
                        ev_pct=ev_pct, kelly_stake=kelly)
    return GameAnalysis(game_id=gid, game_date=date, home_team="H", away_team="A",
                        status="Scheduled", home_pitcher=None, away_pitcher=None,
                        evals={side: se}, best_side=side, confidence=60.0,
                        skipped_reason=skipped, adjusted_ev_pct=ev_pct,
                        filter_reasons=list(filters))


def _ta(gid, side="over", odds=-110, ev_pct=0.05, kelly=0.01, line=8.5, devig_over=0.52,
        sharp_over=0.55):
    """A TotalsAnalysis-shaped stub (only what save_shadow_mlb reads)."""
    from mlb_value_bot.analysis.ev_calculator import SideEvaluation, american_to_decimal

    se = SideEvaluation(side=side, american_odds=odds, decimal_odds=american_to_decimal(odds),
                        model_prob=0.56, market_prob_raw=0.52, market_prob_devigged=devig_over,
                        ev_pct=ev_pct, kelly_stake=kelly)
    ns = SimpleNamespace(game_id=gid, home_team="H", away_team="A", best_eval=se, rd=object(),
                         intel=SimpleNamespace(best_over_price=odds, best_under_price=-105, bet_line=line),
                         pick_side=side, market_total=line, confidence=55.0)
    ns.opening_devig_for = lambda s: devig_over if s == "over" else 1.0 - devig_over
    ns.sharp_close_devig_for = lambda s: sharp_over if s == "over" else 1.0 - sharp_over
    ns.is_value = lambda thr: ev_pct >= thr and kelly > 0
    ns.pass_reason = lambda thr: None if ns.is_value(thr) else "below_threshold"
    ns.reasoning = lambda: {"model_tag": "stub"}
    return ns


def _fb_pick(market="spread", side="home", line=-3.0, is_value=True, odds=-110, sharp_p=0.52,
             hold=None):
    from mlb_value_bot.football.pipeline_football import FootballPick

    return FootballPick(
        market=market, side=side, line=line, american_odds=odds, model_prob=0.55,
        market_prob=0.5, p_push=0.02, raw_ev=0.04, adjusted_ev=0.04, adjustments=[],
        confidence=70.0, tier="standard" if is_value else "pass",
        stake_pct=0.01 if is_value else 0.0, stability_label="stable", is_value=is_value,
        hold_reason=hold,
        reasoning={"market": {"devig_p_a": 0.50, "sharp_devig_p_a": sharp_p, "sharp_line": line},
                   "matchup": {"home_edge": 20.0, "archetype": "neutral"},
                   "projection": {"margin": 4.0, "total": 44.0}})


def _fb_analysis(picks, game_id="2026_01_A_H", date="2026-09-13"):
    from mlb_value_bot.football.pipeline_football import FootballGameAnalysis

    return FootballGameAnalysis(league="nfl", date=date, week=1, game_id=game_id, home="H",
                                away="A", commence_time=f"{date}T17:00:00Z", picks=picks)


# --- state ---------------------------------------------------------------------
def test_load_state_default_cache_and_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(rsi_state, "STATE_DIR", tmp_path)
    _supa_down(monkeypatch)
    st = rsi_state.load_state("mlb", cfg={"max_shadows": 2})
    assert (st.active_tag, st.active_overlay, st.shadow, st.source) == ("biff_v1", {}, [], "default")
    assert rsi_state.load_state("mlb_totals").active_tag == "totals_v1"
    assert rsi_state.load_state("football").active_tag == "matchup_v1"
    assert rsi_state.load_state("nope").active_tag == "nope_v1"

    mem = _MemSupa({
        "rsi_model_versions": [
            {"id": 1, "engine": "mlb", "tag": "biff_v1", "status": "superseded", "overlay": {}},
            {"id": 2, "engine": "mlb", "tag": "biff_v2", "status": "active",
             "overlay": {"ev.threshold": 0.04}}],
        "rsi_proposals": [
            {"id": 12, "engine": "mlb", "sport": "mlb", "status": "approved", "challenger_tag": "biff_p12",
             "overlay": {"filters.min_model_prob": 0.55}, "decided_at": "2026-09-20T10:00:00Z",
             "shadow_started_at": "2026-09-20", "shadow_ends_at": "2026-10-18"},
            {"id": 10, "engine": "mlb", "sport": "mlb", "status": "approved", "challenger_tag": "biff_p10",
             "overlay": {"ev.threshold": 0.035}, "decided_at": "2026-09-10T10:00:00Z"},
            {"id": 11, "engine": "mlb", "sport": "mlb", "status": "approved", "challenger_tag": "biff_p11",
             "overlay": {"model.market_blend": 0.3}, "decided_at": "2026-09-15T10:00:00Z"},
            {"id": 13, "engine": "mlb", "sport": "mlb", "status": "approved", "challenger_tag": None,
             "overlay": {"ev.threshold": 0.02}},                       # no tag -> unusable
            {"id": 14, "engine": "mlb", "sport": "mlb", "status": "approved", "challenger_tag": "biff_p14",
             "overlay": None},                                          # insight -> unusable
            {"id": 15, "engine": "mlb", "sport": "mlb", "status": "pending", "challenger_tag": "biff_p15",
             "overlay": {"ev.threshold": 0.02}},                       # not approved
            {"id": 16, "engine": "football", "sport": "nfl", "status": "approved",
             "challenger_tag": "matchup_p16", "overlay": {"ev.threshold": 0.02}}],
    }).install(monkeypatch)
    st = rsi_state.load_state("mlb", cfg={"max_shadows": 2})
    assert st.source == "supabase" and st.active_tag == "biff_v2"
    assert st.active_overlay == {"ev.threshold": 0.04}
    assert [p.challenger_tag for p in st.shadow] == ["biff_p10", "biff_p11"]   # oldest first, capped
    assert st.shadow[0].overlay == {"ev.threshold": 0.035} and st.shadow[0].sport == "mlb"
    cache = tmp_path / "state_mlb.json"
    assert cache.exists()
    assert rsi_state.load_state("football", cfg={"max_shadows": 3}).shadow[0].id == 16

    # Supabase down again -> the cache serves the last good state.
    _supa_down(monkeypatch)
    st2 = rsi_state.load_state("mlb", cfg={"max_shadows": 2})
    assert st2.source == "cache" and st2.active_tag == "biff_v2"
    assert [p.challenger_tag for p in st2.shadow] == ["biff_p10", "biff_p11"]
    assert st2.shadow[0].shadow_started_at is None
    # A corrupt cache degrades to the default rather than raising.
    cache.write_text("{not json", encoding="utf-8")
    assert rsi_state.load_state("mlb").source == "default"


# --- effective / challenger config ------------------------------------------------
def _fake_states(**overrides):
    base = {
        "mlb": RsiState("mlb", "biff_v2", {"ev.threshold": 0.045}, [], "supabase"),
        "mlb_totals": RsiState("mlb_totals", "totals_v2", {"totals.ev_threshold": 0.05}, [], "supabase"),
        "football": RsiState("football", "matchup_v2", {"ev.threshold": 0.04}, [], "supabase"),
    }
    base.update(overrides)
    return lambda engine, cfg=None: base[engine]


def test_effective_config_applies_overlay_stamps_tag_and_degrades(monkeypatch):
    import copy

    from mlb_value_bot.football import load_football_config
    from mlb_value_bot.utils import load_config

    base = copy.deepcopy(load_config())
    monkeypatch.setattr(rsi_config, "load_state", _fake_states())
    cfg, tag = rsi_config.effective_config("mlb")
    assert tag == "biff_v2" and cfg["ev"]["threshold"] == 0.045
    assert cfg["totals"]["ev_threshold"] == 0.05                     # both MLB overlays applied
    assert cfg["rsi"] == {"tag": "biff_v2", "totals_tag": "totals_v2"}
    assert load_config() == base                                     # the cached base is untouched
    cfg_t, tag_t = rsi_config.effective_config("mlb_totals")
    assert tag_t == "totals_v2" and cfg_t["ev"]["threshold"] == 0.045
    fcfg, ftag = rsi_config.effective_config("football")
    assert ftag == "matchup_v2" and fcfg["model_tag"] == "matchup_v2" and fcfg["ev"]["threshold"] == 0.04

    # No overlay at all -> the champion config IS the yaml (plus the tag bookkeeping).
    monkeypatch.setattr(rsi_config, "load_state", _fake_states(
        mlb=RsiState("mlb", "biff_v1", {}, []), mlb_totals=RsiState("mlb_totals", "totals_v1", {}, []),
        football=RsiState("football", "matchup_v1", {}, [])))
    cfg, tag = rsi_config.effective_config("mlb")
    assert tag == "biff_v1" and {k: v for k, v in cfg.items() if k != "rsi"} == base
    fcfg, ftag = rsi_config.effective_config("football")
    fbase = load_football_config()
    assert {k: v for k, v in fcfg.items() if k != "model_tag"} == {k: v for k, v in fbase.items() if k != "model_tag"}

    # Degrade path: load_state blowing up (it never should) -> base + default tags.
    def _boom(engine, cfg=None):
        raise RuntimeError("boom")
    monkeypatch.setattr(rsi_config, "load_state", _boom)
    cfg, tag = rsi_config.effective_config("mlb")
    assert tag == "biff_v1" and cfg["ev"]["threshold"] == base["ev"]["threshold"]
    assert cfg["rsi"] == {"tag": "biff_v1", "totals_tag": "totals_v1"}
    fcfg, ftag = rsi_config.effective_config("football")
    assert ftag == "matchup_v1" and fcfg["model_tag"] == "matchup_v1"


def test_challenger_config_retags_by_sport():
    cfg = {"ev": {"threshold": 0.03}, "totals": {"ev_threshold": 0.03},
           "rsi": {"tag": "biff_v1", "totals_tag": "totals_v1"}}
    ml = rsi_config.challenger_config(cfg, ShadowProposal(1, "biff_p1", {"ev.threshold": 0.05}, "mlb"))
    assert ml["ev"]["threshold"] == 0.05 and ml["rsi"] == {"tag": "biff_p1", "totals_tag": "totals_v1"}
    assert cfg["ev"]["threshold"] == 0.03                              # deep copy
    tot = rsi_config.challenger_config(cfg, ShadowProposal(2, "totals_p2", {"totals.ev_threshold": 0.04}, "mlb_totals"))
    assert tot["totals"]["ev_threshold"] == 0.04 and tot["rsi"] == {"tag": "biff_v1", "totals_tag": "totals_p2"}
    fb = rsi_config.challenger_config({"model_tag": "matchup_v1", "ev": {"threshold": 0.03}},
                                      ShadowProposal(3, "matchup_p3", {"ev.threshold": 0.02}, "nfl"))
    assert fb["model_tag"] == "matchup_p3" and fb["ev"]["threshold"] == 0.02


# --- shadow picks: MLB -------------------------------------------------------------
def test_save_shadow_mlb_shapes_rows_and_freezes(monkeypatch):
    from mlb_value_bot.tracking.recommendations import _compute_clv

    mem = _MemSupa().install(monkeypatch)
    cfg = {"ev": {"threshold": 0.02}, "totals": {"ev_threshold": 0.03}}
    champion = [_ga(1, ev_pct=0.05, kelly=0.01), _ga(2, ev_pct=0.025, kelly=0.004),
                _ga(3, side="home", ev_pct=0.01, kelly=0.0)]
    challenger = [_ga(1, ev_pct=0.05, kelly=0.01), _ga(2, ev_pct=0.025, kelly=0.004),
                  _ga(3, side="home", ev_pct=0.01, kelly=0.0),
                  _ga(4, ev_pct=0.05, kelly=0.0, skipped="raw model vs market diverge by 0.3")]
    n = rsi_shadow.save_shadow_mlb("biff_p1", "mlb", challenger, champion, 0.02, "2026-06-10", cfg,
                                   champion_threshold=0.03)
    assert n == 4
    r1 = mem.row("rsi_shadow_picks", game_id="1")
    assert r1["challenger_tag"] == "biff_p1" and r1["engine"] == "mlb" and r1["sport"] == "mlb"
    assert r1["game_id"] == "1" and r1["market"] == "moneyline" and r1["clv_metric"] == "clv_pct"
    assert r1["is_value"] is True and r1["pass_reason"] is None and r1["champion_is_value"] is True
    assert r1["opening_price"] == -110 and r1["closing_price"] == -110 and r1["clv"] == 0.0
    assert r1["bet_odds"] == -110 and r1["line"] is None and r1["reasoning"]["model_tag"] == "biff_v1"
    r2 = mem.row("rsi_shadow_picks", game_id="2")
    assert r2["is_value"] is True and r2["champion_is_value"] is False   # 2.5% clears 2% not 3%
    r3 = mem.row("rsi_shadow_picks", game_id="3")
    assert r3["is_value"] is False and r3["pass_reason"] == "below_threshold" and r3["champion_is_value"] is False
    r4 = mem.row("rsi_shadow_picks", game_id="4")
    assert r4["pass_reason"] == "skip:divergence" and r4["champion_is_value"] is None
    assert all("result" not in r and "flat_pl" not in r for r in mem.upserts[0])   # grading owns them

    # Re-run at new prices: value row keeps its opening, refreshes the close +
    # CLV; a pass keeps its opening unless the side flips (re-freeze).
    challenger2 = [_ga(1, odds=-130, ev_pct=0.04, kelly=0.008),
                   _ga(3, side="away", odds=120, ev_pct=0.01, kelly=0.0),
                   _ga(2, odds=-115, ev_pct=0.01, kelly=0.0)]
    rsi_shadow.save_shadow_mlb("biff_p1", "mlb", challenger2, champion, 0.02, "2026-06-10", cfg)
    r1 = mem.row("rsi_shadow_picks", game_id="1")
    assert r1["opening_price"] == -110 and r1["closing_price"] == -130
    assert r1["clv"] == _compute_clv(-110, -130) and r1["clv"] > 0
    assert r1["is_value"] is True and r1["bet_odds"] == -110            # frozen commit
    r3 = mem.row("rsi_shadow_picks", game_id="3")
    assert r3["pick_side"] == "away" and r3["opening_price"] == 120     # flipped -> re-frozen
    r2 = mem.row("rsi_shadow_picks", game_id="2")
    assert r2["is_value"] is True and r2["opening_price"] == -110        # never downgraded
    assert mem.row("rsi_shadow_picks", game_id="4")["result"] == "pending"

    # Totals rows read .totals off the GameAnalysis (or a TotalsAnalysis directly).
    g = _ga(5)
    g.totals = _ta(5, side="under", line=8.5, devig_over=0.48, sharp_over=0.45)
    n = rsi_shadow.save_shadow_mlb("totals_p2", "mlb_totals", [g, _ga(6)], [g], None, "2026-06-10", cfg)
    assert n == 1
    t = mem.row("rsi_shadow_picks", game_id="5", sport="mlb_totals")
    assert t["engine"] == "mlb_totals" and t["market"] == "total" and t["clv_metric"] == "clv_pp"
    assert t["line"] == 8.5 and t["opening_line"] == 8.5 and t["closing_price"] == -105
    assert t["opening_devig_p_side"] == pytest.approx(0.52) and t["sharp_close_devig_p_side"] == pytest.approx(0.55)
    assert t["clv"] == pytest.approx(3.0) and t["is_value"] is True and t["champion_is_value"] is True

    # Best-effort: a failing upsert logs and returns 0, never raises.
    def _boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(rsi_supa, "upsert_rows", _boom)
    assert rsi_shadow.save_shadow_mlb("biff_p1", "mlb", challenger, champion, 0.02, "2026-06-10", cfg) == 0
    assert rsi_shadow.save_shadow_mlb("biff_p1", "hockey", challenger, champion, 0.02, "2026-06-10", cfg) == 0


# --- shadow picks: football -------------------------------------------------------
def test_save_shadow_football_shapes_rows_and_freezes(monkeypatch):
    mem = _MemSupa().install(monkeypatch)
    cfg = {"model_tag": "matchup_p3"}
    champ = [_fb_analysis([_fb_pick(is_value=False, hold="below threshold"),
                           _fb_pick(market="total", side="under", line=44.5, is_value=True)])]
    chal = [_fb_analysis([_fb_pick(sharp_p=0.52),
                          _fb_pick(market="total", side="under", line=44.5, is_value=False,
                                   hold="divergence guard")]),
            _fb_analysis([_fb_pick(side="away", line=3.0, odds=-105)], game_id="2026_01_B_C",
                         date="2026-09-14")]
    n = rsi_shadow.save_shadow_football("matchup_p3", "nfl", chal, champ, cfg)
    assert n == 3
    s = mem.row("rsi_shadow_picks", game_id="2026_01_A_H", market="spread")
    assert s["engine"] == "football" and s["sport"] == "nfl" and s["league"] == "nfl"
    assert s["is_value"] is True and s["champion_is_value"] is False and s["pass_reason"] is None
    assert s["line"] == -3.0 and s["opening_line"] == -3.0 and s["opening_devig_p_side"] == 0.5
    assert s["sharp_close_devig_p_side"] == 0.52 and s["clv"] == pytest.approx(2.0)
    assert s["clv_metric"] == "clv_pp" and s["decimal_odds"] == pytest.approx(1.909, abs=1e-3)
    t = mem.row("rsi_shadow_picks", game_id="2026_01_A_H", market="total")
    assert t["is_value"] is False and t["pass_reason"] == "divergence guard" and t["champion_is_value"] is True
    assert t["line"] == 44.5
    away = mem.row("rsi_shadow_picks", game_id="2026_01_B_C", market="spread")
    assert away["date"] == "2026-09-14" and away["line"] == -3.0        # picked-side line (home -3 -> away +3 stored as picked)
    assert away["champion_is_value"] is None

    # Later run: line moved and the sharps came to us -> opening frozen, close + CLV refreshed.
    chal2 = [_fb_analysis([_fb_pick(line=-3.5, odds=-115, sharp_p=0.55),
                           _fb_pick(market="total", side="over", line=44.0, is_value=False, hold="x")])]
    rsi_shadow.save_shadow_football("matchup_p3", "nfl", chal2, champ, cfg)
    s = mem.row("rsi_shadow_picks", game_id="2026_01_A_H", market="spread")
    assert s["opening_line"] == -3.0 and s["opening_price"] == -110 and s["opening_devig_p_side"] == 0.5
    assert s["closing_line"] == -3.5 and s["closing_price"] == -115
    assert s["sharp_close_devig_p_side"] == 0.55 and s["clv"] == pytest.approx(5.0)
    t = mem.row("rsi_shadow_picks", game_id="2026_01_A_H", market="total")
    assert t["pick_side"] == "over" and t["opening_line"] == 44.0        # side flipped -> re-frozen
    assert rsi_shadow.save_shadow_football("matchup_p3", "nfl", [], champ, cfg) == 0


# --- grading -------------------------------------------------------------------------
def test_grade_shadow_mlb_and_football_outcomes(monkeypatch):
    from mlb_value_bot.data.mlb_client import GameResult
    from mlb_value_bot.football.tracking import football_results as fr

    def _row(i, sport, market, side, gid, date, line=None, dec=1.91, league=None):
        return {"id": i, "sport": sport, "engine": "football" if league else sport, "league": league,
                "market": market, "pick_side": side, "game_id": gid, "date": date, "line": line,
                "decimal_odds": dec, "result": "pending"}

    mem = _MemSupa({"rsi_shadow_picks": [
        _row(1, "mlb", "moneyline", "home", "10", "2026-06-01"),
        _row(2, "mlb", "moneyline", "away", "11", "2026-06-01", dec=2.4),
        _row(3, "mlb", "moneyline", "home", "12", "2026-06-01"),          # postponed -> void
        _row(4, "mlb_totals", "total", "over", "13", "2026-06-01", line=8.5),
        _row(5, "mlb_totals", "total", "under", "14", "2026-06-01", line=9.0),   # exact -> push
        _row(6, "mlb", "moneyline", "home", "15", "2026-06-01"),          # no final yet
        _row(7, "mlb", "moneyline", "home", "16", "2026-09-30"),          # future: not in scope
        _row(8, "nfl", "spread", "home", "g1", "2026-09-13", line=-3.0, league="nfl"),
        _row(9, "nfl", "total", "over", "g2", "2026-09-13", line=44.5, league="nfl"),
        _row(10, "cfb", "spread", "away", "c1", "2026-09-01", line=3.0, league="cfb"),  # no final, old -> void
        _row(11, "nfl", "spread", "away", "g3", "2026-09-13", line=3.0, league="nfl"),  # exact -> push
    ]}).install(monkeypatch)

    class _MLB:
        def get_results(self, d):
            assert d == "2026-06-01"
            return [GameResult(10, "Final", "H", "A", 5, 3), GameResult(11, "Final", "H", "A", 5, 3),
                    GameResult(12, "Postponed", "H", "A", None, None),
                    GameResult(13, "Final", "H", "A", 5, 4), GameResult(14, "Final", "H", "A", 5, 4)]

    out = shadow_grade.grade_shadow("mlb", before="2026-09-26", mlb_client=_MLB())
    assert out["rows"] == 6 and out["graded"] == 5 and out["pending"] == 1
    assert (out["win"], out["loss"], out["push"], out["void"]) == (2, 1, 1, 1)
    got = {i: (f["result"], f["flat_pl"]) for i, f in mem.patches}
    assert got[1] == ("win", pytest.approx(0.91)) and got[2] == ("loss", -1.0)
    assert got[3] == ("void", 0.0) and got[4] == ("win", pytest.approx(0.91)) and got[5] == ("push", 0.0)
    assert 6 not in got and 7 not in got
    assert mem.row("rsi_shadow_picks", id=1)["result"] == "win"

    mem.patches.clear()
    monkeypatch.setattr(fr, "_nfl_finals", lambda season, cfg: {"g1": (27, 20), "g2": (20, 24), "g3": (23, 20)})
    monkeypatch.setattr(fr, "_cfb_finals", lambda season, cfg: {})
    out = shadow_grade.grade_shadow("football", before="2026-09-26",
                                    football_config={"grading": {"void_after_days": 10}})
    assert out["rows"] == 4 and out["graded"] == 4
    got = {i: (f["result"], f["flat_pl"]) for i, f in mem.patches}
    assert got[8] == ("win", pytest.approx(0.91)) and got[9] == ("loss", -1.0)
    assert got[10] == ("void", 0.0) and got[11] == ("push", 0.0)
    with pytest.raises(ValueError):
        shadow_grade.grade_shadow("hockey")
    assert shadow_grade.flat_pl("win", 2.5) == 1.5 and shadow_grade.flat_pl("pending", 2.5) is None


# --- end to end: a challenger re-prices the SAME inputs and differs only by its overlay --
def test_shadow_differs_from_champion_when_overlay_changes_threshold(monkeypatch):
    """Reuses test_core's offline golden slate: fetch once, evaluate the
    champion, then a challenger whose overlay raises ev.threshold to 5%.
    Game 2 (EV ~3.5%) is a pass either way, but ONLY the challenger's
    pass_reason carries below_threshold; the champion output is untouched."""
    import copy

    import mlb_value_bot.pipeline as P
    from mlb_value_bot.analysis.team_metrics import TeamProfile
    from mlb_value_bot.tests.test_core import _serialize_slate, _slate_fixture
    from mlb_value_bot.utils import load_config

    schedule, odds = _slate_fixture()

    class _StubOdds:
        def get_odds(self):
            return list(odds)

    class _StubMLB:
        def get_schedule(self, d):
            return list(schedule)

        def get_per_player_hitting(self, season):
            return {}

        def get_per_pitcher_reliever_stats(self, season):
            return {}

    class _StubProvider:
        def __init__(self, season=None, config=None, mlb_client=None):
            pass

        def build_team_profile(self, name, is_home):
            return TeamProfile(team=name, raw_winpct=0.52 if is_home else 0.48, games=60,
                               wins=31, losses=29, offense_wrc_plus=102, bullpen_fip=4.1,
                               park_factor=100)

    monkeypatch.setattr(P, "TeamMetricsProvider", _StubProvider)
    monkeypatch.setattr(rsi_config, "load_state", _fake_states(
        mlb=RsiState("mlb", "biff_v1", {}, []), mlb_totals=RsiState("mlb_totals", "totals_v1", {}, [])))
    cfg, tag = rsi_config.effective_config("mlb")
    cfg["totals"]["enabled"] = False
    assert tag == "biff_v1"
    plain = copy.deepcopy(load_config())
    plain["totals"]["enabled"] = False

    inputs = P.fetch_slate_inputs("2026-06-10", _StubOdds(), _StubMLB(), cfg)
    champion = P.evaluate_slate_inputs(inputs, cfg)
    # The champion config prices exactly what the plain yaml prices.
    assert _serialize_slate(champion) == _serialize_slate(P.evaluate_slate_inputs(inputs, plain))
    champ_thr = float(cfg["ev"]["threshold"])
    assert champ_thr == 0.03

    proposal = ShadowProposal(7, "biff_p7", {"ev.threshold": 0.05}, "mlb")
    chal_cfg = rsi_config.challenger_config(cfg, proposal)
    assert chal_cfg["rsi"]["tag"] == "biff_p7" and chal_cfg["ev"]["threshold"] == 0.05
    challenger = P.evaluate_slate_inputs(inputs, chal_cfg)
    # Same inputs, same pricing math (side / EV / skip); only the decision
    # layer (threshold, sizing tier prose) moves with the overlay.
    _core = lambda xs: [(a.game_id, a.skipped_reason, a.best_side,  # noqa: E731
                         a.best_eval.ev_pct if a.best_eval else None) for a in xs]
    assert _core(challenger) == _core(champion)

    mem = _MemSupa().install(monkeypatch)
    n = rsi_shadow.save_shadow_mlb(proposal.challenger_tag, "mlb", challenger, champion,
                                   float(chal_cfg["ev"]["threshold"]), "2026-06-10", chal_cfg,
                                   champion_threshold=champ_thr)
    assert n == 2                                                   # the two priced games
    champ_by_id = {a.game_id: a for a in champion}
    r2 = mem.row("rsi_shadow_picks", game_id="2")
    assert r2["ev_pct"] == pytest.approx(champ_by_id[2].best_eval.ev_pct)
    assert "below_threshold" in r2["pass_reason"]
    assert "below_threshold" not in (champ_by_id[2].pass_reason(champ_thr) or "")
    assert r2["is_value"] is False and r2["champion_is_value"] is False
    r1 = mem.row("rsi_shadow_picks", game_id="1")
    assert r1["pass_reason"] == champ_by_id[1].pass_reason(champ_thr)   # unchanged where the overlay is moot
    assert {r["challenger_tag"] for r in mem.tables["rsi_shadow_picks"]} == {"biff_p7"}


# --- continuity of the public record across promoted tags --------------------------
def test_compute_performance_model_tags_filter():
    from mlb_value_bot.tests.test_core import _restore_recs, _rsi_rec, _tmp_recs
    from mlb_value_bot.tracking import performance as perf

    utils, orig, recs = _tmp_recs()
    try:
        recs.upsert_recommendation(_rsi_rec(recs, gid=1, is_value=True, pass_reason=None, model_tag="biff_v1"))
        recs.upsert_recommendation(_rsi_rec(recs, gid=2, is_value=True, pass_reason=None, model_tag="biff_v2"))
        recs.upsert_recommendation(_rsi_rec(recs, gid=3, is_value=True, pass_reason=None, model_tag="biff_p9"))
        recs.upsert_recommendation(_rsi_rec(recs, gid=4, is_value=False, model_tag="biff_v2"))
        assert perf.compute_performance().overall["bets"] == 3
        assert perf.compute_performance(model_tags=["biff_v1", "biff_v2"]).overall["bets"] == 2
        assert perf.compute_performance(model_tags=["biff_v2"]).overall["bets"] == 1
        assert perf.compute_performance(model_tags=["nope"]).overall == {"bets": 0, "settled": 0}
    finally:
        _restore_recs(utils, orig, recs)


def test_push_performance_pushes_lineage_and_active_tag_scopes(monkeypatch):
    from mlb_value_bot.sync import supabase_sync as S
    from mlb_value_bot.tracking import performance as perf

    calls, posted = [], []

    def _fake_perf(since=None, model_tags=None):
        calls.append(model_tags)
        return perf.PerformanceReport(overall={"bets": 1, "settled": 0}, segments={})

    monkeypatch.setattr(perf, "compute_performance", _fake_perf)
    monkeypatch.setattr(S, "_post", lambda url, key, table, rows, on_conflict: posted.append((table, rows, on_conflict)))
    _MemSupa({"rsi_model_versions": [
        {"id": 1, "engine": "mlb", "tag": "biff_v1", "status": "superseded"},
        {"id": 2, "engine": "mlb", "tag": "biff_v2", "status": "active"},
        {"id": 3, "engine": "football", "tag": "matchup_v1", "status": "active"}]}).install(monkeypatch)
    assert S.push_performance("u", "k") == 2
    assert calls == [["biff_v1", "biff_v2"], ["biff_v2"]]
    assert [p["scope"] for p in posted[-1][1]] == ["all", "tag:biff_v2"] and posted[-1][2] == "scope"
    assert S.push_performance("u", "k", since="2026-06-01") == 2
    assert [p["scope"] for p in posted[-1][1]] == ["since:2026-06-01", "tag:biff_v2:since:2026-06-01"]

    calls.clear()
    _supa_down(monkeypatch)
    assert S.push_performance("u", "k") == 1                          # legacy: unfiltered 'all' only
    assert calls == [None] and [p["scope"] for p in posted[-1][1]] == ["all"]


def test_football_record_lineage_scope(monkeypatch, tmp_path):
    from mlb_value_bot.football.tracking import football_performance as fp
    from mlb_value_bot.football.tracking import football_store as store

    def _r(tag, result="win"):
        return {"is_value": 1, "model_tag": tag, "league": "nfl", "market": "spread", "result": result,
                "flat_stake": 0.01, "profit_loss": 0.0091 if result == "win" else -0.01,
                "clv_pp": 1.0, "pick_side": "home", "created_at": "2026-09-13", "week": 1, "date": "2026-09-13"}

    df = pd.DataFrame([_r("matchup_v1"), _r("matchup_v2", "loss"), _r("matchup_p5")])
    assert fp.record(df, "matchup_v1")["bets"] == 1
    lineage = fp.record(df, ["matchup_v1", "matchup_v2"])
    assert lineage["bets"] == 2 and lineage["wins"] == 1 and lineage["losses"] == 1
    assert lineage["model_tag"] == "matchup_v1,matchup_v2"

    _MemSupa({"rsi_model_versions": [{"id": 1, "engine": "football", "tag": "matchup_v1"},
                                     {"id": 2, "engine": "football", "tag": "matchup_v2"},
                                     {"id": 3, "engine": "mlb", "tag": "biff_v1"}]}).install(monkeypatch)
    assert fp.lineage_tags("matchup_v2") == ["matchup_v1", "matchup_v2"]
    assert fp.lineage_tags("matchup_v9") == ["matchup_v1", "matchup_v2", "matchup_v9"]

    # compute_snapshot carries the lineage scope beside the tag-filtered ones.
    monkeypatch.setattr(store, "FOOTBALL_DB_PATH", tmp_path / "fb.db")
    store.save_slate([_fb_analysis([_fb_pick()])], {"model_tag": "matchup_v1", "betting": {"paper_only": True}})
    store.save_slate([_fb_analysis([_fb_pick()], game_id="2026_01_B_C")],
                     {"model_tag": "matchup_v2", "betting": {"paper_only": True}})
    cfg = {"model_tag": "matchup_v2", "distribution_monitor": {"window": 50, "alert_share": 0.6, "min_picks": 25}}
    scopes = fp.compute_snapshot(cfg)
    assert scopes["record:all:all"]["bets"] == 1
    assert scopes["record:all:all:lineage"]["bets"] == 2
    assert scopes["record:all:all:lineage"]["model_tags"] == ["matchup_v1", "matchup_v2"]
    _supa_down(monkeypatch)
    assert fp.lineage_tags("matchup_v2") == ["matchup_v2"]
    assert fp.compute_snapshot(cfg)["record:all:all:lineage"]["bets"] == 1


# --- CLI ---------------------------------------------------------------------------------
def test_cli_rsi_grade_and_state_commands(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from mlb_value_bot.rsi.cli_rsi import cli

    runner = CliRunner()
    out = runner.invoke(cli, ["--help"])
    assert out.exit_code == 0 and "grade" in out.output and "state" in out.output

    monkeypatch.setattr(rsi_state, "STATE_DIR", tmp_path)
    _supa_down(monkeypatch)
    out = runner.invoke(cli, ["state", "--engine", "mlb"])
    assert out.exit_code == 0, out.output
    assert "biff_v1" in out.output and "default" in out.output and "shadows: none" in out.output

    _MemSupa({
        "rsi_model_versions": [{"id": 2, "engine": "football", "tag": "matchup_v2", "status": "active",
                                "overlay": {"ev.threshold": 0.04}}],
        "rsi_proposals": [{"id": 5, "engine": "football", "sport": "nfl", "status": "approved",
                           "challenger_tag": "matchup_p5", "overlay": {"weather.wind_mph": 12}}],
        "rsi_shadow_picks": [],
    }).install(monkeypatch)
    out = runner.invoke(cli, ["state", "--engine", "football"])
    assert out.exit_code == 0, out.output
    assert "matchup_v2" in out.output and "ev.threshold" in out.output and "matchup_p5" in out.output
    out = runner.invoke(cli, ["grade", "--engine", "football"])
    assert out.exit_code == 0, out.output
    assert "0 pending row(s)" in out.output


# --- proposal upsert shape (PGRST102 regression, 2026-09-26 run 1) --------------
def test_proposal_rows_share_one_shape_new_vs_existing():
    """A brand-new proposal row and an existing-row update must carry exactly
    the same key set (PROPOSAL_COLUMNS), with never-set columns null on the
    new row, so a bulk upsert never mixes shapes."""
    from mlb_value_bot.rsi import reconcile as rc

    cfg = {"consecutive_runs_for_pending": 2, "clv_min_tracked": 10}
    stats = {"settled": 50, "wins": 30, "losses": 20, "flat_roi": 0.1, "avg_clv": 1.0,
             "clv_metric": "clv_pp", "clv_tracked": 50, "t_stat": 2.5, "p_value": 0.01,
             "date_from": "2026-09-01", "date_to": "2026-09-20", "rows": 50}
    new_finding = {"engine": "football", "sport": "cfb", "key": "fb-bets|rest_bucket=short (<6)",
                   "pool": "fb-bets", "dimension": "rest_bucket", "value": "short (<6)",
                   "direction": "negative", "settled": 50, "stats": stats,
                   "bonferroni_significant": False, "clv_agrees": True, "cells_tested": 100,
                   "suggestion": {"kind": "insight", "title": "t", "description": "d"}}
    existing = {"id": 7, "engine": "football", "sport": "nfl",
                "finding_key": "fb-spread-holds|lean_type=dog", "kind": "overlay",
                "title": "x", "description": "y", "suggested_change": None, "overlay": {"a": 1},
                "status": "watch", "confidence": "low", "direction": "positive",
                "latest_stats": stats, "evidence": [{"run_date": "2026-09-15", "settled": 40}],
                "consecutive_hits": 1, "consecutive_misses": 0, "first_seen": "2026-09-15",
                "last_seen": "2026-09-15", "snoozed_until": None, "decided_at": None,
                "decision_reason": None, "challenger_tag": None, "shadow_started_at": None,
                "shadow_ends_at": None, "shadow_stats": None, "version_id": None,
                "created_at": "2026-09-15T00:00:00", "updated_at": "2026-09-15T00:00:00"}
    present_again = dict(new_finding, key="fb-spread-holds|lean_type=dog", sport="nfl",
                         dimension="lean_type", value="dog", direction="positive")
    changes = rc.reconcile([new_finding, present_again], [existing], "2026-09-22", cfg)
    rows = [rc.normalize_row(ch.row) for ch in changes]
    assert len(rows) == 2
    assert all(tuple(r.keys()) == rc.PROPOSAL_COLUMNS for r in rows)
    assert "id" not in rows[0] and "created_at" not in rows[0] and "updated_at" not in rows[0]
    new_row = next(r for r in rows if r["finding_key"].startswith("fb-bets|rest"))
    old_row = next(r for r in rows if r["finding_key"].startswith("fb-spread"))
    assert new_row["challenger_tag"] is None and new_row["status"] == "watch"
    # The existing row keeps its fetched values (nothing padded to null).
    assert old_row["overlay"] == {"a": 1} and old_row["first_seen"] == "2026-09-15"


def test_upsert_rows_groups_mixed_shapes(monkeypatch):
    """Safety net below the template: rows with different key sets are posted
    in separate uniform bodies, never padded with nulls."""
    from mlb_value_bot.rsi import supa

    bodies: list[list[dict]] = []
    monkeypatch.setattr(supa, "_credentials", lambda: ("http://x", "k"))
    monkeypatch.setattr(supa, "_post", lambda url, key, table, rows, on_conflict=None: bodies.append(rows))
    rows = [
        {"engine": "football", "finding_key": "a", "status": "watch", "overlay": None},
        {"engine": "football", "finding_key": "b", "status": "pending"},
        {"engine": "football", "finding_key": "c", "status": "watch", "overlay": {"k": 1}},
    ]
    assert supa.upsert_rows("rsi_proposals", rows, on_conflict="engine,finding_key") == 3
    assert len(bodies) == 2
    for body in bodies:
        keysets = {tuple(sorted(r.keys())) for r in body}
        assert len(keysets) == 1
    assert sum(len(b) for b in bodies) == 3
    partial = next(r for b in bodies for r in b if r["finding_key"] == "b")
    assert "overlay" not in partial
