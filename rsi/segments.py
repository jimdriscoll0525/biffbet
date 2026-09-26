"""THE single source of segment (bucket) definitions for every engine.

Both the performance pages (tracking/performance.py delegates its `_prepare`
here) and the weekly RSI review read buckets from this module, so a bucket
can never drift between "what the site shows" and "what the review mines".

Every prepare_* accepts a frame in EITHER shape:

  * the rsi_scored_games VIEW shape: engine, sport, league, date, game_key,
    market, pick_side, features (dict or JSON string), ev_pct, confidence,
    decision, pass_reason, opening_line/price, pick_price, pick_line,
    closing_line/price, result, clv, clv_metric, flat_pl_units ...
  * the legacy SQLite shape (tracking/*.to_dataframe): recommended_side,
    american_odds, decimal_odds, model_prob, clv_pct / clv_pp, reasoning_json,
    kelly_stake, opening_line, closing_line ...

Normalised helper columns (all prefixed `_`) are added first; the bucket
columns are then computed once from those. Legacy MLB columns keep the exact
dtypes/labels tracking/performance.py has always produced (categoricals in the
same order) so its report is byte-identical.

Dimension names are kept from the old retro script where they existed so
ledger keys ('<pool>|<dim>=<value>') stay comparable across the migration.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from mlb_value_bot.analysis.ev_calculator import american_to_decimal

SETTLED = {"win", "loss"}
CALENDAR_DIMS = {"month"}
# Dims that are tabulated for the report but NEVER become findings: a cell
# defined by its own CLV sign is circular (CLV- bets losing to the close is
# a tautology, not a pattern).
DIAGNOSTIC_DIMS = {"clv_sign"}
# Bucket values that mean "data absent" -- tabulated, never a finding.
NA_VALUES = {"n/a", "unknown", "none", ""}

# --- dim lists per pool ------------------------------------------------------
ML_BET_DIMS = ["stability", "sharp_fade", "odds_bucket", "fav_size", "confidence_bucket",
               "ev_bucket", "bet_tier", "blend_tier", "side_type", "venue_side",
               "clv_sign", "dispersion_bucket", "model_prob_bucket", "line_move", "month"]
ML_PASS_DIMS = ["pass_reason", "ev_shortfall", "ev_bucket", "stability", "sharp_fade",
                "odds_bucket", "fav_size", "confidence_bucket", "side_type", "venue_side",
                "dispersion_bucket", "model_prob_bucket", "line_move", "month"]
TOTALS_BET_DIMS = ["pick_side", "tier", "stability", "confidence_bucket", "ev_bucket",
                   "line_bucket", "roof", "wind_bucket", "rain", "tilt_bucket",
                   "line_move", "month"]
TOTALS_PASS_DIMS = ["pass_reason", "pick_side", "stability", "confidence_bucket", "ev_bucket",
                    "line_bucket", "roof", "wind_bucket", "rain", "tilt_bucket", "month"]
GRIFF_BET_DIMS = ["stability", "sharp_fade", "odds_bucket", "fav_size", "confidence_bucket",
                  "ev_bucket", "side_type", "venue_side", "clv_sign", "model_prob_bucket",
                  "line_move", "month"]
FB_BET_DIMS = ["league_market", "pick_side", "spread_band", "lean_type", "stability", "tier",
               "venue_side", "fav_size", "wind_bucket", "rain", "conference_game",
               "division_game", "rest_bucket", "archetype", "dual_edge", "ol_grade_pick",
               "phase_edge_bucket", "line_move", "month"]
FB_HOLD_DIMS = ["hold_kind", "spread_band", "lean_type", "league_market", "venue_side",
                "fav_size", "rest_bucket", "archetype", "dual_edge", "phase_edge_bucket",
                "conference_game", "month"]

# Price range (American, inclusive) each price label covers. Used by the
# reconcile overlap rule to detect nested cells (fav_size within odds_bucket
# within side_type).
PRICE_RANGES: dict[str, tuple[float, float]] = {
    "big fav -200+": (-math.inf, -200),
    "fav -150..-199": (-199, -150),
    "fav -120..-149": (-149, -120),
    "small fav -101..-119": (-119, -101),
    "small dog +100..+119": (100, 119),
    "dog +120..+149": (120, 149),
    "dog +150..+199": (150, 199),
    "big dog +200+": (200, math.inf),
    "big fav (-150+)": (-math.inf, -150),
    "small fav (-101..-149)": (-149, -101),
    "small dog (+100..+150)": (100, 150),
    "big dog (+151+)": (151, math.inf),
    "favorite": (-math.inf, -101),
    "underdog": (100, math.inf),
}
NESTED_PRICE_DIMS = ("fav_size", "odds_bucket", "side_type")


# --- feature parsing ---------------------------------------------------------
def parse_features(raw: Any) -> dict:
    """features / reasoning as a dict (accepts dict, JSON string, None)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _get(d: dict, *path, default=None):
    node: Any = d
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
        if node is None:
            return default
    return node


def _num(value) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _first_col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


def _string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("n/a")


def _cut(series: pd.Series, bins, labels) -> pd.Series:
    """pd.cut -> string dtype with 'n/a' for NaN."""
    return _string(pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins, labels=labels))


# --- normalisation ------------------------------------------------------------
def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Add the `_`-prefixed helper columns both shapes share."""
    df = df.copy()
    feats = _first_col(df, "features", "reasoning_json", "reasoning")
    df["_features"] = feats.apply(parse_features) if feats is not None else [{} for _ in range(len(df))]

    side = _first_col(df, "pick_side", "recommended_side")
    df["_side"] = side.astype(object).where(side.notna(), None) if side is not None else None

    price = _first_col(df, "pick_price", "american_odds", "bet_odds")
    df["_price"] = pd.to_numeric(price, errors="coerce") if price is not None else np.nan

    dec = _first_col(df, "decimal_odds")
    if dec is None:
        dec = df["_price"].apply(lambda p: american_to_decimal(p) if not math.isnan(p) else np.nan)
    df["_dec"] = pd.to_numeric(dec, errors="coerce")

    clv = _first_col(df, "clv", "clv_pct", "clv_pp", "clv_blended_vs_sharp")
    df["_clv"] = pd.to_numeric(clv, errors="coerce") if clv is not None else np.nan

    df["_date"] = df["date"].astype(str) if "date" in df.columns else ""
    df["_result"] = df["result"].fillna("pending").astype(str) if "result" in df.columns else "pending"
    df["settled"] = df["_result"].isin(SETTLED)

    flat = _first_col(df, "flat_pl_units", "flat_pl")
    if flat is None:
        # Legacy shape: derive from the American price (exactly what
        # tracking/performance.py always did), falling back to decimal_odds.
        dec_from_price = df["_price"].apply(
            lambda p: american_to_decimal(p) if not math.isnan(p) else np.nan)
        dec_for_pl = dec_from_price.where(dec_from_price.notna(), df["_dec"])
        flat = pd.Series([_flat_pl(r, d) for r, d in zip(df["_result"], dec_for_pl)],
                         index=df.index, dtype=float)
    df["flat_pl"] = pd.to_numeric(flat, errors="coerce")
    df["month"] = df["_date"].str.slice(0, 7)
    return df


def _flat_pl(result: str, dec: float) -> float:
    if result == "win":
        return float(dec) - 1.0 if not math.isnan(dec) else np.nan
    if result == "loss":
        return -1.0
    return np.nan


def _pick_prob_ml(feat: dict, side, model_prob) -> float:
    """Blended pick-side probability: stored column, else market_anchor."""
    if model_prob is not None and not (isinstance(model_prob, float) and math.isnan(model_prob)):
        return float(model_prob)
    home = _get(feat, "market_anchor", "blended_home_prob")
    if home is None:
        return float("nan")
    return float(home) if side == "home" else 1.0 - float(home)


def _implied(american: float) -> float:
    if american is None or math.isnan(american) or american == 0:
        return float("nan")
    return 100.0 / (american + 100.0) if american > 0 else -american / (-american + 100.0)


# --- shared lenses ------------------------------------------------------------
def fav_size_label(price: float) -> str:
    if price is None or math.isnan(price):
        return "n/a"
    if price <= -200:
        return "big fav -200+"
    if price <= -150:
        return "fav -150..-199"
    if price <= -120:
        return "fav -120..-149"
    if price < 0:
        return "small fav -101..-119"
    if price < 120:
        return "small dog +100..+119"
    if price < 150:
        return "dog +120..+149"
    if price < 200:
        return "dog +150..+199"
    return "big dog +200+"


def line_move_price(open_price: float, close_price: float, tol_pp: float = 0.5) -> str:
    """Did the moneyline move toward our pick (price shortened) or against?"""
    o, c = _implied(_num(open_price)), _implied(_num(close_price))
    if math.isnan(o) or math.isnan(c):
        return "n/a"
    delta = (c - o) * 100.0
    if delta > tol_pp:
        return "toward"
    if delta < -tol_pp:
        return "against"
    return "flat"


def line_move_number(open_line: float, close_line: float, side, market: str) -> str:
    """Totals / spreads: did the NUMBER move toward the pick?"""
    o, c = _num(open_line), _num(close_line)
    if math.isnan(o) or math.isnan(c):
        return "n/a"
    if market == "total":
        if side not in ("over", "under"):
            return "n/a"
        delta = (c - o) if side == "over" else (o - c)
    else:
        # The pick side's own spread: a falling number (-3 -> -4, +7 -> +6)
        # means the market moved toward the pick.
        delta = o - c
    if delta > 0:
        return "toward"
    if delta < 0:
        return "against"
    return "flat"


def line_move_prob(open_p: float, close_p: float, tol_pp: float = 0.5) -> str:
    o, c = _num(open_p), _num(close_p)
    if math.isnan(o) or math.isnan(c):
        return "n/a"
    delta = (c - o) * 100.0
    if delta > tol_pp:
        return "toward"
    if delta < -tol_pp:
        return "against"
    return "flat"


def rain_label(feat: dict) -> str:
    prob = _num(_get(feat, "weather", "precip_prob"))
    mm = _num(_get(feat, "weather", "precip_mm"))
    if math.isnan(prob) and math.isnan(mm):
        return "n/a"
    if (not math.isnan(prob) and prob >= 50) or (not math.isnan(mm) and mm > 0.5):
        return "rain"
    return "dry"


def rest_bucket_label(days) -> str:
    d = _num(days)
    if math.isnan(d):
        return "n/a"
    if d < 6:
        return "short (<6)"
    if d <= 8:
        return "normal (6-8)"
    if d <= 13:
        return "long (9-13)"
    return "bye (14+)"


def pass_reason_token(reason) -> str:
    """First token of a pass_reason / hold_reason ('filter:heavy_favorite+skip:x'
    -> 'filter:heavy_favorite'; 'divergence guard: model ...' -> 'divergence guard')."""
    if reason is None or (isinstance(reason, float) and math.isnan(reason)):
        return "n/a"
    text = str(reason).strip()
    if not text:
        return "n/a"
    if "below threshold" in text:
        return "below EV threshold"
    token = text.split("+")[0]
    if ":" in token and not token.startswith(("filter:", "skip:")):
        token = token.split(":")[0]
    return token.strip() or "n/a"


def _model_prob_bucket(p: float) -> str:
    if p is None or math.isnan(p):
        return "n/a"
    if p < 0.50:
        return "<50%"
    if p < 0.55:
        return "50-55%"
    if p < 0.60:
        return "55-60%"
    return "60%+"


# --- MLB moneyline ------------------------------------------------------------
def prepare_mlb(df: pd.DataFrame, ev_threshold: float = 0.03) -> pd.DataFrame:
    """MLB moneyline buckets (bets and passes alike; GriffBet reuses it).

    Legacy-shape columns keep tracking/performance.py's exact dtypes/labels.
    """
    if df.empty:
        return df
    df = _normalise(df)
    feats = df["_features"]

    df["confidence_bucket"] = pd.cut(
        pd.to_numeric(df["confidence"], errors="coerce"), bins=[0, 40, 60, 80, 100],
        labels=["0-40", "40-60", "60-80", "80-100"], include_lowest=True)
    df["ev_bucket"] = pd.cut(
        pd.to_numeric(df["ev_pct"], errors="coerce"),
        bins=[-np.inf, 0.03, 0.05, 0.08, 0.12, np.inf],
        labels=["<3%", "3-5%", "5-8%", "8-12%", "12%+"])
    if "kelly_stake" in df.columns:
        df["kelly_bucket"] = pd.cut(
            pd.to_numeric(df["kelly_stake"], errors="coerce"),
            bins=[-np.inf, 0.005, 0.01, 0.015, np.inf],
            labels=["<0.5%", "0.5-1%", "1-1.5%", "1.5%+"])
    df["side_type"] = np.where(df["_price"] < 0, "favorite", "underdog")
    df["venue_side"] = np.where(df["_side"] == "home", "home", "road")
    df["clv_sign"] = np.where(
        df["_clv"].isna(), "unknown", np.where(df["_clv"] > 0, "CLV+", "CLV-"))
    df["bet_tier"] = feats.apply(
        lambda f: _get(f, "bet_sizing", "tier") or "n/a").astype("string")
    df["blend_tier"] = feats.apply(
        lambda f: _get(f, "market_anchor", "blend_tier") or "n/a").astype("string")
    df["stability"] = pd.Categorical(
        feats.apply(lambda f: _get(f, "stability", "label") or "n/a"),
        categories=["stable", "moderate", "fragile", "n/a"])

    model_prob_col = df["model_prob"] if "model_prob" in df.columns else pd.Series([None] * len(df), index=df.index)
    pick_prob = pd.Series(
        [_pick_prob_ml(f, s, mp) for f, s, mp in zip(feats, df["_side"], model_prob_col)],
        index=df.index, dtype=float)
    df["_model_prob"] = pick_prob

    def _fade(f: dict, side, p: float) -> float:
        sharp_home = _get(f, "market_intel", "sharp_devig_home")
        if sharp_home is None or math.isnan(p):
            return np.nan
        sharp_pick = float(sharp_home) if side == "home" else 1.0 - float(sharp_home)
        return (p - sharp_pick) * 100.0

    fade_pp = pd.Series([_fade(f, s, p) for f, s, p in zip(feats, df["_side"], pick_prob)],
                        index=df.index, dtype=float)
    df["sharp_fade"] = pd.cut(
        fade_pp, bins=[-np.inf, -3.0, 3.0, 4.0, np.inf],
        labels=["sharps agree 3pp+", "neutral", "fade 3-4pp", "fade 4pp+"],
    ).cat.add_categories(["n/a"]).fillna("n/a")
    df["odds_bucket"] = pd.cut(
        df["_price"], bins=[-np.inf, -150, 0, 150, np.inf],
        labels=["big fav (-150+)", "small fav (-101..-149)",
                "small dog (+100..+150)", "big dog (+151+)"])

    # --- RSI lenses (string dtype, 'n/a' when unknown) ---------------------
    df["fav_size"] = df["_price"].apply(fav_size_label).astype("string")
    df["model_prob_bucket"] = pick_prob.apply(_model_prob_bucket).astype("string")
    disp = feats.apply(lambda f: _num(_get(f, "market_intel", "dispersion_pp")))
    df["dispersion_bucket"] = _cut(
        disp, bins=[-np.inf, 0.25, 0.45, 0.60, np.inf],
        labels=["tight (<0.25pp)", "normal (0.25-0.45pp)", "wide (0.45-0.6pp)",
                "very wide (0.6pp+)"])
    open_price = _first_col(df, "opening_price", "opening_line")
    close_price = _first_col(df, "closing_price", "closing_line")
    if open_price is not None and close_price is not None:
        df["line_move"] = pd.Series(
            [line_move_price(o, c) for o, c in zip(open_price, close_price)],
            index=df.index).astype("string")
    else:
        df["line_move"] = pd.Series(["n/a"] * len(df), index=df.index).astype("string")
    shortfall = ev_threshold - pd.to_numeric(df["ev_pct"], errors="coerce")
    df["ev_shortfall"] = _cut(
        shortfall, bins=[-np.inf, 0.0, 0.01, 0.02, 0.03, np.inf],
        labels=["cleared threshold", "0-1pp short", "1-2pp short", "2-3pp short",
                "3pp+ short"])
    df["pass_reason"] = _pass_reason_series(df, feats)
    return df


def _pass_reason_series(df: pd.DataFrame, feats: pd.Series) -> pd.Series:
    col = _first_col(df, "pass_reason")
    if col is not None:
        raw = col
    else:
        raw = feats.apply(_legacy_pass_reason)
    out = pd.Series([pass_reason_token(r) for r in raw], index=df.index)
    return out.astype("string")


def _legacy_pass_reason(feat: dict) -> str | None:
    """Legacy rows carry selection_filters strings instead of pass_reason."""
    hold = feat.get("hold_reason")
    if hold:
        return str(hold)
    reasons = []
    for f in feat.get("selection_filters") or []:
        f = str(f)
        if "heavy favorite" in f:
            reasons.append("filter:heavy_favorite")
        elif "no sharp book" in f:
            reasons.append("filter:no_sharp_coverage")
        elif "min_model_prob" in f:
            reasons.append("filter:min_model_prob")
        else:
            reasons.append("filter:other")
    return "+".join(reasons) if reasons else None


def prepare_griff(df: pd.DataFrame) -> pd.DataFrame:
    """GriffBet moneyline rows share the MLB lenses (clv_blended_vs_sharp)."""
    return prepare_mlb(df)


# --- MLB totals --------------------------------------------------------------
def prepare_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _normalise(df)
    feats = df["_features"]

    df["pick_side"] = _string(pd.Series(df["_side"], index=df.index))
    tier = _first_col(df, "tier")
    df["tier"] = _string(tier) if tier is not None else \
        feats.apply(lambda f: _get(f, "pick", "tier") or "n/a").astype("string")
    stab = _first_col(df, "stability")
    if stab is not None and stab.notna().any():
        df["stability"] = _string(stab)
    else:
        df["stability"] = feats.apply(lambda f: _get(f, "stability", "label") or "n/a").astype("string")
    df["confidence_bucket"] = _cut(df["confidence"], bins=[0, 40, 60, 80, 100],
                                   labels=["0-40", "40-60", "60-80", "80-100"])
    df["ev_bucket"] = _cut(df["ev_pct"], bins=[-np.inf, 0.0, 0.03, 0.05, 0.08, np.inf],
                           labels=["<0%", "0-3%", "3-5%", "5-8%", "8%+"])
    line = _first_col(df, "pick_line", "market_total")
    if line is None:
        line = feats.apply(lambda f: _num(_get(f, "pick", "line")))
    df["_line"] = pd.to_numeric(line, errors="coerce")
    df["line_bucket"] = _cut(df["_line"], bins=[-np.inf, 7.5, 8.5, 9.5, np.inf],
                             labels=["low (<=7.5)", "mid (8-8.5)", "high (9-9.5)",
                                     "very high (10+)"])
    df["roof"] = feats.apply(lambda f: _get(f, "weather", "roof") or "n/a").astype("string")
    wind = feats.apply(lambda f: _num(_get(f, "weather", "wind_out_component")))
    df["wind_bucket"] = _cut(wind, bins=[-np.inf, -5.0, 5.0, 15.0, np.inf],
                             labels=["blowing in (5+)", "calm (-5..5)", "out 5-15", "out 15+"])
    df["rain"] = feats.apply(rain_label).astype("string")
    tilt = feats.apply(lambda f: _num(_get(f, "run_distribution", "tilt")))
    df["tilt_bucket"] = _cut(
        tilt, bins=[-np.inf, -0.75, -0.25, 0.25, 0.75, np.inf],
        labels=["strong under (-0.75+)", "mild under (-0.25..-0.75)", "neutral (+/-0.25)",
                "mild over (0.25..0.75)", "strong over (0.75+)"])
    df["line_move"] = _number_line_move(df, feats, market="total").astype("string")
    df["pass_reason"] = _pass_reason_series(df, feats)
    return df


def _number_line_move(df: pd.DataFrame, feats: pd.Series, market: str | None) -> pd.Series:
    """Totals/football: opening vs sharp-close de-vigged prob if the features
    carry them, else the opening vs closing number."""
    open_p = _first_col(df, "opening_devig_p_side")
    close_p = _first_col(df, "sharp_close_devig_p_side")
    if open_p is None:
        open_p = feats.apply(lambda f: f.get("opening_devig_p_side"))
    if close_p is None:
        close_p = feats.apply(lambda f: f.get("sharp_close_devig_p_side"))
    open_line = _first_col(df, "opening_line")
    close_line = _first_col(df, "closing_line")
    markets = df["market"] if "market" in df.columns else pd.Series([market] * len(df), index=df.index)
    out = []
    for i in df.index:
        op, cp = open_p.loc[i], close_p.loc[i]
        if op is not None and cp is not None and not math.isnan(_num(op)) and not math.isnan(_num(cp)):
            out.append(line_move_prob(op, cp))
            continue
        if open_line is None or close_line is None:
            out.append("n/a")
            continue
        out.append(line_move_number(open_line.loc[i], close_line.loc[i], df["_side"].loc[i],
                                    str(markets.loc[i] or market)))
    return pd.Series(out, index=df.index)


# --- Football ----------------------------------------------------------------
def _hold_kind(reason) -> str:
    if reason is None or (isinstance(reason, float) and math.isnan(reason)):
        return "pick"
    text = str(reason)
    if not text:
        return "pick"
    if "below threshold" in text:
        return "below EV threshold"
    return text.split(":")[0].strip()


def _ol_grade_label(grade) -> str:
    if isinstance(grade, str):
        return grade
    g = _num(grade)
    if math.isnan(g):
        return "n/a"
    if g <= -0.33:
        return "weak"
    if g < 0.33:
        return "neutral"
    return "strong"


def _phase_edge(feat: dict) -> float:
    edges = _get(feat, "matchup", "edges") or []
    vals = [abs(_num(e.get("edge"))) for e in edges if isinstance(e, dict)]
    vals = [v for v in vals if not math.isnan(v)]
    return max(vals) if vals else float("nan")


def _dual_edge_label(feat: dict, side) -> str:
    dual = _get(feat, "matchup", "dual_edge_side")
    if not dual:
        return "none"
    if side in ("home", "away"):
        return "dual edge (pick)" if dual == side else "dual edge (other)"
    return "dual edge"


def prepare_football(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = _normalise(df)
    feats = df["_features"]

    league = _first_col(df, "league")
    if league is None or league.isna().all():
        league = _first_col(df, "sport")
    df["_league"] = league.astype(str) if league is not None else "n/a"
    market = df["market"].astype(str) if "market" in df.columns else "n/a"
    df["_market"] = market
    df["league_market"] = (df["_league"] + " " + df["_market"]).astype("string")
    df["pick_side"] = _string(pd.Series(df["_side"], index=df.index))

    line = _first_col(df, "pick_line", "line")
    df["_line"] = pd.to_numeric(line, errors="coerce") if line is not None else np.nan
    is_spread = df["_market"] == "spread"
    band = _cut(df["_line"].abs().where(is_spread),
                bins=[-np.inf, 3.5, 7.5, 14.0, 28.0, np.inf],
                labels=["|line| 0-3.5", "3.5-7.5", "7.5-14", "14-28", "28+"])
    df["spread_band"] = band
    df["lean_type"] = pd.Series(np.where(
        ~is_spread, df["pick_side"].astype(object),
        np.where(df["_line"] > 0, "dog", "favorite")), index=df.index).astype("string")
    df["stability"] = _feat_or_col(df, feats, "stability", ("stability", "label"))
    df["tier"] = _feat_or_col(df, feats, "tier", ("tier",))
    reason = _first_col(df, "pass_reason")
    if reason is None:
        reason = feats.apply(lambda f: f.get("hold_reason"))
    df["hold_kind"] = reason.apply(_hold_kind).astype("string")
    df["venue_side"] = pd.Series(np.where(
        is_spread & df["pick_side"].isin(["home", "away"]), df["pick_side"].astype(object), "n/a"),
        index=df.index).astype("string")
    df["fav_size"] = pd.Series(np.where(
        is_spread, df["lean_type"].astype(object) + " " + df["spread_band"].astype(object), "n/a"),
        index=df.index).astype("string")

    def _wind(f: dict) -> str:
        if _get(f, "weather", "indoor"):
            return "indoor"
        mph = _num(_get(f, "weather", "wind_mph"))
        if math.isnan(mph):
            return "n/a"
        if mph < 8:
            return "calm (<8)"
        if mph < 12:
            return "breezy (8-12)"
        if mph < 18:
            return "windy (12-18)"
        return "strong (18+)"

    df["wind_bucket"] = feats.apply(_wind).astype("string")
    df["rain"] = feats.apply(rain_label).astype("string")
    df["conference_game"] = feats.apply(
        lambda f: _bool_label(_get(f, "context", "conference_game"), "conference",
                              "non-conference")).astype("string")
    df["division_game"] = feats.apply(
        lambda f: _bool_label(_get(f, "context", "division_game"), "division",
                              "non-division")).astype("string")
    df["rest_bucket"] = pd.Series(
        [rest_bucket_label(_get(f, "context", f"{s}_rest_days")) if s in ("home", "away") else "n/a"
         for f, s in zip(feats, df["_side"])], index=df.index).astype("string")
    df["archetype"] = feats.apply(
        lambda f: _get(f, "matchup", "archetype") or "n/a").astype("string")
    df["dual_edge"] = pd.Series([_dual_edge_label(f, s) for f, s in zip(feats, df["_side"])],
                                index=df.index).astype("string")
    df["ol_grade_pick"] = pd.Series(
        [_ol_grade_label(_get(f, "ol", s, "grade")) if s in ("home", "away") else "n/a"
         for f, s in zip(feats, df["_side"])], index=df.index).astype("string")
    df["phase_edge_bucket"] = _cut(feats.apply(_phase_edge),
                                   bins=[-np.inf, 10.0, 20.0, 35.0, np.inf],
                                   labels=["<10", "10-20", "20-35", "35+"])
    df["line_move"] = _number_line_move(df, feats, market=None).astype("string")
    df["pass_reason"] = df["hold_kind"]
    return df


def _bool_label(value, yes: str, no: str) -> str:
    if value is None:
        return "n/a"
    return yes if bool(value) else no


def _feat_or_col(df: pd.DataFrame, feats: pd.Series, col: str, path: tuple) -> pd.Series:
    series = _first_col(df, col)
    if series is not None and series.notna().any():
        return _string(series)
    return feats.apply(lambda f: _get(f, *path) or "n/a").astype("string")


def price_range(dim: str, value: str) -> tuple[float, float] | None:
    """Inclusive American price range a nested-price cell covers, or None."""
    if dim not in NESTED_PRICE_DIMS:
        return None
    return PRICE_RANGES.get(str(value))
