"""Weekly self-improvement retro for BiffBet (mlb_value_bot).

Read-only diagnostics: mines the FULL recommendation history — settled bets
AND the is_value=0 "pass" rows the engine analyzed but never bet — for
segment-level trends that are strong enough to become ability proposals.

What it does, in order:
  1. Loads `recommendations` (+ `totals_recommendations`) from the local SQLite.
  2. Counterfactually grades passes against MLB Stats API final scores
     (one schedule call per un-graded date; results cached in
     storage/retro/pass_grades.json so each date is fetched once, ever).
  3. Segments bets and passes along the same dimensions the performance page
     uses (stability, sharp fade, odds bucket, EV bucket, confidence, tier,
     blend tier, venue, month, EV-shortfall for passes) and computes, per cell:
     record, flat P/L, a one-sample t on flat P/L, Wilson CI on hit rate vs
     the odds-implied breakeven, and (bets only) avg CLV.
  4. Flags candidate trends that clear the evidence bar and writes
     retro_report_<run-date>.json + .md into storage/retro/.

This script deliberately does NOT import mlb_value_bot modules: it is a
read-only sidecar (stdlib + pandas + scipy + urllib) so it can never mutate
engine state, and it runs even if the venv/package import path is broken.
Bucket definitions mirror tracking/performance.py — if you change buckets
there, change them here too.

It also NEVER writes to the engine DB. Counterfactual pass grades live in
their own cache file. The engine's is_value=0 rows keep result='pending'.

Usage (from the repo root, any Python with pandas):
    python .claude/skills/self-improve/scripts/retro_analysis.py
    python ... retro_analysis.py --min-n 20 --skip-pass-grading
    python ... retro_analysis.py --db storage/mlb_value_bot.db --out-dir storage/retro

Caveats baked into every report (do not silently drop them):
  * Pass rows are refreshed on every engine run: the stored side/odds/EV are
    the LAST analysis before the slate closed, not a frozen open. Counter-
    factuals are therefore "what if we had bet the final analysis snapshot".
  * Passes have no CLV (no committed opening price), so pass trends can never
    clear the CLV-agreement bar on their own — they are hypothesis fuel.
  * With ~70 settled bets, most bet-side cells are small. The t-stat and the
    cells-tested count exist precisely so nobody trades on a 9-game streak.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import urllib.request
import urllib.error
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - scipy is in requirements.txt
    _scipy_stats = None

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------

# .claude/skills/self-improve/scripts/retro_analysis.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DB = REPO_ROOT / "storage" / "mlb_value_bot.db"
DEFAULT_OUT = REPO_ROOT / "storage" / "retro"

MLB_SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
)
FINAL_STATES = {"Final", "Game Over", "Completed Early"}
VOID_STATES = {"Postponed", "Cancelled", "Canceled", "Suspended"}

SETTLED = {"win", "loss"}

# Evidence bar defaults (overridable via CLI). See SKILL.md for the lifecycle
# these feed into — clearing the bar makes a WATCH item, not an ability.
DEFAULT_MIN_N = 25          # settled games per cell
DEFAULT_T_THRESHOLD = 2.0   # |t| on flat P/L


# --------------------------------------------------------------------------
# Small stats helpers
# --------------------------------------------------------------------------

def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half, center + half)


def _t_test(series: pd.Series) -> tuple[float, float]:
    """One-sample t of mean(series) vs 0. Returns (t, p). NaN-safe."""
    x = series.dropna().astype(float)
    n = len(x)
    if n < 2 or float(x.std(ddof=1)) == 0.0:
        return (float("nan"), float("nan"))
    if _scipy_stats is not None:
        t, p = _scipy_stats.ttest_1samp(x, 0.0)
        return (float(t), float(p))
    t = float(x.mean() / (x.std(ddof=1) / math.sqrt(n)))
    # Normal approximation fallback.
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2))))
    return (t, p)


# --------------------------------------------------------------------------
# Load + derive segmentation columns (mirrors tracking/performance.py)
# --------------------------------------------------------------------------

def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()


def _reasoning(raw) -> dict:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _filter_reason(reasoning: dict) -> str:
    """Collapse selection_filters strings to their type (odds vary per row)."""
    reasons = set()
    for f in reasoning.get("selection_filters") or []:
        f = str(f)
        if "heavy favorite" in f:
            reasons.add("heavy favorite")
        elif "no sharp book" in f:
            reasons.add("no sharp coverage")
        else:
            reasons.add("other")
    return "+".join(sorted(reasons)) if reasons else "none"


def _sharp_fade_pp(reasoning: dict, side: str, model_prob: float) -> float:
    sharp_home = (reasoning.get("market_intel") or {}).get("sharp_devig_home")
    if sharp_home is None:
        return float("nan")
    sharp_pick = sharp_home if side == "home" else 1.0 - sharp_home
    return (float(model_prob) - float(sharp_pick)) * 100.0


def _prepare_moneyline(df: pd.DataFrame, ev_threshold: float) -> pd.DataFrame:
    """Add the segmentation columns. Works for bets and passes alike."""
    df = df.copy()
    parsed = df["reasoning_json"].apply(_reasoning)

    df["stability"] = parsed.apply(
        lambda r: (r.get("stability") or {}).get("label") or "n/a")
    df["bet_tier"] = parsed.apply(
        lambda r: (r.get("bet_sizing") or {}).get("tier") or "n/a")
    df["blend_tier"] = parsed.apply(
        lambda r: (r.get("market_anchor") or {}).get("blend_tier") or "n/a")

    fade_pp = pd.Series(
        [_sharp_fade_pp(r, s, mp) for r, s, mp in
         zip(parsed, df["recommended_side"], df["model_prob"])],
        index=df.index, dtype=float)
    df["sharp_fade"] = pd.cut(
        fade_pp, bins=[-np.inf, -3.0, 3.0, 4.0, np.inf],
        labels=["sharps agree 3pp+", "neutral", "fade 3-4pp", "fade 4pp+"],
    ).astype("string").fillna("n/a")

    df["odds_bucket"] = pd.cut(
        df["american_odds"], bins=[-np.inf, -150, 0, 150, np.inf],
        labels=["big fav (-150+)", "small fav (-101..-149)",
                "small dog (+100..+150)", "big dog (+151+)"]).astype("string")
    df["confidence_bucket"] = pd.cut(
        df["confidence"], bins=[0, 40, 60, 80, 100],
        labels=["0-40", "40-60", "60-80", "80-100"],
        include_lowest=True).astype("string")
    df["ev_bucket"] = pd.cut(
        df["ev_pct"], bins=[-np.inf, 0.0, 0.03, 0.05, 0.08, 0.12, np.inf],
        labels=["<0%", "0-3%", "3-5%", "5-8%", "8-12%", "12%+"]).astype("string")
    df["side_type"] = np.where(df["american_odds"] < 0, "favorite", "underdog")
    df["venue_side"] = np.where(
        df["recommended_side"] == "home", "home", "road")
    df["month"] = df["date"].astype(str).str.slice(0, 7)
    df["clv_sign"] = np.where(
        df["clv_pct"].isna(), "unknown",
        np.where(df["clv_pct"] > 0, "CLV+", "CLV-"))

    # Retro-only lenses (not in tracking/performance.py). Book disagreement:
    # when books disagree, our "best price across books" may be a soft-book
    # outlier and the stored EV partly illusory. Bin edges sit near the
    # observed quartiles (median ~0.35pp).
    disp = parsed.apply(
        lambda r: (r.get("market_intel") or {}).get("dispersion_pp"))
    df["dispersion_bucket"] = pd.cut(
        disp.astype(float), bins=[-np.inf, 0.25, 0.45, 0.60, np.inf],
        labels=["tight (<0.25pp)", "normal (0.25-0.45pp)",
                "wide (0.45-0.6pp)", "very wide (0.6pp+)"],
    ).astype("string").fillna("n/a")

    # Which selection filter (if any) blocked the game — explains passes that
    # cleared the EV threshold but were never bet.
    df["filter_reason"] = parsed.apply(_filter_reason)

    # Pass-only lens: how far below the bet threshold did the EV land?
    shortfall = ev_threshold - df["ev_pct"]
    df["ev_shortfall"] = pd.cut(
        shortfall, bins=[-np.inf, 0.0, 0.01, 0.02, 0.03, np.inf],
        labels=["cleared threshold", "0-1pp short", "1-2pp short",
                "2-3pp short", "3pp+ short"]).astype("string")
    return df


def _prepare_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = df["reasoning_json"].apply(_reasoning)

    # Retro-only lenses from reasoning_json. Roof state and wind-out component
    # (km/h toward the outfield; suppress-only in the model) test whether the
    # paper edge concentrates in weather-sensitive parks. Tilt (runs vs the
    # market-implied mean) only exists since the 2026-07 provenance commit —
    # older rows show n/a.
    df["roof"] = parsed.apply(
        lambda r: (r.get("weather") or {}).get("roof") or "n/a")
    wind = parsed.apply(
        lambda r: (r.get("weather") or {}).get("wind_out_component"))
    df["wind_bucket"] = pd.cut(
        wind.astype(float), bins=[-np.inf, -5.0, 5.0, 15.0, np.inf],
        labels=["blowing in (5+)", "calm (-5..5)",
                "out 5-15", "out 15+"]).astype("string").fillna("n/a")
    tilt = parsed.apply(
        lambda r: (r.get("run_distribution") or {}).get("tilt"))
    df["tilt_bucket"] = pd.cut(
        tilt.astype(float),
        bins=[-np.inf, -0.75, -0.25, 0.25, 0.75, np.inf],
        labels=["strong under (-0.75+)", "mild under (-0.25..-0.75)",
                "neutral (+/-0.25)", "mild over (0.25..0.75)",
                "strong over (0.75+)"]).astype("string").fillna("n/a")

    df["tier"] = df["tier"].fillna("n/a")
    df["stability"] = df["stability"].fillna("n/a")
    df["pick_side"] = df["pick_side"].fillna("n/a")
    df["confidence_bucket"] = pd.cut(
        df["confidence"], bins=[0, 40, 60, 80, 100],
        labels=["0-40", "40-60", "60-80", "80-100"],
        include_lowest=True).astype("string")
    df["ev_bucket"] = pd.cut(
        df["ev_pct"], bins=[-np.inf, 0.0, 0.03, 0.05, 0.08, np.inf],
        labels=["<0%", "0-3%", "3-5%", "5-8%", "8%+"]).astype("string")
    df["line_bucket"] = pd.cut(
        df["market_total"], bins=[-np.inf, 7.5, 8.5, 9.5, np.inf],
        labels=["low (<=7.5)", "mid (8-8.5)", "high (9-9.5)", "very high (10+)"]
    ).astype("string")
    df["month"] = df["date"].astype(str).str.slice(0, 7)
    return df


# --------------------------------------------------------------------------
# Counterfactual grading of passes (MLB Stats API, cached per date)
# --------------------------------------------------------------------------

def _fetch_finals(game_date: str) -> dict[int, dict]:
    """gamePk -> {state, home_score, away_score} for one date. Raises on I/O."""
    url = MLB_SCHEDULE_URL.format(date=game_date)
    req = urllib.request.Request(url, headers={"User-Agent": "biffbet-retro/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.load(resp)
    out: dict[int, dict] = {}
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            status = (game.get("status") or {}).get("detailedState", "")
            teams = game.get("teams") or {}
            out[int(game["gamePk"])] = {
                "state": status,
                "home_score": (teams.get("home") or {}).get("score"),
                "away_score": (teams.get("away") or {}).get("score"),
            }
    return out


def _load_grade_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def grade_passes(df: pd.DataFrame, cache_path: Path,
                 today: str, offline: bool) -> tuple[pd.DataFrame, dict]:
    """Attach cf_result / final scores to every pass row whose game is over.

    Only settles games strictly before `today`. Network results are cached in
    `cache_path` keyed "date:game_id"; a date is only re-fetched while it
    still has unresolved games on it.
    """
    cache = _load_grade_cache(cache_path)
    meta = {"fetched_dates": 0, "fetch_errors": []}

    need_dates = sorted({
        str(d) for d, g in zip(df["date"], df["game_id"])
        if str(d) < today and f"{d}:{g}" not in cache
    })
    if not offline:
        for game_date in need_dates:
            try:
                finals = _fetch_finals(game_date)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                meta["fetch_errors"].append(f"{game_date}: {exc}")
                continue
            meta["fetched_dates"] += 1
            for _, row in df[df["date"].astype(str) == game_date].iterrows():
                game = finals.get(int(row["game_id"]))
                if game is None:
                    continue
                state = game["state"]
                if state in VOID_STATES:
                    cache[f"{game_date}:{int(row['game_id'])}"] = {
                        "state": "void", "home_score": None, "away_score": None}
                elif state in FINAL_STATES and game["home_score"] is not None:
                    cache[f"{game_date}:{int(row['game_id'])}"] = {
                        "state": "final",
                        "home_score": int(game["home_score"]),
                        "away_score": int(game["away_score"])}
                # else: still in progress / not started -> leave uncached, retry next run
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))

    def _cf(row) -> str:
        entry = cache.get(f"{row['date']}:{int(row['game_id'])}")
        if entry is None:
            return "pending"
        if entry["state"] == "void":
            return "void"
        home, away = entry["home_score"], entry["away_score"]
        if home == away:  # shouldn't happen in MLB, but never guess
            return "void"
        winner = "home" if home > away else "away"
        return "win" if winner == row["recommended_side"] else "loss"

    def _cf_total(row) -> str:
        entry = cache.get(f"{row['date']}:{int(row['game_id'])}")
        if entry is None:
            return "pending"
        if entry["state"] == "void":
            return "void"
        line = row.get("opening_line")
        if line is None or (isinstance(line, float) and math.isnan(line)):
            line = row.get("market_total")
        if line is None or (isinstance(line, float) and math.isnan(line)):
            return "pending"
        total = entry["home_score"] + entry["away_score"]
        if total == line:
            return "push"
        went_over = total > line
        won = (row["pick_side"] == "over" and went_over) or \
              (row["pick_side"] == "under" and not went_over)
        return "win" if won else "loss"

    df = df.copy()
    if "recommended_side" in df.columns:
        df["cf_result"] = df.apply(_cf, axis=1)
    else:
        df["cf_result"] = df.apply(_cf_total, axis=1)
    return df, meta


# --------------------------------------------------------------------------
# Per-cell stats + candidate detection
# --------------------------------------------------------------------------

def _flat_pl(result: str, decimal_odds: float) -> float:
    if result == "win":
        return float(decimal_odds) - 1.0
    if result == "loss":
        return -1.0
    return float("nan")


def _cell_stats(sub: pd.DataFrame, result_col: str) -> dict:
    settled = sub[sub[result_col].isin(SETTLED)]
    n = len(settled)
    wins = int((settled[result_col] == "win").sum())
    flat = settled.apply(
        lambda r: _flat_pl(r[result_col], r["decimal_odds"]), axis=1) \
        if n else pd.Series(dtype=float)
    t, p = _t_test(flat)
    lo, hi = _wilson_ci(wins, n)
    breakeven = float((1.0 / settled["decimal_odds"]).mean()) if n else float("nan")
    clv = sub["clv_pct"].dropna() if "clv_pct" in sub.columns else pd.Series(dtype=float)
    return {
        "rows": len(sub),
        "settled": n,
        "wins": wins,
        "losses": n - wins,
        "hit_rate": round(wins / n, 4) if n else None,
        "hit_rate_wilson_95": [round(lo, 4), round(hi, 4)] if n else None,
        "breakeven_hit_rate": round(breakeven, 4) if n else None,
        "flat_pl_units": round(float(flat.sum()), 3) if n else 0.0,
        "flat_roi": round(float(flat.mean()), 4) if n else None,
        "t_stat": round(t, 2) if not math.isnan(t) else None,
        "p_value": round(p, 4) if not math.isnan(p) else None,
        "avg_ev_pct": round(float(sub["ev_pct"].mean()) * 100, 2) if len(sub) else None,
        "avg_clv_pct": round(float(clv.mean()), 3) if len(clv) else None,
        "clv_tracked": int(len(clv)),
    }


def _segment_report(df: pd.DataFrame, dims: list[str], result_col: str,
                    min_n: int, t_threshold: float, pool: str) -> tuple[dict, list]:
    tables: dict[str, dict] = {}
    candidates: list[dict] = []
    cells_tested = 0
    for dim in dims:
        if dim not in df.columns:
            continue
        table = {}
        for value, sub in df.groupby(dim, dropna=True, observed=True):
            stats = _cell_stats(sub, result_col)
            table[str(value)] = stats
            if stats["settled"] and stats["settled"] >= 5:
                cells_tested += 1
            if (stats["settled"] and stats["settled"] >= min_n
                    and stats["t_stat"] is not None
                    and abs(stats["t_stat"]) >= t_threshold):
                direction = "positive" if stats["flat_roi"] > 0 else "negative"
                clv_agrees = None
                if stats["avg_clv_pct"] is not None and stats["clv_tracked"] >= 10:
                    clv_agrees = (stats["avg_clv_pct"] > 0) == (direction == "positive")
                candidates.append({
                    "key": f"{pool}|{dim}={value}",
                    "pool": pool,
                    "dimension": dim,
                    "value": str(value),
                    "direction": direction,
                    "clv_agrees": clv_agrees,
                    **stats,
                })
        tables[dim] = table
    return {"tables": tables, "cells_tested": cells_tested}, candidates


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

def _md_table(table: dict, title: str) -> str:
    lines = [f"### {title}", "",
             "| bucket | rows | settled | W-L | hit | breakeven | flat ROI | t | avg CLV% |",
             "|---|---|---|---|---|---|---|---|---|"]
    for bucket, s in table.items():
        hit = f"{s['hit_rate']:.0%}" if s["hit_rate"] is not None else "-"
        be = f"{s['breakeven_hit_rate']:.0%}" if s["breakeven_hit_rate"] is not None else "-"
        roi = f"{s['flat_roi']:+.1%}" if s["flat_roi"] is not None else "-"
        t = s["t_stat"] if s["t_stat"] is not None else "-"
        clv = f"{s['avg_clv_pct']:+.2f}" if s["avg_clv_pct"] is not None else "-"
        lines.append(
            f"| {bucket} | {s['rows']} | {s['settled']} | "
            f"{s['wins']}-{s['losses']} | {hit} | {be} | {roi} | {t} | {clv} |")
    lines.append("")
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    md = [f"# BiffBet weekly retro — {report['run_date']}", ""]
    md.append(f"DB: `{report['db']}` · rows: {report['row_counts']} · "
              f"cells tested (n>=5): {report['cells_tested']} · "
              f"evidence bar: settled >= {report['params']['min_n']}, "
              f"|t| >= {report['params']['t_threshold']}")
    md.append("")
    md.append("## Candidates clearing the evidence bar")
    md.append("")
    if not report["candidates"]:
        md.append("_None this run. That is a fine outcome — the bar exists so "
                  "noise never becomes an ability._")
    else:
        for c in report["candidates"]:
            clv = {True: "CLV agrees", False: "CLV DISAGREES",
                   None: "CLV n/a"}[c["clv_agrees"]]
            md.append(
                f"- **{c['key']}** — {c['direction']}, "
                f"{c['wins']}-{c['losses']} ({c['hit_rate']:.0%} vs "
                f"{c['breakeven_hit_rate']:.0%} breakeven), flat ROI "
                f"{c['flat_roi']:+.1%}, t={c['t_stat']}, p={c['p_value']} · {clv}")
    md.append("")
    for pool_name, pool in report["pools"].items():
        md.append(f"## {pool_name}")
        md.append("")
        overall = pool["overall"]
        md.append(f"Overall: {overall['wins']}-{overall['losses']} settled of "
                  f"{overall['rows']} rows, flat ROI "
                  f"{overall['flat_roi'] if overall['flat_roi'] is not None else 'n/a'}, "
                  f"avg CLV {overall['avg_clv_pct']}")
        md.append("")
        for dim, table in pool["tables"].items():
            if table:
                md.append(_md_table(table, dim))
    md.append("## Caveats")
    md.append("")
    for caveat in report["caveats"]:
        md.append(f"- {caveat}")
    md.append("")
    return "\n".join(md)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

BET_DIMS = ["stability", "sharp_fade", "odds_bucket", "confidence_bucket",
            "ev_bucket", "bet_tier", "blend_tier", "side_type", "venue_side",
            "clv_sign", "month", "dispersion_bucket"]
PASS_DIMS = ["ev_shortfall", "stability", "sharp_fade", "odds_bucket",
             "confidence_bucket", "side_type", "venue_side", "month",
             "dispersion_bucket", "filter_reason"]
TOTALS_BET_DIMS = ["pick_side", "tier", "stability", "confidence_bucket",
                   "ev_bucket", "line_bucket", "month", "roof", "wind_bucket",
                   "tilt_bucket"]
TOTALS_PASS_DIMS = ["pick_side", "stability", "confidence_bucket",
                    "ev_bucket", "line_bucket", "month", "roof", "wind_bucket",
                    "tilt_bucket"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--since", type=str, default=None,
                    help="Only consider rows with date >= this (YYYY-MM-DD).")
    ap.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    ap.add_argument("--t-threshold", type=float, default=DEFAULT_T_THRESHOLD)
    ap.add_argument("--ev-threshold", type=float, default=0.03,
                    help="The engine's bet threshold (config.yaml ev.threshold).")
    ap.add_argument("--skip-pass-grading", action="store_true",
                    help="Offline mode: use only already-cached pass grades.")
    ap.add_argument("--skip-totals", action="store_true")
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"ERROR: DB not found at {args.db}", file=sys.stderr)
        return 2

    today = _date.today().isoformat()
    run_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pools: dict[str, dict] = {}
    candidates: list[dict] = []
    cells_total = 0
    row_counts: dict[str, int] = {}
    grade_meta: dict = {}

    # ---- Moneyline ------------------------------------------------------
    ml = _load_table(args.db, "recommendations")
    if args.since:
        ml = ml[ml["date"].astype(str) >= args.since]
    ml = _prepare_moneyline(ml, args.ev_threshold)
    bets = ml[ml["is_value"].fillna(1).astype(int) == 1]
    passes = ml[ml["is_value"].fillna(1).astype(int) == 0]
    row_counts["moneyline_bets"] = len(bets)
    row_counts["moneyline_passes"] = len(passes)

    passes, grade_meta = grade_passes(
        passes, args.out_dir / "pass_grades.json", today,
        offline=args.skip_pass_grading)

    seg, cand = _segment_report(bets, BET_DIMS, "result",
                                args.min_n, args.t_threshold, "ml-bets")
    pools["Moneyline — settled bets"] = {
        "overall": _cell_stats(bets, "result"), **seg}
    candidates += cand
    cells_total += seg["cells_tested"]

    seg, cand = _segment_report(passes, PASS_DIMS, "cf_result",
                                args.min_n, args.t_threshold, "ml-passes")
    pools["Moneyline — passed games (counterfactual)"] = {
        "overall": _cell_stats(passes, "cf_result"), **seg}
    candidates += cand
    cells_total += seg["cells_tested"]

    # ---- Totals (paper) -------------------------------------------------
    if not args.skip_totals:
        try:
            tot = _load_table(args.db, "totals_recommendations")
        except (pd.errors.DatabaseError, sqlite3.DatabaseError):
            tot = pd.DataFrame()
        if len(tot):
            if args.since:
                tot = tot[tot["date"].astype(str) >= args.since]
            tot = _prepare_totals(tot)
            tbets = tot[tot["is_value"].fillna(1).astype(int) == 1]
            tpasses = tot[tot["is_value"].fillna(1).astype(int) == 0]
            row_counts["totals_bets"] = len(tbets)
            row_counts["totals_passes"] = len(tpasses)
            tpasses, _ = grade_passes(
                tpasses, args.out_dir / "pass_grades.json", today,
                offline=args.skip_pass_grading)

            seg, cand = _segment_report(tbets, TOTALS_BET_DIMS, "result",
                                        args.min_n, args.t_threshold, "totals-bets")
            pools["Totals (paper) — settled bets"] = {
                "overall": _cell_stats(tbets, "result"), **seg}
            candidates += cand
            cells_total += seg["cells_tested"]

            seg, cand = _segment_report(tpasses, TOTALS_PASS_DIMS, "cf_result",
                                        args.min_n, args.t_threshold, "totals-passes")
            pools["Totals (paper) — passed games (counterfactual)"] = {
                "overall": _cell_stats(tpasses, "cf_result"), **seg}
            candidates += cand
            cells_total += seg["cells_tested"]

    # Multiple-comparisons context: with `cells_total` cells inspected, flag
    # how many false positives the bar would let through by chance alone.
    for c in candidates:
        c["bonferroni_significant"] = (
            c["p_value"] is not None and c["p_value"] * max(cells_total, 1) < 0.05)

    candidates.sort(key=lambda c: abs(c.get("t_stat") or 0), reverse=True)

    report = {
        "run_date": run_stamp,
        "db": str(args.db),
        "params": {"min_n": args.min_n, "t_threshold": args.t_threshold,
                   "ev_threshold": args.ev_threshold, "since": args.since},
        "row_counts": row_counts,
        "cells_tested": cells_total,
        "pass_grading": grade_meta,
        "candidates": candidates,
        "pools": pools,
        "caveats": [
            "Pass rows hold the LAST pre-slate analysis (side/odds/EV refresh "
            "every run); counterfactuals assume betting that final snapshot.",
            "Passes carry no CLV, so pass-pool trends are hypotheses only — "
            "they cannot clear the CLV-agreement bar until bet live or papered.",
            f"{cells_total} cells were inspected; at p<0.05 expect "
            f"~{max(cells_total, 1) * 0.05:.1f} false positives by chance. "
            "Prefer candidates with bonferroni_significant=true, and require "
            "persistence across consecutive weekly runs regardless.",
            "Bet-pool cells overlap (the same bets appear in every dimension); "
            "candidates are correlated, not independent findings.",
        ],
    }

    json_path = args.out_dir / f"retro_report_{run_stamp}.json"
    md_path = args.out_dir / f"retro_report_{run_stamp}.md"
    json_path.write_text(json.dumps(report, indent=1, default=str))
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Candidates: {len(candidates)} "
          f"({sum(1 for c in candidates if c['bonferroni_significant'])} "
          f"Bonferroni-significant of {cells_total} cells tested)")
    if grade_meta.get("fetch_errors"):
        print(f"WARNING: {len(grade_meta['fetch_errors'])} date(s) failed to "
              f"grade: {grade_meta['fetch_errors'][:3]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
