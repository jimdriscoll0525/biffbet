"""The weekly review: mine rsi_scored_games per sport, reconcile findings into
rsi_proposals, record the run, refresh shadow stats and the rollback monitor.

run_review(["all"]) is what the Monday job and the CLI call. dry_run computes
everything and writes nothing (prints the summary instead).
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd

from mlb_value_bot.rsi import load_rsi_config
from mlb_value_bot.rsi import reconcile as _reconcile
from mlb_value_bot.rsi import segments as seg
from mlb_value_bot.rsi import stats as st
from mlb_value_bot.rsi import suggest as sg
from mlb_value_bot.rsi.holdout import is_holdout, sport_key_for
from mlb_value_bot.rsi.report import build_summary
from mlb_value_bot.utils import get_logger

log = get_logger("rsi.review")

SPORTS = ("mlb", "mlb_totals", "football", "griffbet")
ENGINE_FOR_SPORT = {"mlb": "mlb", "mlb_totals": "mlb_totals", "football": "football",
                    "griffbet": "griffbet"}
CLV_METRIC = {"mlb": "clv_pct", "mlb_totals": "clv_pp", "football": "clv_pp",
              "griffbet": "clv_blended_vs_sharp"}


@dataclass
class SportReview:
    sport: str
    engine: str
    rows: int = 0
    settled: int = 0
    cells_tested: int = 0
    pools: dict = field(default_factory=dict)          # pool -> baseline cell_stats
    tables: dict = field(default_factory=dict)         # pool -> dim -> value -> stats
    findings: list = field(default_factory=list)       # Candidate objects
    changes: list = field(default_factory=list)        # ProposalChange objects
    proposals_after: list = field(default_factory=list)
    error: str | None = None


@dataclass
class ReviewResult:
    run_date: str
    sports: list[SportReview]
    dry_run: bool = False
    run_id: int | None = None
    shadows: list = field(default_factory=list)
    rollbacks: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# --- pools ----------------------------------------------------------------------
def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(picks, settled passes)."""
    decision = df["decision"].astype(str) if "decision" in df.columns else pd.Series("pick", index=df.index)
    picks = df[decision == "pick"]
    passes = df[(decision == "pass") & (df["result"].fillna("pending").astype(str) != "pending")]
    return picks, passes


def pools_for(sport: str, df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, list[str]]]:
    """[(pool, prepared frame, dims)] for one sport."""
    if df.empty:
        return []
    picks, passes = _split(df)
    if sport == "mlb":
        return [("ml-bets", seg.prepare_mlb(picks), seg.ML_BET_DIMS),
                ("ml-passes", seg.prepare_mlb(passes), seg.ML_PASS_DIMS)]
    if sport == "mlb_totals":
        return [("totals-bets", seg.prepare_totals(picks), seg.TOTALS_BET_DIMS),
                ("totals-passes", seg.prepare_totals(passes), seg.TOTALS_PASS_DIMS)]
    if sport == "griffbet":
        return [("griff-bets", seg.prepare_griff(picks), seg.GRIFF_BET_DIMS)]
    if sport == "football":
        market = df["market"].astype(str)
        return [("fb-bets", seg.prepare_football(picks), seg.FB_BET_DIMS),
                ("fb-spread-holds", seg.prepare_football(passes[market.loc[passes.index] == "spread"]),
                 seg.FB_HOLD_DIMS),
                ("fb-total-holds", seg.prepare_football(passes[market.loc[passes.index] == "total"]),
                 seg.FB_HOLD_DIMS)]
    return []


def _drop_holdout(df: pd.DataFrame, engine: str, pct: int) -> pd.DataFrame:
    if df.empty:
        return df
    if "is_holdout" in df.columns:
        mask = df["is_holdout"].fillna(False).astype(bool)
    else:
        mask = pd.Series([is_holdout(sport_key_for(engine, s, l), g, pct)
                          for s, l, g in zip(df.get("sport"), df.get("league"), df["game_key"])],
                         index=df.index)
    return df[~mask]


def _base_config(engine: str) -> dict:
    try:
        if engine == "football":
            from mlb_value_bot.football import load_football_config
            return load_football_config()
        if engine in ("mlb", "mlb_totals"):
            from mlb_value_bot.utils import load_config
            return load_config()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load base config for %s: %s", engine, exc)
    return {}


def _preregistered(pool: str, frame: pd.DataFrame, cfg: dict, min_n: int, t_threshold: float,
                   clv_min: int, clv_metric: str, base_cfg: dict) -> st.Candidate | None:
    """The pre-registered small-favorites cell: one pooled test, own p-value."""
    sf = cfg.get("small_favorites") or {}
    if pool not in (sf.get("pools") or []) or frame.empty:
        return None
    dim, values = sf.get("dim", "fav_size"), [str(v) for v in (sf.get("values") or [])]
    if dim not in frame.columns:
        return None
    sub = frame[frame[dim].astype(str).isin(values)]
    stats = st.cell_stats(sub, clv_metric)
    if stats["settled"] < min_n or stats["t_stat"] is None:
        return None
    t_trigger = abs(stats["t_stat"]) >= t_threshold
    clv_sign, strong = st.clv_strength(sub, t_threshold)
    clv_trigger = strong and stats["clv_tracked"] >= clv_min
    if not (t_trigger or clv_trigger):
        return None
    direction = ("positive" if (stats["flat_roi"] or 0) > 0 else "negative") if t_trigger \
        else ("positive" if clv_sign > 0 else "negative")
    clv_agrees = None
    if stats["avg_clv"] is not None and stats["clv_tracked"] >= clv_min:
        clv_agrees = (stats["avg_clv"] > 0) == (direction == "positive")
    value = "small favs -101..-149"
    cand = st.Candidate(key=f"{pool}|{dim}={value}", pool=pool, dimension=dim, value=value,
                        direction=direction, stats=stats, t_trigger=t_trigger,
                        clv_trigger=clv_trigger, clv_agrees=clv_agrees, preregistered=True,
                        bonferroni_significant=(stats["p_value"] is not None and stats["p_value"] < 0.05))
    cand.suggestion = sg.small_favorites_suggestion(pool, direction, base_cfg, stats).to_dict()
    return cand


def review_sport(sport: str, df: pd.DataFrame, existing: list[dict], cfg: dict,
                 run_date: str) -> SportReview:
    """Pure-ish (no I/O): pools -> candidates -> reconcile for one sport."""
    engine = ENGINE_FOR_SPORT[sport]
    sr = SportReview(sport=sport, engine=engine)
    min_n = int((cfg.get("min_sample") or {}).get(sport, 75))
    t_threshold = float(cfg.get("t_threshold", 2.0))
    clv_min = int(cfg.get("clv_min_tracked", 10))
    clv_metric = CLV_METRIC[sport]
    base_cfg = _base_config(engine)

    df = _drop_holdout(df, engine, int(cfg.get("holdout_pct", 20)))
    sr.rows = int(len(df))
    if not df.empty:
        sr.settled = int(df["result"].fillna("pending").astype(str).isin(seg.SETTLED).sum())

    findings: list[st.Candidate] = []
    cells_total = 0
    for pool, frame, dims in pools_for(sport, df):
        if frame.empty:
            continue
        sr.pools[pool] = st.cell_stats(frame, clv_metric)
        pre = _preregistered(pool, frame, cfg, min_n, t_threshold, clv_min, clv_metric, base_cfg)
        if pre is not None:
            pre.engine, pre.sport = engine, sport
            findings.append(pre)
        cands, tables, tested = st.segment_report(frame, dims, pool, min_n, t_threshold,
                                                  clv_min, clv_metric)
        sr.tables[pool] = tables
        cells_total += tested
        base = sr.pools[pool]
        for c in cands:
            c.engine = engine
            c.sport = c.sport or (sport if sport != "football" else "football")
            if base.get("avg_clv") is not None and c.stats.get("avg_clv") is not None:
                c.delta_clv = round(c.stats["avg_clv"] - base["avg_clv"], 4)
            if base.get("flat_roi") is not None and c.stats.get("flat_roi") is not None:
                c.delta_roi = round(c.stats["flat_roi"] - base["flat_roi"], 4)
            c.suggestion = sg.suggest(engine, pool, c.dimension, c.value, c.direction, base_cfg,
                                      c.stats, sport=c.sport).to_dict()
            findings.append(c)
    sr.cells_tested = cells_total
    for c in findings:
        c.cells_tested = cells_total
        if not c.preregistered:
            c.bonferroni_significant = st.bonferroni(c.stats.get("p_value"), cells_total)
    sr.findings = findings
    sr.changes = _reconcile.reconcile(findings, existing, run_date, cfg)
    sr.proposals_after = _reconcile.apply_changes(existing, sr.changes)
    return sr


# --- I/O ------------------------------------------------------------------------
def _write_changes(sr: SportReview) -> None:
    from mlb_value_bot.rsi import supa

    # One shape for every row (new or existing) -- see reconcile.PROPOSAL_COLUMNS.
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [{**_reconcile.normalize_row(ch.row), "updated_at": stamp} for ch in sr.changes]
    if not rows:
        return
    supa.upsert_rows("rsi_proposals", rows, on_conflict="engine,finding_key")
    ids = {(p["engine"], p["finding_key"]): p["id"]
           for p in supa.get_rows("rsi_proposals", {"engine": sr.engine}, select="id,engine,finding_key")}
    events = []
    for ch in sr.changes:
        pid = ids.get((ch.engine, ch.finding_key))
        ch.proposal_id = pid
        for ev in ch.events:
            if pid is None:
                continue
            events.append({"proposal_id": pid, "event": ev["event"], "actor": "engine",
                           "payload": ev.get("payload")})
    supa.insert_rows("rsi_proposal_events", events)
    for p in sr.proposals_after:
        pid = ids.get((p.get("engine"), p.get("finding_key")))
        if pid is not None:
            p["id"] = pid


def run_review(sports: list[str], since: str | None = None, dry_run: bool = False,
               run_date: str | None = None, cfg: dict | None = None) -> ReviewResult:
    cfg = cfg or load_rsi_config()
    run_date = run_date or date.today().isoformat()
    wanted = list(SPORTS) if (not sports or "all" in sports) else [s for s in sports if s in SPORTS]
    result = ReviewResult(run_date=run_date, sports=[], dry_run=dry_run)
    from mlb_value_bot.rsi import supa

    run_row: dict | None = None
    if not dry_run:
        try:
            run_row = supa.insert_returning("rsi_review_runs", {
                "run_date": run_date, "sport": "all" if len(wanted) == len(SPORTS) else ",".join(wanted),
                "status": "running",
                "params": {"min_sample": cfg.get("min_sample"), "t_threshold": cfg.get("t_threshold"),
                           "holdout_pct": cfg.get("holdout_pct"), "since": since,
                           "consecutive_runs_for_pending": cfg.get("consecutive_runs_for_pending"),
                           "clv_min_tracked": cfg.get("clv_min_tracked")}})
            result.run_id = run_row.get("id")
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"could not open review run: {exc}")

    for sport in wanted:
        engine = ENGINE_FOR_SPORT[sport]
        try:
            df = supa.fetch_scored_games(engine=engine, since=since)
            existing = supa.get_rows("rsi_proposals", {"engine": engine}, order="id.asc")
            sr = review_sport(sport, df, existing, cfg, run_date)
            if not dry_run:
                _write_changes(sr)
        except Exception as exc:  # noqa: BLE001 - one sport must not sink the run
            log.error("review failed for %s: %s\n%s", sport, exc, traceback.format_exc())
            sr = SportReview(sport=sport, engine=engine, error=str(exc))
            result.errors.append(f"{sport}: {exc}")
        result.sports.append(sr)

    if not dry_run:
        try:
            from mlb_value_bot.rsi import shadow_stats
            result.shadows = shadow_stats.refresh_all(cfg)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"shadow stats: {exc}")
        try:
            from mlb_value_bot.rsi import versions
            result.rollbacks = versions.rolling_check_all(cfg)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"rolling check: {exc}")

    result.summary = build_summary(result)
    row_counts = {sr.sport: {"rows": sr.rows, "settled": sr.settled,
                             "pools": {p: b.get("rows") for p, b in sr.pools.items()}}
                  for sr in result.sports}
    if dry_run:
        print(json.dumps({"row_counts": row_counts, "summary": result.summary}, indent=1, default=str))
        for sr in result.sports:
            for ch in sr.changes:
                print(f"  {sr.sport}: {ch.finding_key}: {ch.status_before} -> {ch.status_after} "
                      f"{'(new)' if ch.is_new else ''} {ch.note}")
    elif run_row is not None:
        try:
            supa.patch_rows("rsi_review_runs", {"id": result.run_id}, {
                "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": "error" if result.errors and not any(s.error is None for s in result.sports) else "ok",
                "row_counts": row_counts,
                "cells_tested": sum(sr.cells_tested for sr in result.sports),
                "summary": result.summary,
                "error": "; ".join(result.errors) if result.errors else None,
            })
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"could not close review run: {exc}")
    return result
