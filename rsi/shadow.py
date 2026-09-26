"""Shadow picks: a challenger's output for one slate, written to
rsi_shadow_picks (never to the public recommendation tables).

Row shaping mirrors the champion stores (one row per game x market, game_id
as text, market moneyline | total | spread) and the CLV unit matches the
engine's own: clv_pct (price re-pricing) for MLB moneyline, clv_pp
(de-vigged probability points vs the sharp close) for totals and football.

Freeze semantics = football_store.upsert_pick:
  * an existing VALUE row keeps its opening_* and only refreshes closing_*
    + clv (the bet is frozen at commit);
  * an existing PASS row keeps its opening_* unless the pick side flipped,
    the row is being promoted to a value pick, or it never had an opening
    (re-freeze at the current price);
  * a new row sets opening_* from the current price / line / de-vig.
result / flat_pl are never sent from here (grading owns them), so re-running
a past date cannot un-settle a graded shadow pick.

Both save_* functions are best-effort: they catch, log and return the number
of rows written. A shadow failure must never touch the champion path.
"""
from __future__ import annotations

from datetime import datetime, timezone

from mlb_value_bot.utils import get_logger

log = get_logger("rsi.shadow")

_CONFLICT = "challenger_tag,sport,date,game_id,market"
_EXISTING_SELECT = ("id,game_id,market,pick_side,is_value,opening_price,opening_line,"
                    "opening_devig_p_side,ev_pct,adjusted_ev_pct,confidence,model_prob,"
                    "market_prob_devigged,bet_odds,decimal_odds,line,pass_reason,"
                    "champion_is_value,home_team,away_team,reasoning,league,engine")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clv_pct(opening: int | None, closing: int | None) -> float | None:
    from mlb_value_bot.tracking.recommendations import _compute_clv

    return _compute_clv(opening, closing)


def _clv_pp(opening: float | None, closing: float | None) -> float | None:
    from mlb_value_bot.tracking.totals_recommendations import _clv_pp as pp

    return pp(opening, closing)


def _fetch_existing(challenger_tag: str, sport: str, dates: list[str]) -> dict[tuple[str, str], dict]:
    from mlb_value_bot.rsi import supa

    if not dates:
        return {}
    date_filter = dates[0] if len(dates) == 1 else ("in", sorted(set(dates)))
    rows = supa.get_rows("rsi_shadow_picks",
                         {"challenger_tag": challenger_tag, "sport": sport, "date": date_filter},
                         select=_EXISTING_SELECT)
    return {(str(r.get("game_id")), str(r.get("market"))): r for r in rows}


def _merge_freeze(fresh: dict, existing: dict | None, clv_fn) -> dict:
    """Apply the freeze rules to a freshly shaped row. `fresh` carries the
    current price/line/de-vig in BOTH opening_* and closing_*; `clv_fn` takes
    (opening_ref, closing_ref) in the engine's CLV unit."""
    now = _now()
    if existing is None:
        fresh["updated_at"] = now
        return fresh
    if bool(existing.get("is_value")):
        # Frozen value pick: keep everything the challenger committed to,
        # refresh only the close + CLV (+ the champion's latest decision).
        row = {k: v for k, v in existing.items() if k not in ("id",)}
        row.update({
            "closing_price": fresh.get("closing_price"),
            "closing_line": fresh.get("closing_line"),
            "sharp_close_devig_p_side": fresh.get("sharp_close_devig_p_side"),
            "champion_is_value": fresh.get("champion_is_value"),
            "updated_at": now,
        })
        # Natural-key + not-null columns the upsert needs.
        for col in ("challenger_tag", "sport", "date", "game_id", "market", "engine",
                    "home_team", "away_team", "pick_side", "clv_metric", "league"):
            row.setdefault(col, fresh.get(col))
        row["clv"] = clv_fn(row.get(_open_ref(fresh)), fresh.get(_close_ref(fresh)))
        return row
    side_flipped = str(existing.get("pick_side")) != str(fresh.get("pick_side"))
    refreeze = bool(fresh.get("is_value")) or side_flipped or existing.get(_open_ref(fresh)) is None
    if not refreeze:
        fresh["opening_price"] = existing.get("opening_price")
        fresh["opening_line"] = existing.get("opening_line")
        fresh["opening_devig_p_side"] = existing.get("opening_devig_p_side")
    fresh["clv"] = clv_fn(fresh.get(_open_ref(fresh)), fresh.get(_close_ref(fresh)))
    fresh["updated_at"] = now
    return fresh


def _open_ref(row: dict) -> str:
    return "opening_price" if row.get("clv_metric") == "clv_pct" else "opening_devig_p_side"


def _close_ref(row: dict) -> str:
    return "closing_price" if row.get("clv_metric") == "clv_pct" else "sharp_close_devig_p_side"


def _threshold(sport: str, cfg: dict, threshold: float | None) -> float:
    if threshold is not None:
        return float(threshold)
    if sport == "mlb_totals":
        return float((cfg.get("totals") or {}).get("ev_threshold", 0.03))
    return float((cfg.get("ev") or {}).get("threshold", 0.03))


# --- MLB (moneyline + totals) --------------------------------------------------
def _ml_row(a, champ, challenger_tag: str, game_date: str, thr: float, champ_thr: float) -> dict | None:
    be = a.best_eval
    if be is None:
        return None
    price = int(be.american_odds)
    return {
        "challenger_tag": challenger_tag, "engine": "mlb", "sport": "mlb", "league": None,
        "date": game_date, "game_id": str(a.game_id), "market": "moneyline",
        "home_team": a.home_team, "away_team": a.away_team, "pick_side": a.best_side or "",
        "is_value": bool(a.is_value(thr)), "pass_reason": a.pass_reason(thr),
        "champion_is_value": (bool(champ.is_value(champ_thr)) if champ is not None
                              and champ.best_eval is not None else None),
        "ev_pct": be.ev_pct, "adjusted_ev_pct": a.adjusted_ev_pct, "confidence": a.confidence,
        "model_prob": be.model_prob, "market_prob_devigged": be.market_prob_devigged,
        "bet_odds": price, "decimal_odds": be.decimal_odds, "line": None,
        "opening_price": price, "opening_line": None,
        "opening_devig_p_side": be.market_prob_devigged,
        "closing_price": price, "closing_line": None, "sharp_close_devig_p_side": None,
        "clv": _clv_pct(price, price), "clv_metric": "clv_pct",
        "reasoning": a.reasoning(),
    }


def _totals_row(t, champ, challenger_tag: str, game_date: str, thr: float, champ_thr: float) -> dict | None:
    be = t.best_eval
    if be is None or t.rd is None or t.intel is None:
        return None
    side = t.pick_side or ""
    price = int(be.american_odds)
    open_devig = t.opening_devig_for(side)
    sharp_devig = t.sharp_close_devig_for(side)
    close_price = t.intel.best_over_price if side == "over" else t.intel.best_under_price
    champ_value = None
    if champ is not None and champ.best_eval is not None and champ.rd is not None:
        champ_value = bool(champ.is_value(champ_thr))
    return {
        "challenger_tag": challenger_tag, "engine": "mlb_totals", "sport": "mlb_totals",
        "league": None, "date": game_date, "game_id": str(t.game_id), "market": "total",
        "home_team": t.home_team, "away_team": t.away_team, "pick_side": side,
        "is_value": bool(t.is_value(thr)), "pass_reason": t.pass_reason(thr),
        "champion_is_value": champ_value,
        "ev_pct": be.ev_pct, "adjusted_ev_pct": None, "confidence": t.confidence,
        "model_prob": be.model_prob, "market_prob_devigged": be.market_prob_devigged,
        "bet_odds": price, "decimal_odds": be.decimal_odds, "line": t.market_total,
        "opening_price": price, "opening_line": t.market_total, "opening_devig_p_side": open_devig,
        "closing_price": close_price, "closing_line": t.intel.bet_line,
        "sharp_close_devig_p_side": sharp_devig,
        "clv": _clv_pp(open_devig, sharp_devig), "clv_metric": "clv_pp",
        "reasoning": t.reasoning(),
    }


def save_shadow_mlb(challenger_tag: str, sport: str, analyses: list, champion_analyses: list,
                    threshold: float | None, game_date: str, cfg: dict,
                    champion_threshold: float | None = None) -> int:
    """Write the challenger's slate for `sport` ('mlb' -> moneyline rows,
    'mlb_totals' -> totals rows) to rsi_shadow_picks. `analyses` /
    `champion_analyses` are GameAnalysis lists (totals are read from
    `.totals`; TotalsAnalysis lists are accepted too). `threshold` is the
    CHALLENGER's EV threshold (None = read it from `cfg`); the champion's
    decision is re-derived with `champion_threshold` (defaults to
    `threshold`). Best-effort: returns the number of rows upserted."""
    try:
        from mlb_value_bot.rsi import supa

        if sport not in ("mlb", "mlb_totals"):
            raise ValueError(f"save_shadow_mlb: unknown sport {sport!r}")
        thr = _threshold(sport, cfg, threshold)
        champ_thr = float(champion_threshold) if champion_threshold is not None else thr
        if sport == "mlb_totals":
            items = [getattr(a, "totals", a) for a in analyses]
            champs = [getattr(a, "totals", a) for a in champion_analyses or []]
            shape = _totals_row
        else:
            items, champs, shape = list(analyses), list(champion_analyses or []), _ml_row
        champ_by_id = {str(c.game_id): c for c in champs if c is not None}
        fresh: list[dict] = []
        for a in items:
            if a is None:
                continue
            row = shape(a, champ_by_id.get(str(a.game_id)), challenger_tag, game_date, thr, champ_thr)
            if row is not None:
                fresh.append(row)
        if not fresh:
            return 0
        existing = _fetch_existing(challenger_tag, sport, [game_date])
        clv_fn = _clv_pct if sport == "mlb" else _clv_pp
        rows = [_merge_freeze(r, existing.get((r["game_id"], r["market"])), clv_fn) for r in fresh]
        n = supa.upsert_rows("rsi_shadow_picks", rows, on_conflict=_CONFLICT)
        log.info("Shadow %s/%s %s: %d row(s) (%d value)", challenger_tag, sport, game_date, n,
                 sum(1 for r in rows if r.get("is_value")))
        return n
    except Exception as exc:  # noqa: BLE001 - shadows never break the champion run
        log.warning("shadow save failed for %s/%s on %s: %s", challenger_tag, sport, game_date, exc)
        return 0


# --- Football ---------------------------------------------------------------------
def _football_row(analysis, pick, champ_pick, challenger_tag: str, league: str) -> dict:
    from mlb_value_bot.analysis.ev_calculator import american_to_decimal
    from mlb_value_bot.football.tracking.football_store import _picked_side_fields

    f = _picked_side_fields(pick)
    price = int(pick.american_odds)
    return {
        "challenger_tag": challenger_tag, "engine": "football", "sport": league, "league": league,
        "date": analysis.date, "game_id": str(analysis.game_id), "market": pick.market,
        "home_team": analysis.home, "away_team": analysis.away, "pick_side": pick.side,
        "is_value": bool(pick.is_value),
        "pass_reason": None if pick.is_value else (pick.hold_reason or "below_threshold"),
        "champion_is_value": bool(champ_pick.is_value) if champ_pick is not None else None,
        "ev_pct": pick.raw_ev, "adjusted_ev_pct": pick.adjusted_ev, "confidence": pick.confidence,
        "model_prob": pick.model_prob, "market_prob_devigged": pick.market_prob,
        "bet_odds": price, "decimal_odds": american_to_decimal(price), "line": f["line"],
        "opening_price": price, "opening_line": f["line"], "opening_devig_p_side": f["devig_side"],
        "closing_price": price, "closing_line": f["line"],
        "sharp_close_devig_p_side": f["sharp_devig_side"],
        "clv": _clv_pp(f["devig_side"], f["sharp_devig_side"]), "clv_metric": "clv_pp",
        "reasoning": pick.reasoning,
    }


def save_shadow_football(challenger_tag: str, league: str, analyses: list,
                         champion_analyses: list, cfg: dict) -> int:
    """Write the challenger's board for one league (every priced market of
    every game) to rsi_shadow_picks. Best-effort; returns rows upserted."""
    try:
        from mlb_value_bot.rsi import supa

        champ_picks: dict[tuple[str, str], object] = {}
        for a in champion_analyses or []:
            for p in a.picks:
                champ_picks[(str(a.game_id), p.market)] = p
        fresh: list[dict] = []
        for a in analyses or []:
            for p in a.picks:
                fresh.append(_football_row(a, p, champ_picks.get((str(a.game_id), p.market)),
                                           challenger_tag, league))
        if not fresh:
            return 0
        existing = _fetch_existing(challenger_tag, league, sorted({r["date"] for r in fresh}))
        rows = [_merge_freeze(r, existing.get((r["game_id"], r["market"])), _clv_pp) for r in fresh]
        n = supa.upsert_rows("rsi_shadow_picks", rows, on_conflict=_CONFLICT)
        log.info("Shadow %s/%s: %d row(s) (%d value)", challenger_tag, league, n,
                 sum(1 for r in rows if r.get("is_value")))
        return n
    except Exception as exc:  # noqa: BLE001
        log.warning("football shadow save failed for %s/%s: %s", challenger_tag, league, exc)
        return 0
