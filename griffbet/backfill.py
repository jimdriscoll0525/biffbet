"""Historical backfill of Stage-4 features into GriffBet's feature store.

For every graded game in the stored history, reconstruct the griff_features block
and write it to GriffBet's OWN feature store (never BiffBet's frozen rows). The
training extractor joins it in, so the features stop being zero-variance on the
existing sample and can actually move the out-of-sample verdict / ablation.

Look-ahead discipline (the review's #2/#5):
  * lineup state -- from the stored reasoning.lineup block (what was known at bet
    time), NOT re-fetched final lineups.
  * weather -- Open-Meteo ARCHIVE for the game date (forecast was knowable a day
    out; using actual is a minor idealization, not meaningful leakage).
  * pitcher pitch-quality -- rebuilt from AS-OF-DATE Statcast (data only up to the
    game date), so a pitcher's later starts never leak backward.

Every component degrades independently: a failed pitcher lookup leaves those
features 0 while lineup/weather still populate. Reports coverage.
"""
from __future__ import annotations

import json
from datetime import date as date_cls

from mlb_value_bot.griffbet import tracking as gtrack
from mlb_value_bot.griffbet.features import (
    GRIFF_FEATURE_KEYS,
    lineup_features_from_reasoning,
    pitcher_quality_features,
    weather_archive,
)
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.backfill")


def _reasoning(raw) -> dict:
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _pitcher_id(name: str):
    """MLBAM id for a pitcher name via pybaseball, or None (degrade-safe)."""
    if not name:
        return None
    try:
        from pybaseball import playerid_lookup
        parts = name.replace(".", "").split()
        if len(parts) < 2:
            return None
        first, last = parts[0], parts[-1]
        df = playerid_lookup(last, first, fuzzy=True)
        if df is None or df.empty:
            return None
        val = df.iloc[0].get("key_mlbam")
        return int(val) if val and not (val != val) else None  # not NaN
    except Exception as exc:  # noqa: BLE001
        log.debug("playerid_lookup failed for %r (%s)", name, exc)
        return None


def _pitcher_features(reasoning: dict, game_date: str) -> tuple[dict, bool]:
    """Pitch-quality nets from AS-OF-DATE Statcast for the bet-time starters.
    Returns (features, ok). ok=False when either profile couldn't be built."""
    from mlb_value_bot.analysis.pitcher_metrics import build_pitcher_profile

    pitchers = reasoning.get("pitchers") or {}
    h_name, a_name = pitchers.get("home"), pitchers.get("away")
    season = int(game_date[:4])
    as_of = date_cls.fromisoformat(game_date)
    h_id, a_id = _pitcher_id(h_name), _pitcher_id(a_name)
    if h_id is None or a_id is None:
        return {k: 0.0 for k in ("whiff_net", "csw_net", "hardhit_net")}, False
    try:
        home_pp = build_pitcher_profile(h_id, h_name, season, as_of)
        away_pp = build_pitcher_profile(a_id, a_name, season, as_of)
    except Exception as exc:  # noqa: BLE001
        log.debug("profile build failed for %s/%s (%s)", h_name, a_name, exc)
        return {k: 0.0 for k in ("whiff_net", "csw_net", "hardhit_net")}, False
    pq = pitcher_quality_features(home_pp, away_pp)
    ok = any(v != 0.0 for v in pq.values())
    return pq, ok


def backfill(config: dict, limit: int | None = None) -> dict:
    """Backfill Stage-4 features for all graded games. Returns coverage stats."""
    from mlb_value_bot.tracking import recommendations as biff

    seen: set[tuple] = set()
    games: list[dict] = []
    for df in (biff.to_dataframe(), gtrack.to_dataframe()):
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            key = (r.get("date"), int(r.get("game_id")))
            if key in seen or r.get("result") not in ("win", "loss"):
                continue
            seen.add(key)
            games.append({"date": r["date"], "game_id": int(r["game_id"]),
                          "home_team": r["home_team"],
                          "reasoning": _reasoning(r.get("reasoning_json"))})
    games.sort(key=lambda g: g["date"])
    if limit:
        games = games[:limit]

    n = {"games": 0, "pitcher_ok": 0, "lineup_confirmed": 0, "weather_ok": 0}
    for g in games:
        reasoning = g["reasoning"]
        pq, pq_ok = _pitcher_features(reasoning, g["date"])
        lu = lineup_features_from_reasoning(reasoning)
        wx = weather_archive(g["home_team"], g["date"], config)
        feats = {k: 0.0 for k in GRIFF_FEATURE_KEYS}
        feats.update(pq)
        feats.update(lu)
        feats.update(wx)
        gtrack.upsert_features(g["date"], g["game_id"], feats, source="backfill")
        n["games"] += 1
        n["pitcher_ok"] += int(pq_ok)
        n["lineup_confirmed"] += int(lu["lineup_confirmed"] == 1.0)
        n["weather_ok"] += int(wx["temp"] != 0.0 or wx["wind"] != 0.0)
        log.info("backfilled %s game %s: pitcher_ok=%s lineup=%s wx_temp=%s",
                 g["date"], g["game_id"], pq_ok, lu["lineup_confirmed"], wx["temp"])
    return n
