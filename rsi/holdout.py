"""Deterministic 20% holdout split -- the Python twin of the SQL in the
rsi_scored_games view:

    (('x' || substr(md5('<sport_key>:<game_id>'), 1, 7))::bit(28)::int % 100) < 20

Seven hex chars = 28 bits, so the int is non-negative and both sides agree
exactly. The weekly review NEVER mines holdout games; the shadow gate uses them
as the untouched validation slice.
"""
from __future__ import annotations

import hashlib


def is_holdout(sport_key: str, game_id, pct: int = 20) -> bool:
    digest = hashlib.md5(f"{sport_key}:{game_id}".encode()).hexdigest()
    return int(digest[:7], 16) % 100 < pct


def sport_key_for(engine: str, sport: str | None = None, league: str | None = None) -> str:
    """The key the view hashes: 'mlb' | 'mlb_totals' | 'griffbet' | football league."""
    if engine == "football":
        return str(league or sport or "nfl")
    if engine in ("mlb", "mlb_totals", "griffbet"):
        return engine
    return str(sport or engine)
