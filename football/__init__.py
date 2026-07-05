"""Football — the American-football (NFL + college FBS) matchup-exploitation
model, run as a parallel engine beside BiffBet's MLB models.

Structural reuse of the repo's proven machinery BY IMPORT (EV math, de-vig,
odds client, cache, sharp/square grouping) with its OWN fully independent
config, its OWN persistence (a separate SQLite DB + separate Supabase tables:
football_recommendations / football_snapshot), and football-native analysis:

  * unit percentiles (0-100) computed WITHIN league (NFL pool and FBS pool are
    never mixed), quartile strong/weak classification,
  * matchup archetypes (pass O vs pass D, rush O vs run D, both directions,
    plus compound dual-edge / neutral cases) with an OL modifier layer,
  * projected spread + total blended 35/65 toward the de-vigged market,
  * spreads AND totals bet, PAPER-ONLY until CLV proves out (config hard gate).

HARD INVARIANTS (mirror GriffBet's):
  * Football never modifies MLB code. Shared pure code is imported, not copied.
  * Records are aggregated ONLY from football's own store, always filtered by
    model_tag x league x market (the GriffBet record-bug lesson, commit 3fba802).
  * The sharp-fade/RLM EV magnitude adjustment is single-homed in the pipeline's
    adjusted-EV step; stability may flag, never re-adjust.
"""
from __future__ import annotations

from pathlib import Path

# Independent config + storage, kept entirely separate from BiffBet's and
# GriffBet's. load_config(path) is lru-cached per path, so all configs coexist
# in one process without any reading another.
FOOTBALL_CONFIG_PATH = Path(__file__).resolve().parent / "config_football.yaml"
FOOTBALL_DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "football.db"


def load_football_config() -> dict:
    """Load football's independent config. Never reads config.yaml."""
    from mlb_value_bot.utils import load_config

    return load_config(str(FOOTBALL_CONFIG_PATH))


def season_for_date(date_iso: str) -> int:
    """Football season year for a game date (seasons span Aug-Jan: a January
    game belongs to the prior calendar year's season)."""
    year, month = int(date_iso[:4]), int(date_iso[5:7])
    return year if month >= 8 else year - 1
