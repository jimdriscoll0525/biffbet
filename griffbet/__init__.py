"""GriffBet — a challenger model that competes head-to-head against BiffBet.

GriffBet is a STRUCTURAL re-use of BiffBet's transparent, hand-weighted,
market-blended model with its OWN fully independent config, its OWN persistence
(a separate SQLite DB + separate Supabase tables), and a few disciplined
differences:

  * a starter-neutralized base rate (config flag, default OFF -- §B of the plan),
  * a slate-level exposure cap + correlation haircut on top of per-bet Kelly,
  * a richer CLV split (raw-model vs blended pick streams, graded against the
    SHARP closing line, with the best-available close kept as "obtainable"),
  * the dead "-2pp fade past 5pp" adjusted-EV haircut omitted.

HARD INVARIANT: GriffBet never modifies BiffBet. It reuses BiffBet's pure,
model-agnostic code BY IMPORT (EV math, de-vig, data clients, metric providers,
the pipeline's pure helpers) and forks ONLY what must diverge (the base-rate
toggle, the orchestration glue, persistence, and sync). With the neutralization
flag OFF and an identical config, GriffBet's win probabilities are identical to
BiffBet's -- locked by tests/test_griffbet.py::test_golden_equivalence_*.
"""
from __future__ import annotations

from pathlib import Path

# GriffBet's independent config + its own storage, kept entirely separate from
# BiffBet's. load_config(path) is lru-cached per path, so both configs coexist
# in one process without either reading the other.
GRIFF_CONFIG_PATH = Path(__file__).resolve().parent / "config_griff.yaml"
GRIFF_DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "griffbet.db"


def load_griff_config() -> dict:
    """Load GriffBet's independent config. Never reads BiffBet's config.yaml."""
    from mlb_value_bot.utils import load_config

    return load_config(str(GRIFF_CONFIG_PATH))
