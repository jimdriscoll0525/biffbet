"""Effective (champion) config per engine = base yaml + the active version's
cumulative overlay, and the challenger config for a shadow proposal.

The two MLB engines (moneyline 'mlb' and 'mlb_totals') share config.yaml:
the mlb_totals overlay is restricted to `totals.` keys and the mlb overlay
to everything else (rsi/config_rsi.yaml overlay_allowlist), so both active
overlays are applied to the one shared base -- mlb's first, mlb_totals's
second. Football has its own yaml and its own single overlay.

Contract: with no overlay and no shadows the returned config is a deep copy
of the base with only the tag bookkeeping added (`model_tag` for football,
because football_store reads it; `rsi: {tag, totals_tag}` for MLB), so the
champion prices exactly what it always priced. Any failure degrades to
(base, default_tag) with a logged warning -- never an exception.
"""
from __future__ import annotations

import copy

from mlb_value_bot.rsi.overlay import apply_overlay
from mlb_value_bot.rsi.state import DEFAULT_TAGS, ShadowProposal, load_state
from mlb_value_bot.utils import get_logger

log = get_logger("rsi.config")


def _base(engine: str) -> dict:
    if engine == "football":
        from mlb_value_bot.football import load_football_config

        return load_football_config()
    from mlb_value_bot.utils import load_config

    return load_config()


def _stamp_default(cfg: dict, engine: str) -> tuple[dict, str]:
    tag = DEFAULT_TAGS.get(engine, f"{engine}_v1")
    if engine == "football":
        cfg["model_tag"] = tag
    else:
        cfg["rsi"] = {"tag": DEFAULT_TAGS["mlb"], "totals_tag": DEFAULT_TAGS["mlb_totals"]}
    return cfg, tag


def effective_config(engine: str) -> tuple[dict, str]:
    """(config with the active overlay applied, active tag) for `engine`
    ('mlb' | 'mlb_totals' | 'football'). Degrades to the plain base."""
    base = _base(engine)
    try:
        if engine == "football":
            st = load_state("football")
            cfg = apply_overlay(base, st.active_overlay)
            cfg["model_tag"] = st.active_tag
            return cfg, st.active_tag
        if engine not in ("mlb", "mlb_totals"):
            raise ValueError(f"unknown engine {engine!r}")
        ml, tot = load_state("mlb"), load_state("mlb_totals")
        cfg = apply_overlay(apply_overlay(base, ml.active_overlay), tot.active_overlay)
        cfg["rsi"] = {"tag": ml.active_tag, "totals_tag": tot.active_tag}
        return cfg, (ml.active_tag if engine == "mlb" else tot.active_tag)
    except Exception as exc:  # noqa: BLE001 - the slate must run on the base config
        log.warning("effective_config(%s) degraded to the base yaml: %s", engine, exc)
        return _stamp_default(copy.deepcopy(base), engine)


def challenger_config(cfg: dict, proposal: ShadowProposal) -> dict:
    """The champion config with the proposal's overlay on top, re-tagged with
    the challenger tag (model_tag for football, rsi.tag / rsi.totals_tag for
    the MLB engines, by the proposal's sport)."""
    out = apply_overlay(cfg, proposal.overlay)
    if proposal.sport in ("mlb", "mlb_totals"):
        rsi = dict(out.get("rsi") or {})
        rsi["totals_tag" if proposal.sport == "mlb_totals" else "tag"] = proposal.challenger_tag
        out["rsi"] = rsi
    else:
        out["model_tag"] = proposal.challenger_tag
    return out
