"""RSI runtime state: the ACTIVE model version (tag + cumulative overlay) and
the APPROVED challengers running in shadow, per engine.

The engine reads exactly two things from Supabase at the top of every run:
rsi_model_versions(status='active') and rsi_proposals(status='approved').
load_state() never raises -- when Supabase is unreachable (or the credentials
are missing) it falls back to the last good snapshot in
storage/rsi/state_<engine>.json, and when there is no snapshot either it
returns the baseline (biff_v1 / totals_v1 / matchup_v1, no overlay, no
shadows). A network blip must never change what the champion prices.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mlb_value_bot.utils import STORAGE_DIR, get_logger

log = get_logger("rsi.state")

ENGINES = ("mlb", "mlb_totals", "football")
DEFAULT_TAGS = {"mlb": "biff_v1", "mlb_totals": "totals_v1", "football": "matchup_v1"}
STATE_DIR: Path = STORAGE_DIR / "rsi"
_DEFAULT_MAX_SHADOWS = 3


@dataclass
class ShadowProposal:
    id: int
    challenger_tag: str
    overlay: dict
    sport: str
    shadow_started_at: str | None = None
    shadow_ends_at: str | None = None


@dataclass
class RsiState:
    engine: str
    active_tag: str
    active_overlay: dict = field(default_factory=dict)
    shadow: list[ShadowProposal] = field(default_factory=list)
    source: str = "default"        # supabase | cache | default (diagnostic only)

    def to_json(self) -> dict:
        return {"engine": self.engine, "active_tag": self.active_tag,
                "active_overlay": self.active_overlay,
                "shadow": [asdict(p) for p in self.shadow]}

    @classmethod
    def from_json(cls, data: dict, source: str = "cache") -> "RsiState":
        shadows = [ShadowProposal(
            id=int(p["id"]), challenger_tag=str(p["challenger_tag"]),
            overlay=dict(p.get("overlay") or {}), sport=str(p.get("sport") or ""),
            shadow_started_at=p.get("shadow_started_at"), shadow_ends_at=p.get("shadow_ends_at"),
        ) for p in (data.get("shadow") or [])]
        return cls(engine=str(data["engine"]), active_tag=str(data["active_tag"]),
                   active_overlay=dict(data.get("active_overlay") or {}),
                   shadow=shadows, source=source)


def default_state(engine: str) -> RsiState:
    return RsiState(engine=engine, active_tag=DEFAULT_TAGS.get(engine, f"{engine}_v1"),
                    active_overlay={}, shadow=[], source="default")


def state_path(engine: str) -> Path:
    return STATE_DIR / f"state_{engine}.json"


def _max_shadows(cfg: dict | None) -> int:
    if cfg is None:
        try:
            from mlb_value_bot.rsi import load_rsi_config

            cfg = load_rsi_config()
        except Exception:  # noqa: BLE001 - config trouble must not block the run
            cfg = {}
    try:
        return max(0, int(cfg.get("max_shadows", _DEFAULT_MAX_SHADOWS)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_SHADOWS


def _fetch_state(engine: str, cap: int) -> RsiState:
    """The two reads. Raises on any Supabase trouble (caller degrades)."""
    from mlb_value_bot.rsi import supa

    versions = supa.get_rows("rsi_model_versions", {"engine": engine, "status": "active"},
                             order="promoted_at.desc", limit=1)
    if versions:
        active_tag = str(versions[0].get("tag") or DEFAULT_TAGS.get(engine, ""))
        active_overlay = dict(versions[0].get("overlay") or {})
    else:
        active_tag, active_overlay = DEFAULT_TAGS.get(engine, f"{engine}_v1"), {}

    approved = supa.get_rows("rsi_proposals", {"engine": engine, "status": "approved"},
                             order="id.asc")
    usable = [p for p in approved
              if p.get("challenger_tag") and isinstance(p.get("overlay"), dict) and p.get("overlay")]
    # Oldest approved first: decided_at, then id (a null decided_at sorts last).
    usable.sort(key=lambda p: (str(p.get("decided_at") or "9999"), int(p.get("id") or 0)))
    shadows = [ShadowProposal(
        id=int(p["id"]), challenger_tag=str(p["challenger_tag"]), overlay=dict(p["overlay"]),
        sport=str(p.get("sport") or engine),
        shadow_started_at=p.get("shadow_started_at"), shadow_ends_at=p.get("shadow_ends_at"),
    ) for p in usable[:cap]]
    if len(usable) > cap:
        log.warning("%s: %d approved challengers, capped to max_shadows=%d",
                    engine, len(usable), cap)
    return RsiState(engine=engine, active_tag=active_tag, active_overlay=active_overlay,
                    shadow=shadows, source="supabase")


def _write_cache(state: RsiState) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state_path(state.engine).write_text(json.dumps(state.to_json(), indent=2, default=str),
                                            encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not write RSI state cache for %s: %s", state.engine, exc)


def _read_cache(engine: str) -> RsiState | None:
    path = state_path(engine)
    try:
        if not path.exists():
            return None
        return RsiState.from_json(json.loads(path.read_text(encoding="utf-8")), source="cache")
    except Exception as exc:  # noqa: BLE001
        log.warning("unreadable RSI state cache %s: %s", path, exc)
        return None


def load_state(engine: str, cfg: dict | None = None) -> RsiState:
    """Active version + approved challengers for `engine`. NEVER raises:
    Supabase -> local cache -> baseline default, in that order."""
    if engine not in ENGINES:
        log.warning("unknown RSI engine %r; using a baseline state", engine)
        return default_state(engine)
    cap = _max_shadows(cfg)
    try:
        state = _fetch_state(engine, cap)
    except Exception as exc:  # noqa: BLE001 - degrade, never crash the slate
        log.warning("RSI state for %s unavailable from Supabase (%s); using cache/default",
                    engine, exc)
        cached = _read_cache(engine)
        if cached is not None:
            cached.shadow = cached.shadow[:cap]
            return cached
        return default_state(engine)
    _write_cache(state)
    return state
