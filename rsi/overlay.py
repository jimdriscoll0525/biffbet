"""Overlays: dotted-key config patches that a promoted proposal applies on top
of the base yaml. Pure functions -- no I/O.

    apply_overlay({"filters": {"min_model_prob": 0.5}}, {"filters.min_model_prob": 0.55})
    -> {"filters": {"min_model_prob": 0.55}}

A value of None DELETES the key (the ledger's "retire = set the key to null"
convention). Lists replace wholesale (never merged).
"""
from __future__ import annotations

import copy
from typing import Any

_SCALARS = (str, int, float, bool)


def _set_dotted(target: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    if value is None:
        node.pop(parts[-1], None)
    else:
        node[parts[-1]] = copy.deepcopy(value)


def apply_overlay(base: dict, overlay: dict | None) -> dict:
    """Return a deep copy of `base` with every dotted key of `overlay` set."""
    out = copy.deepcopy(base or {})
    for dotted, value in (overlay or {}).items():
        _set_dotted(out, dotted, value)
    return out


def _json_value(value: Any) -> bool:
    if value is None or isinstance(value, _SCALARS):
        return True
    if isinstance(value, list):
        return all(v is None or isinstance(v, _SCALARS) for v in value)
    return False


def validate_overlay(engine: str, overlay: dict | None, allowlist: dict) -> list[str]:
    """Problems with `overlay` for `engine` (empty list = valid)."""
    problems: list[str] = []
    if overlay is None:
        return problems
    if not isinstance(overlay, dict):
        return ["overlay must be a JSON object of dotted keys"]
    prefixes = [str(p) for p in (allowlist.get(engine) or [])]
    if not prefixes:
        problems.append(f"no overlay allowlist for engine '{engine}'")
    for key, value in overlay.items():
        if not isinstance(key, str) or not key or key.startswith(".") or key.endswith("."):
            problems.append(f"bad key {key!r}")
            continue
        if prefixes and not any(key == p.rstrip(".") or key.startswith(p) for p in prefixes):
            problems.append(f"key '{key}' is not in the {engine} allowlist")
        if not _json_value(value):
            problems.append(f"key '{key}' must be a JSON scalar or a list of scalars")
    return problems


def _flatten(node: Any, prefix: str, out: dict) -> None:
    if isinstance(node, dict) and node:
        for k, v in node.items():
            _flatten(v, f"{prefix}{k}.", out)
    else:
        out[prefix.rstrip(".")] = node


def flatten(cfg: dict) -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1} (lists are leaves)."""
    out: dict = {}
    for k, v in (cfg or {}).items():
        _flatten(v, f"{k}.", out)
    return out


def diff_overlay(base: dict, cfg: dict) -> dict:
    """Dotted keys whose value differs between `base` and `cfg` (cfg wins;
    keys missing from cfg map to None = delete)."""
    flat_base, flat_cfg = flatten(base), flatten(cfg)
    out: dict = {}
    for key in sorted(set(flat_base) | set(flat_cfg)):
        if key not in flat_cfg:
            out[key] = None
        elif key not in flat_base or flat_base[key] != flat_cfg[key]:
            out[key] = flat_cfg[key]
    return out
