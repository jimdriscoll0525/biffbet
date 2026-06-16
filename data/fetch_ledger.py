"""Output-neutral fetch ledger (diagnostic side-channel).

Records what happened at each external-fetch seam (cache producer, MLB Stats API,
Odds API) so the provenance classifier can distinguish a real FETCH_FAILED from a
legitimate UPSTREAM_ENTITY_MISSING. This is WRITE-ONLY side data: `record()` never
raises and never changes any caller's return value, so instrumenting the fetch
layer is provably model-output-neutral (BiffBet stays frozen).

Each entry: {source, outcome, http_status, stale, detail, key, ts}. Outcomes:
  ok | empty | stale | http_error | rate_limited | timeout | error | skipped
"""
from __future__ import annotations

import re
import threading
from datetime import datetime, timezone

_LOCK = threading.Lock()
_ENTRIES: list[dict] = []
_MAX = 5000  # bound memory in long-lived processes


def reset() -> None:
    """Clear the ledger (e.g. at the start of a slate run)."""
    with _LOCK:
        _ENTRIES.clear()


def record(source: str, outcome: str, *, http_status: int | None = None,
           detail: object | None = None, stale: bool = False,
           key: str | None = None) -> None:
    """Append one fetch outcome. MUST be side-effect-free w.r.t. the caller:
    swallows all of its own errors so instrumentation can never break a fetch."""
    try:
        entry = {
            "source": source,
            "outcome": outcome,
            "http_status": http_status,
            "stale": bool(stale),
            "detail": (str(detail)[:200] if detail is not None else None),
            "key": key,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with _LOCK:
            _ENTRIES.append(entry)
            if len(_ENTRIES) > _MAX:
                del _ENTRIES[: len(_ENTRIES) - _MAX]
    except Exception:  # noqa: BLE001 -- instrumentation must never affect the fetch
        pass


def entries() -> list[dict]:
    with _LOCK:
        return list(_ENTRIES)


def find(substr: str) -> list[dict]:
    """Entries whose source OR key contains `substr` (case-insensitive)."""
    s = substr.lower()
    out = []
    for e in entries():
        if s in (e.get("source") or "").lower() or s in (e.get("key") or "").lower():
            out.append(e)
    return out


def classify_exception(exc: BaseException) -> tuple[str, int | None]:
    """Map a swallowed fetch exception to (outcome, http_status). The HTTP status
    is only present in the exception message for the cache/pybaseball path, so we
    parse it; typed requests timeouts/connection errors are detected directly."""
    try:
        import requests
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return "timeout", None
    except Exception:  # noqa: BLE001
        pass
    msg = str(exc)
    m = re.search(r"\b(4\d\d|5\d\d)\b", msg)
    status = int(m.group(1)) if m else None
    if status == 429:
        return "rate_limited", 429
    if status is not None:
        return "http_error", status
    return "error", None
