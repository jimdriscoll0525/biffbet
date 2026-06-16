"""Local cache layer.

Two responsibilities:
  1. Parquet cache for DataFrames (pybaseball / Statcast pulls) with a TTL.
     Statcast pulls are slow and rate-limited, so caching is essential.
  2. A tiny key/value-ish helper for arbitrary JSON-able blobs (optional).

A cache entry is a single parquet file under storage/cache/, named from a
sanitized key. Freshness is judged by the file's modification time vs the
configured TTL (default 24h, per spec).
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from mlb_value_bot.data import fetch_ledger
from mlb_value_bot.utils import CACHE_DIR, ensure_dirs, get_logger, load_config

log = get_logger("data.cache")


def _ttl_seconds() -> float:
    hours = float(load_config().get("cache", {}).get("ttl_hours", 24))
    return hours * 3600.0


def _safe_filename(key: str) -> str:
    """Turn an arbitrary cache key into a filesystem-safe parquet filename.

    Readable prefix + short hash to avoid collisions and overlong names.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")[:80]
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{slug}__{digest}.parquet"


def cache_path(key: str) -> Path:
    return CACHE_DIR / _safe_filename(key)


def is_fresh(key: str, ttl_seconds: float | None = None) -> bool:
    """True if a cached parquet exists and is younger than the TTL."""
    path = cache_path(key)
    if not path.exists():
        return False
    ttl = _ttl_seconds() if ttl_seconds is None else ttl_seconds
    age = time.time() - path.stat().st_mtime
    return age < ttl


def load_cached(key: str, ttl_seconds: float | None = None) -> pd.DataFrame | None:
    """Return the cached DataFrame if present and fresh, else None."""
    if not is_fresh(key, ttl_seconds):
        return None
    path = cache_path(key)
    try:
        df = pd.read_parquet(path)
        log.debug("cache hit: %s (%d rows)", key, len(df))
        return df
    except Exception as exc:  # corrupt cache file -> treat as miss
        log.warning("cache read failed for %s: %s", key, exc)
        return None


def store_cached(key: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame to the parquet cache."""
    ensure_dirs()
    path = cache_path(key)
    try:
        df.to_parquet(path, index=False)
        log.debug("cached %d rows -> %s", len(df), path.name)
    except Exception as exc:
        log.warning("cache write failed for %s: %s", key, exc)


def cached_dataframe(
    key: str,
    producer: Callable[[], pd.DataFrame],
    ttl_seconds: float | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return cached df for `key`, or call `producer()` and cache the result.

    `producer` is only invoked on a miss (or when force_refresh=True). If it
    raises, a *stale* cached copy (if any) is returned as a fallback so a
    transient API outage doesn't take the whole bot down.
    """
    if not force_refresh:
        hit = load_cached(key, ttl_seconds)
        if hit is not None:
            fetch_ledger.record(key, "ok", key=key, detail="cache hit")
            return hit

    try:
        df = producer()
    except Exception as exc:
        outcome, status = fetch_ledger.classify_exception(exc)
        stale = cache_path(key)
        if stale.exists():
            log.warning("producer failed for %s (%s); using STALE cache", key, exc)
            fetch_ledger.record(key, outcome, http_status=status, detail=exc, stale=True, key=key)
            return pd.read_parquet(stale)
        # No cache to fall back on. Per the graceful-degradation contract, do NOT
        # crash the run: log loudly and return empty so the model degrades to
        # whatever data IS available (with correspondingly lower confidence).
        log.warning("producer failed for %s (%s); no cache -> degrading to empty", key, exc)
        fetch_ledger.record(key, outcome, http_status=status, detail=exc, key=key)
        return pd.DataFrame()

    if df is not None and not df.empty:
        store_cached(key, df)
        fetch_ledger.record(key, "ok", key=key)
        return df
    fetch_ledger.record(key, "empty", key=key)
    return df if df is not None else pd.DataFrame()


def clear_cache() -> int:
    """Delete all cached parquet files. Returns the count removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for f in CACHE_DIR.glob("*.parquet"):
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    log.info("cleared %d cache files", removed)
    return removed
