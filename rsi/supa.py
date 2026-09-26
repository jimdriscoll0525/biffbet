"""Thin PostgREST access for the rsi_* tables and the rsi_scored_games view.

Reuses sync/supabase_sync.py's credential + header conventions (service key,
bypasses RLS -- the rsi_* tables have NO policies, so this is the only way
in). Filters use a tiny DSL:

    get_rows("rsi_proposals", {"engine": "mlb", "status": ("in", ["watch", "pending"])})
    get_rows("rsi_scored_games", {"date": ("gte", "2026-08-01")})

A plain value means `eq`; a (op, value) tuple maps to PostgREST `col=op.value`
(eq, neq, gt, gte, lt, lte, like, ilike, is, in). `in` takes a list.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd
import requests

from mlb_value_bot.sync.supabase_sync import _clean, _credentials, _headers, _post
from mlb_value_bot.utils import get_logger

log = get_logger("rsi.supa")

_PAGE = 1000
_TIMEOUT = 30

Filters = dict[str, Any] | None


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _quote(value: Any) -> str:
    s = _scalar(value)
    if any(ch in s for ch in ",() "):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _filter_params(filters: Filters) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []
    for col, spec in (filters or {}).items():
        if isinstance(spec, tuple) and len(spec) == 2:
            op, value = spec
        else:
            op, value = "eq", spec
        if op == "in":
            items = ",".join(_quote(v) for v in value)
            params.append((col, f"in.({items})"))
        elif value is None or op == "is":
            params.append((col, f"is.{'null' if value is None else _scalar(value)}"))
        else:
            params.append((col, f"{op}.{_scalar(value)}"))
    return params


def _read_headers(key: str) -> dict[str, str]:
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def get_rows(table: str, filters: Filters = None, select: str = "*",
             order: str | None = None, limit: int | None = None) -> list[dict]:
    """Read rows from `table` (or a view), paginating past the PostgREST cap."""
    url, key = _credentials()
    out: list[dict] = []
    offset = 0
    while True:
        page = _PAGE if limit is None else min(_PAGE, limit - len(out))
        if page <= 0:
            return out
        params: list[tuple[str, str]] = [("select", select)]
        params += _filter_params(filters)
        if order:
            params.append(("order", order))
        params += [("limit", str(page)), ("offset", str(offset))]
        resp = requests.get(f"{url}/rest/v1/{table}", params=params,
                            headers=_read_headers(key), timeout=_TIMEOUT)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"Supabase read from {table} failed ({resp.status_code}): {resp.text[:300]}")
        batch = resp.json()
        out.extend(batch)
        if len(batch) < page:
            return out
        offset += page


def upsert_rows(table: str, rows: list[dict], on_conflict: str) -> int:
    """Upsert `rows` (JSON-sanitised) merging on `on_conflict`; returns count.

    PostgREST rejects a bulk body whose objects carry different key sets
    (PGRST102 "All object keys must match"), and the review legitimately
    mixes full rows (new proposals) with partial patches (existing ones), so
    rows are grouped by key set and each group is posted on its own. Padding
    with nulls instead would overwrite existing values on merge-duplicates.
    """
    if not rows:
        return 0
    url, key = _credentials()
    groups: dict[tuple[str, ...], list[dict]] = {}
    for r in rows:
        clean = _clean(r)
        groups.setdefault(tuple(sorted(clean.keys())), []).append(clean)
    total = 0
    for group in groups.values():
        for start in range(0, len(group), 500):
            _post(url, key, table, group[start:start + 500], on_conflict=on_conflict)
        total += len(group)
    return total


def patch_rows(table: str, filters: Filters, fields: dict) -> None:
    """PATCH every row matching `filters` with `fields`."""
    url, key = _credentials()
    params = _filter_params(filters)
    if not params:
        raise ValueError("patch_rows refuses to run without a filter")
    resp = requests.patch(f"{url}/rest/v1/{table}", params=params,
                          headers=_headers(key, upsert_on=None),
                          data=json.dumps(_clean(fields), default=str), timeout=_TIMEOUT)
    if resp.status_code >= 300:
        raise RuntimeError(
            f"Supabase patch on {table} failed ({resp.status_code}): {resp.text[:300]}")


def insert_returning(table: str, row: dict) -> dict:
    """INSERT one row and return it (with its identity id)."""
    url, key = _credentials()
    headers = _headers(key, upsert_on=None)
    headers["Prefer"] = "return=representation"
    resp = requests.post(f"{url}/rest/v1/{table}", headers=headers,
                         data=json.dumps(_clean(row), default=str), timeout=_TIMEOUT)
    if resp.status_code >= 300:
        raise RuntimeError(
            f"Supabase insert into {table} failed ({resp.status_code}): {resp.text[:300]}")
    body = resp.json()
    return body[0] if isinstance(body, list) and body else (body or {})


def insert_rows(table: str, rows: Iterable[dict]) -> int:
    """Plain INSERT (no conflict merge) for append-only tables (events)."""
    rows = [_clean(r) for r in rows]
    if not rows:
        return 0
    url, key = _credentials()
    for start in range(0, len(rows), 500):
        resp = requests.post(f"{url}/rest/v1/{table}", headers=_headers(key, upsert_on=None),
                             data=json.dumps(rows[start:start + 500], default=str),
                             timeout=_TIMEOUT)
        if resp.status_code >= 300:
            raise RuntimeError(
                f"Supabase insert into {table} failed ({resp.status_code}): {resp.text[:300]}")
    return len(rows)


def fetch_scored_games(engine: str | None = None, since: str | None = None,
                       extra_filters: Filters = None) -> pd.DataFrame:
    """rsi_scored_games as a DataFrame (paged). `since` = date >= YYYY-MM-DD."""
    filters: dict[str, Any] = dict(extra_filters or {})
    if engine:
        filters["engine"] = engine
    if since:
        filters["date"] = ("gte", since)
    rows = get_rows("rsi_scored_games", filters, order="date.asc,game_key.asc")
    df = pd.DataFrame(rows)
    log.info("Fetched %d scored game(s) for engine=%s since=%s", len(df), engine, since)
    return df
