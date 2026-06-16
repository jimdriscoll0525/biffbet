"""Provenance classifier tests — no network. Run via pytest or
`python -m mlb_value_bot.tests.test_provenance`."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from mlb_value_bot.data import fetch_ledger
from mlb_value_bot.diagnostics import provenance as P


def _comp(name, available, note=""):
    return SimpleNamespace(name=name, available=available, note=note)


def _analysis(*, home_pitcher="Ace", away_pitcher="Mid", components=None,
              game_datetime=None, home_lu=None, away_lu=None):
    return SimpleNamespace(
        game_id=1, home_team="STL", away_team="SD",
        home_pitcher=home_pitcher, away_pitcher=away_pitcher,
        game_datetime=game_datetime,
        home_lineup_status=home_lu, away_lineup_status=away_lu,
        wp=SimpleNamespace(components=components or []),
    )


CFG = {"lineup": {"hours_before_first_pitch_to_check": 3.0}}


def test_tbd_starter_cascades_to_form():
    fetch_ledger.reset()
    a = _analysis(away_pitcher=None, components=[
        _comp("starter", False, "missing pitcher rate stat(s)"),
        _comp("form", False, "missing recent Statcast form"),
        _comp("home_field", True, "+0.025 home"),
    ])
    dh, prov = P.classify_analysis(a, CFG)
    assert prov["starter"]["reason_code"] == P.UPSTREAM
    assert prov["starter"]["reason_label"] == "PITCHER NOT NAMED"
    assert prov["form"]["reason_code"] == P.UPSTREAM
    assert prov["form"]["cascaded_from"] == "starter"      # tagged as cascade, not independent
    assert prov["home_field"]["reason_code"] == P.OK
    assert dh["expected_absences"] == 2 and dh["fetch_failures"] == 0


def test_named_starter_no_rate_is_fetch_failed_only_when_statcast_errors():
    # FanGraphs 403 is the baseline (fallback is Statcast) -> NOT a fetch failure.
    fetch_ledger.reset()
    fetch_ledger.record("fg_pitching_2026", "http_error", http_status=403, key="fg_pitching_2026")
    a = _analysis(home_pitcher="A", away_pitcher="B",
                  components=[_comp("starter", False, "missing pitcher rate stat(s)")])
    _, prov = P.classify_analysis(a, CFG)
    assert prov["starter"]["reason_code"] == P.NOT_APPLICABLE   # fg blocked is baseline; no rate = debut/unmatched

    # Statcast (the working fallback) hard-errors -> THIS is a real fetch failure.
    fetch_ledger.reset()
    fetch_ledger.record("statcast_pitcher_42_x", "timeout", key="statcast_pitcher_42_x")
    dh, prov2 = P.classify_analysis(a, CFG)
    assert prov2["starter"]["reason_code"] == P.FETCH_FAILED
    assert dh["fetch_failures"] == 1


def test_form_unavailable_with_available_starter_is_thin_sample():
    """The KC@WAS real case: starter rate resolved (Statcast worked) but form is
    missing -> a call-up's thin recent sample, NOT a fetch failure."""
    fetch_ledger.reset()
    a = _analysis(home_pitcher="A", away_pitcher="B", components=[
        _comp("starter", True, "rate H=4.28(statcast_rate) A=4.18(statcast_rate)"),
        _comp("form", False, "missing recent Statcast form"),
    ])
    dh, prov = P.classify_analysis(a, CFG)
    assert prov["form"]["reason_code"] == P.NOT_APPLICABLE
    assert prov["form"]["reason_label"] == "THIN RECENT DATA"
    assert dh["fetch_failures"] == 0                            # no false positive


def test_lineup_timing_three_cases():
    fetch_ledger.reset()
    now = datetime(2026, 6, 15, 18, 0, tzinfo=timezone.utc)  # ~2pm ET
    first_pitch = (now + timedelta(hours=5)).isoformat()      # 7pm-ish, well before window
    proj = SimpleNamespace(status="projected", is_confirmed=False, notes=[])
    # both projected, too early -> UPSTREAM (not posted yet)
    a = _analysis(game_datetime=first_pitch, home_lu=proj, away_lu=proj,
                  components=[_comp("lineup", False, "H projected; A projected")])
    _, prov = P.classify_analysis(a, CFG, now=now)
    assert prov["lineup"]["reason_code"] == P.UPSTREAM
    assert prov["lineup"]["reason_label"] == "LINEUP NOT POSTED YET"

    # feed unavailable on one side -> FETCH_FAILED
    unavail = SimpleNamespace(status="unavailable", is_confirmed=False, notes=["lineup feed unavailable"])
    a2 = _analysis(game_datetime=first_pitch, home_lu=proj, away_lu=unavail,
                   components=[_comp("lineup", False, "")])
    _, prov2 = P.classify_analysis(a2, CFG, now=now)
    assert prov2["lineup"]["reason_code"] == P.FETCH_FAILED

    # confirmed both -> component would be available; if reported available -> OK
    a3 = _analysis(components=[_comp("lineup", True, "confirmed")])
    _, prov3 = P.classify_analysis(a3, CFG, now=now)
    assert prov3["lineup"]["reason_code"] == P.OK


def test_park_bullpen_unavailable_is_fetch_failed():
    fetch_ledger.reset()
    fetch_ledger.record("fg_team_batting_2026", "http_error", http_status=403, key="fg_team_batting_2026")
    a = _analysis(components=[
        _comp("park", False, "missing team offense (wRC+)"),
        _comp("bullpen", False, "missing bullpen FIP"),
    ])
    dh, prov = P.classify_analysis(a, CFG)
    assert prov["park"]["reason_code"] == P.FETCH_FAILED
    assert prov["bullpen"]["reason_code"] == P.FETCH_FAILED
    assert dh["fetch_failures"] == 2


def test_merge_is_additive():
    """Merging adds data_health + per-component reason fields, preserving the
    existing component values (deltas untouched)."""
    reasoning = {"home_win_prob": 0.55, "components": [
        {"name": "starter", "weighted_delta": 0.0, "available": False, "note": "x"},
        {"name": "home_field", "weighted_delta": 0.025, "available": True, "note": "y"},
    ]}
    dh = {"expected_absences": 1, "fetch_failures": 0, "stale": 0, "ok": 1}
    per = {"starter": P._prov(P.UPSTREAM, "PITCHER NOT NAMED"),
           "home_field": P._prov(P.OK, "")}
    merged = P._merge_into_reasoning(reasoning, dh, per)
    assert merged["home_win_prob"] == 0.55                 # untouched
    assert merged["components"][0]["weighted_delta"] == 0.0  # delta untouched
    assert merged["components"][0]["reason_code"] == P.UPSTREAM
    assert merged["data_health"]["expected_absences"] == 1


def test_ledger_is_output_neutral():
    """Instrumented cached_dataframe returns byte-identical results; record()
    never raises."""
    import pandas as pd
    from mlb_value_bot.data import cache
    fetch_ledger.reset()
    out = cache.cached_dataframe("prov_neutral_raise",
                                 lambda: (_ for _ in ()).throw(RuntimeError("API 503 err")))
    assert out.empty and list(out.columns) == []           # unchanged degrade-to-empty
    assert any(e["outcome"] == "http_error" and e["http_status"] == 503
               for e in fetch_ledger.find("prov_neutral_raise"))
    out2 = cache.cached_dataframe("prov_neutral_ok", lambda: pd.DataFrame({"a": [1]}))
    assert len(out2) == 1                                    # data unchanged
    fetch_ledger.record(None, None, http_status="x")         # garbage -> must not raise


def _run_all():
    import inspect
    import sys
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith("test_")]
    for n, f in fns:
        f()
        print(f"  PASS  {n}")
    print(f"\n{len(fns)}/{len(fns)} provenance tests passed.")


if __name__ == "__main__":
    _run_all()
