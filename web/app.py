"""Streamlit web viewer for mlb_value_bot.

Launch with:  python -m mlb_value_bot serve   (or: streamlit run mlb_value_bot/web/app.py)

Design:
  * Reads the SQLite tracking DB by default (fast, free) — just viewing never
    spends an Odds API request.
  * The "Run / refresh analysis" button is the only thing that triggers the slow,
    metered pipeline (odds + Statcast). Results are cached in session_state.
  * Three views mirror the CLI: Today (slate + model breakdown), Results (P/L),
    Performance (ROI / CLV, segmented + charts).
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

# Make `import mlb_value_bot` work when Streamlit execs this file directly.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from mlb_value_bot.data.fangraphs_csv import status as fg_status
from mlb_value_bot.pipeline import analyze_slate, save_value_bets
from mlb_value_bot.tracking import performance as perf
from mlb_value_bot.tracking import recommendations as recs
from mlb_value_bot.tracking import results as results_mod
from mlb_value_bot.utils import get_bankroll, load_config

st.set_page_config(page_title="MLB Value Bot", page_icon="⚾", layout="wide")

CFG = load_config()
BANKROLL = get_bankroll()
THRESHOLD = float(CFG["ev"]["threshold"])
BLEND = CFG["model"].get("market_blend")


# --- formatting helpers ------------------------------------------------------
def _style(df: pd.DataFrame, fmt: dict, highlight_col: str | None = None):
    sty = df.style.format(fmt, na_rep="—")
    if highlight_col and highlight_col in df.columns:
        def _hl(row):
            color = "background-color: rgba(38,166,91,0.18)" if row.get(highlight_col) else ""
            return [color] * len(row)
        sty = sty.apply(_hl, axis=1)
    return sty


def _sidebar() -> str:
    st.sidebar.title("⚾ MLB Value Bot")
    page = st.sidebar.radio("View", ["Today", "Results", "Performance"], label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**EV threshold** {THRESHOLD*100:.1f}%  \n"
        f"**market_blend** {BLEND}  \n"
        f"**bankroll** {BANKROLL:,.0f}"
    )
    fg = fg_status(date.today().year)
    if fg["pitching"]["loaded"]:
        st.sidebar.success(f"FanGraphs pitching CSV loaded ({fg['pitching']['rows']} rows)")
    else:
        st.sidebar.warning(
            "No FanGraphs CSV → starter rates use the Statcast fallback; "
            "bullpen/park sit out. Add CSVs to storage/fangraphs/ (see `data-status`)."
        )
    st.sidebar.caption("Viewing reads the local DB. Only **Run analysis** spends an Odds API request.")
    return page


# --- Today -------------------------------------------------------------------
def page_today() -> None:
    st.header("Today's slate")
    c1, c2, c3 = st.columns([2, 2, 3])
    sel_date = c1.date_input("Game date", value=date.today())
    show_all = c2.checkbox("Show all games (not just +EV)", value=False)
    date_str = sel_date.isoformat()
    run = c3.button("▶ Run / refresh analysis", type="primary",
                    help="Pulls live odds + schedule + Statcast. Uses 1 Odds API request; first run for a date is slow.")

    if run:
        with st.spinner("Pulling odds + schedule + Statcast… (first run for a date can take a few minutes)"):
            try:
                analyses = analyze_slate(date_str, config=CFG)
                st.session_state[f"analyses_{date_str}"] = analyses
                value = [a for a in analyses if a.best_eval is not None and a.is_value(THRESHOLD)]
                saved = save_value_bets(value, date_str)
                st.session_state[f"saved_{date_str}"] = saved
                st.success(f"Analyzed {len([a for a in analyses if a.best_eval])} games · "
                           f"{len(value)} +EV · saved {saved} to the tracking DB.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed: {exc}")

    analyses = st.session_state.get(f"analyses_{date_str}")
    if analyses:
        _render_live_slate(analyses, show_all)
    else:
        df = recs.to_dataframe()
        df = df[df["date"] == date_str] if not df.empty else df
        if not df.empty:
            st.info("Showing saved recommendations from the DB. Click **Run / refresh analysis** for live odds.")
            _render_saved_recs(df)
        else:
            st.info("No analysis yet for this date — click **Run / refresh analysis**.")


def _render_live_slate(analyses, show_all: bool) -> None:
    evaluable = [a for a in analyses if a.best_eval is not None]
    value = [a for a in evaluable if a.is_value(THRESHOLD)]
    skipped = [a for a in analyses if a.best_eval is None]

    m1, m2, m3 = st.columns(3)
    m1.metric("Evaluable games", len(evaluable))
    m2.metric("+EV bets", len(value))
    m3.metric("Avg edge (+EV)", f"{(sum(a.best_eval.ev_pct for a in value)/len(value)*100):.1f}%" if value else "—")

    rows = []
    for a in (evaluable if show_all else value):
        be = a.best_eval
        pick = a.home_team if a.best_side == "home" else a.away_team
        rows.append({
            "Matchup": f"{a.away_team} @ {a.home_team}",
            "Pick": f"{pick} ({a.best_side})",
            "Odds": int(be.american_odds),
            "Model%": be.model_prob * 100,
            "Mkt%": be.market_prob_devigged * 100,
            "EV%": be.ev_pct * 100,
            "Kelly%": be.kelly_stake * 100,
            "Stake$": be.kelly_stake * BANKROLL,
            "Conf": a.confidence,
            "+EV": be.ev_pct >= THRESHOLD and be.kelly_stake > 0,
        })
    if rows:
        df = pd.DataFrame(rows)
        fmt = {"Odds": "{:+d}", "Model%": "{:.1f}", "Mkt%": "{:.1f}", "EV%": "{:+.1f}",
               "Kelly%": "{:.1f}", "Stake$": "{:,.0f}", "Conf": "{:.0f}"}
        st.dataframe(_style(df, fmt, highlight_col="+EV"), hide_index=True, width="stretch")
    else:
        st.info("No +EV bets at the current threshold. Toggle **Show all games** to see the full slate.")

    if value:
        st.subheader("Why these picks — model breakdown")
        for a in value:
            _render_breakdown(a)

    if skipped:
        st.subheader("Skipped games")
        st.dataframe(
            pd.DataFrame([{"Matchup": f"{a.away_team} @ {a.home_team}", "Reason": a.skipped_reason} for a in skipped]),
            hide_index=True, use_container_width=True,
        )


def _render_breakdown(a) -> None:
    be = a.best_eval
    title = f"{a.away_team} @ {a.home_team}  —  pick {a.best_side} @ {be.american_odds:+d}   ({be.ev_pct*100:+.1f}% EV · conf {a.confidence:.0f})"
    with st.expander(title):
        st.markdown(f"**Pitchers:** {a.away_pitcher or '?'} (away) vs {a.home_pitcher or '?'} (home)")
        st.markdown(f"Base (team strength via log5): **{a.wp.base_prob:.3f}** → raw model home **{a.wp.home_win_prob:.3f}**")
        comp = pd.DataFrame([{
            "Component": c.name, "Δ weighted": c.weighted_delta, "raw": c.raw_delta,
            "weight": c.weight, "available": c.available, "detail": c.note,
        } for c in a.wp.components])
        st.dataframe(
            comp.style.format({"Δ weighted": "{:+.4f}", "raw": "{:+.4f}", "weight": "{:g}"}, na_rep="—"),
            hide_index=True, use_container_width=True,
        )
        if a.market_home_prob is not None and a.blended_home_prob is not None:
            st.markdown(
                f"**Market anchor:** model {a.wp.home_win_prob:.3f} × {a.blend:g} + "
                f"market {a.market_home_prob:.3f} × {1 - a.blend:g} → "
                f"**{a.blended_home_prob:.3f}** (home win prob used for EV)"
            )


def _render_saved_recs(df: pd.DataFrame) -> None:
    view = pd.DataFrame({
        "Matchup": df["away_team"] + " @ " + df["home_team"],
        "Side": df["recommended_side"],
        "Odds": df["american_odds"].astype(int),
        "Model%": df["model_prob"] * 100,
        "Mkt%": df["market_prob_devigged"] * 100,
        "EV%": df["ev_pct"] * 100,
        "Kelly%": df["kelly_stake"] * 100,
        "Conf": df["confidence"],
        "Result": df["result"],
        "CLV%": df["clv_pct"],
    })
    fmt = {"Odds": "{:+d}", "Model%": "{:.1f}", "Mkt%": "{:.1f}", "EV%": "{:+.1f}",
           "Kelly%": "{:.1f}", "Conf": "{:.0f}", "CLV%": "{:+.2f}"}
    st.dataframe(_style(view, fmt), hide_index=True, width="stretch")


# --- Results -----------------------------------------------------------------
def page_results() -> None:
    st.header("Results & P/L")
    c1, c2 = st.columns([2, 3])
    sel = c1.date_input("Date", value=date.today() - timedelta(days=1))
    date_str = sel.isoformat()
    if c2.button("Grade this date (fetch final scores)", type="primary"):
        with st.spinner("Fetching final scores from the MLB Stats API…"):
            st.session_state[f"graded_{date_str}"] = results_mod.grade_date(date_str)

    summary = st.session_state.get(f"graded_{date_str}")
    if summary:
        roi = f"{summary.roi*100:.1f}%" if summary.staked > 0 else "—"
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Settled", summary.graded, f"{summary.wins}W-{summary.losses}L")
        k2.metric("P/L (units)", f"{summary.profit_loss:+.4f}")
        k3.metric("ROI", roi)
        k4.metric("Void / Pending", f"{summary.voids} / {summary.pending}")

    df = recs.to_dataframe()
    df = df[df["date"] == date_str] if not df.empty else df
    if not df.empty:
        st.subheader(f"Tracked bets on {date_str}")
        _render_saved_recs(df)
    else:
        st.info("No recommendations saved for this date. Run a slate on the Today page first.")


# --- Performance -------------------------------------------------------------
def page_performance() -> None:
    st.header("Performance")
    st.caption("ROI/hit-rate are noisy at low N — **CLV is the early signal** of real edge.")
    all_time = st.checkbox("All time", value=True)
    since = None
    if not all_time:
        since = st.date_input("Since", value=date.today() - timedelta(days=30)).isoformat()

    report = perf.compute_performance(since=since)
    o = report.overall
    if o.get("bets", 0) == 0:
        st.info("No recommendations recorded yet. Run a slate on the Today page first.")
        return

    def pct(x):
        return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x*100:.1f}%"

    k = st.columns(6)
    k[0].metric("Bets", o["bets"])
    k[1].metric("Settled", o["settled"], f"{o['wins']}W-{o['losses']}L")
    k[2].metric("Hit rate", pct(o["hit_rate"]))
    k[3].metric("Kelly ROI", pct(o["kelly_roi"]))
    k[4].metric("Flat ROI", pct(o["flat_roi"]))
    k[5].metric("Avg CLV", "—" if pd.isna(o["avg_clv_pct"]) else f"{o['avg_clv_pct']:+.2f}%")

    for title, seg in report.segments.items():
        if seg.empty:
            continue
        st.subheader(title)
        seg_col = seg.columns[0]
        view = pd.DataFrame({
            seg_col: seg[seg_col],
            "bets": seg["bets"], "settled": seg["settled"],
            "hit_rate": seg["hit_rate"] * 100,
            "kelly_roi": seg["kelly_roi"] * 100,
            "flat_roi": seg["flat_roi"] * 100,
            "avg_clv%": seg["avg_clv_pct"],
            "avg_ev%": seg["avg_ev_pct"],
        })
        fmt = {"hit_rate": "{:.1f}%", "kelly_roi": "{:+.1f}%", "flat_roi": "{:+.1f}%",
               "avg_clv%": "{:+.2f}", "avg_ev%": "{:.1f}"}
        ctab, cchart = st.columns([3, 2])
        ctab.dataframe(_style(view, fmt), hide_index=True, width="stretch")
        # Chart avg CLV by bucket (the most stable signal).
        chart_df = view[[seg_col, "avg_clv%"]].dropna().set_index(seg_col)
        if not chart_df.empty:
            cchart.caption("Avg CLV% by bucket")
            cchart.bar_chart(chart_df)


# --- main --------------------------------------------------------------------
PAGE = _sidebar()
if PAGE == "Today":
    page_today()
elif PAGE == "Results":
    page_results()
else:
    page_performance()
