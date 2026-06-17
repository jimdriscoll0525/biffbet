"""Command-line interface for mlb_value_bot.

Commands:
  today        Pull today's slate, run the model, print ranked +EV table, save recs.
  results      Fetch final scores, settle open bets, print daily P/L.
  performance  ROI / hit rate / CLV, overall and segmented.
  backtest     Re-run the model over historical games (CSV odds fallback).
  clear-cache  Drop cached pybaseball/Statcast parquet files.

Run as a module:  python -m mlb_value_bot <command> [options]
"""
from __future__ import annotations

import math
import sys
from datetime import date

# Windows legacy consoles default to cp1252; force UTF-8 so rich tables and any
# unicode render cleanly. No-op on platforms that already use UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mlb_value_bot.pipeline import (  # save_value_bets kept for back-compat imports
    GameAnalysis,
    analyze_slate,
    flag_starter_scratches,
    refresh_skipped_closing_lines,
    save_slate,
    save_value_bets,
)
from mlb_value_bot.tracking import performance as perf
from mlb_value_bot.tracking import results as results_mod
from mlb_value_bot.utils import get_bankroll, get_logger, load_config, setup_logging

console = Console()
log = get_logger("cli")


def _fmt_pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x * 100:.{digits}f}%"


def _fmt_num(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{digits}f}"


def _american(x: int | None) -> str:
    return "-" if x is None else f"{x:+d}"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """+EV MLB betting assistant (starting-pitcher focused)."""
    setup_logging()


# ---------------------------------------------------------------------------
# today
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--date", "date_", default=None, help="Game date YYYY-MM-DD (default: today).")
@click.option("--save/--no-save", default=True, help="Save +EV bets to the tracking DB.")
@click.option("--min-ev", type=float, default=None, help="Override EV threshold (e.g. 0.03).")
@click.option("--all", "show_all", is_flag=True, help="Show every game, not just +EV ones.")
@click.option("--market-blend", type=float, default=None,
              help="Override the model/market blend weight (0=market only, 1=model only).")
def today(date_: str | None, save: bool, min_ev: float | None, show_all: bool,
          market_blend: float | None) -> None:
    """Analyze today's slate and print the ranked +EV table."""
    config = load_config()
    if market_blend is not None:
        config = {**config, "model": {**config["model"], "market_blend": market_blend}}
    game_date = date_ or date.today().isoformat()
    threshold = min_ev if min_ev is not None else float(config["ev"]["threshold"])
    bankroll = get_bankroll()

    # Diagnostic provenance: clear the output-neutral fetch ledger so it reflects
    # only this run (used after save to label WHY any component is unavailable).
    from mlb_value_bot.data import fetch_ledger
    fetch_ledger.reset()

    try:
        analyses = analyze_slate(game_date, config=config)
    except Exception as exc:
        console.print(f"[bold red]Failed to analyze slate:[/] {exc}")
        log.exception("analyze_slate failed")
        raise SystemExit(1)

    evaluable = [a for a in analyses if a.best_eval is not None]
    value_bets = [a for a in evaluable if a.is_value(threshold)]
    skipped = [a for a in analyses if a.best_eval is None]

    _render_slate_table(evaluable if show_all else value_bets, threshold, bankroll, game_date)
    _render_value_breakdowns(value_bets)
    _render_skipped(skipped)

    console.print(
        f"\n[bold]{len(value_bets)}[/] +EV bet(s) at EV >= {threshold * 100:.1f}% "
        f"out of {len(evaluable)} evaluable game(s) on {game_date}."
    )

    # Nudge toward the CSV workflow if FanGraphs metrics aren't available.
    from mlb_value_bot.data.fangraphs_csv import status as fg_status

    if not fg_status(int(game_date[:4]))["pitching"]["loaded"]:
        console.print(
            "[dim]No FanGraphs pitching CSV found -> starter rates use the Statcast fallback, "
            "and bullpen/park sit out. Run `data-status` for how to add it.[/]"
        )

    if save and evaluable:
        total, n_value = save_slate(evaluable, threshold, game_date)
        console.print(
            f"[green]Saved/updated {total} slate row(s) "
            f"({n_value} flagged +EV) to the tracking DB.[/]"
        )
    elif save:
        console.print("[dim]Nothing to save (no evaluable games).[/]")

    # Post-save maintenance over the FULL slate (skipped games included --
    # a scratched game usually trips the divergence guard, so the skipped
    # list is where it lands):
    #   * scratch detection annotates committed bets whose probable starter
    #     changed since commit (the bets themselves stay frozen);
    #   * closing-line refresh keeps CLV moving on committed bets whose game
    #     was sanity-skipped this run (a skip must not freeze the close).
    if save and analyses:
        n_scratches = flag_starter_scratches(analyses, game_date)
        if n_scratches:
            console.print(
                f"[bold red]{n_scratches} starter change(s) detected on committed "
                f"pick(s) -- their edge basis is stale. See log / site for details.[/]"
            )
        n_refreshed = refresh_skipped_closing_lines(analyses, game_date)
        if n_refreshed:
            console.print(
                f"[dim]Refreshed closing line on {n_refreshed} committed pick(s) "
                f"whose game was skipped this run.[/]"
            )

    # Data-provenance: label WHY each unavailable component is missing
    # (legitimate absence vs real fetch failure) and merge an additive
    # data_health block into reasoning_json. Purely diagnostic -- never changes a
    # probability, component value, EV, or pick. Guarded so it can't break a run.
    if save and evaluable:
        try:
            from mlb_value_bot.diagnostics.provenance import annotate_slate
            from mlb_value_bot.utils import DB_PATH
            health = annotate_slate(evaluable, DB_PATH, "recommendations", game_date, config)
            if health["fetch_failures"] or health["stale"]:
                console.print(
                    f"[bold red]Data health: {health['fetch_failures']} fetch failure(s), "
                    f"{health['stale']} stale across {health['games']} game(s).[/] "
                    f"{health['details']}"
                )
            else:
                console.print(
                    f"[dim]Data health: all absences across {health['games']} game(s) are "
                    f"legitimate (no fetch failures).[/]"
                )
        except Exception as exc:  # noqa: BLE001 -- diagnostics must never break a run
            log.warning("provenance annotation failed (%s)", exc)

    # Totals (over/under) -- a parallel, PAPER-ONLY market. Rendered + saved to
    # its own tracking table; graded on CLV vs the totals close. Never affects
    # the moneyline output above.
    _render_totals(analyses, config, bankroll, game_date, save, show_all)


def _render_slate_table(analyses: list[GameAnalysis], threshold: float, bankroll: float, game_date: str) -> None:
    title = f"MLB slate — {game_date}  (bankroll {bankroll:.0f}, EV threshold {threshold*100:.1f}%)"
    table = Table(title=title, header_style="bold cyan", expand=False)
    for col in ("Matchup", "Pick", "Odds", "Model%", "Mkt%", "Raw EV%", "Adj EV%", "Kelly%", "Stake", "Conf"):
        table.add_column(col)

    if not analyses:
        console.print(table)
        console.print("[yellow]No games to display.[/]")
        return

    for a in analyses:
        be = a.best_eval
        assert be is not None
        pick_team = a.home_team if a.best_side == "home" else a.away_team
        is_value = be.ev_pct >= threshold and be.kelly_stake > 0
        style = "bold green" if is_value else "white"
        stake_dollars = be.kelly_stake * bankroll
        # Adjusted EV (Step 4) drives sizing; Raw EV shown alongside unchanged.
        adj_ev = a.adjusted_ev_pct if a.adjusted_ev_pct is not None else be.ev_pct
        table.add_row(
            f"{a.away_team} @ {a.home_team}",
            f"[{style}]{pick_team} ({a.best_side})[/]",
            _american(be.american_odds),
            _fmt_pct(be.model_prob),
            _fmt_pct(be.market_prob_devigged),
            f"{be.ev_pct*100:+.1f}%",
            f"[{style}]{adj_ev*100:+.1f}%[/]",
            _fmt_pct(be.kelly_stake),
            f"{stake_dollars:,.0f}",
            _fmt_num(a.confidence, 0),
        )
    console.print(table)


def _render_value_breakdowns(value_bets: list[GameAnalysis]) -> None:
    """Show the per-component model reasoning for each +EV bet (transparency)."""
    for a in value_bets:
        if not a.wp:
            continue
        lines = [
            f"[bold]{a.away_team} @ {a.home_team}[/]  "
            f"pick [green]{a.best_side}[/]  base {a.wp.base_prob:.3f} -> raw model {a.wp.home_win_prob:.3f} (home)",
            f"  pitchers: {a.away_pitcher or '?'} (A) vs {a.home_pitcher or '?'} (H)",
        ]
        for c in a.wp.components:
            flag = "" if c.available else "  [yellow](missing)[/]"
            lines.append(
                f"  - {c.name:<11} d={c.weighted_delta:+.4f} (raw {c.raw_delta:+.4f} x w{c.weight:g})  {c.note}{flag}"
            )
        if a.market_home_prob is not None and a.blended_home_prob is not None:
            lines.append(
                f"  = market blend: model {a.wp.home_win_prob:.3f} x {a.blend:g} + "
                f"market {a.market_home_prob:.3f} x {1 - a.blend:g} "
                f"-> [bold]{a.blended_home_prob:.3f}[/] (home, used for EV)"
            )
        # Edge stability badge (Step 3) + Raw/Adjusted EV (Step 4).
        be = a.best_eval
        if a.stability is not None:
            label = a.stability.label.upper()
            color = {"STABLE": "green", "MODERATE": "yellow", "FRAGILE": "red"}.get(label, "white")
            lines.append(f"  edge stability: [{color}]{label}[/]  tier [bold]{a.tier}[/]")
        if be is not None and a.adjusted_ev_pct is not None:
            adj_note = ("; ".join(a.adjusted_ev_reasons)) or "no adjustments"
            lines.append(
                f"  EV: raw [bold]{be.ev_pct*100:+.1f}%[/] -> adjusted "
                f"[bold]{a.adjusted_ev_pct*100:+.1f}%[/]  ({adj_note})"
            )
        console.print(Panel("\n".join(lines), border_style="green", expand=False))


def _render_skipped(skipped: list[GameAnalysis]) -> None:
    if not skipped:
        return
    table = Table(title="Skipped games", header_style="dim", expand=False)
    table.add_column("Matchup")
    table.add_column("Reason")
    for a in skipped:
        table.add_row(f"{a.away_team} @ {a.home_team}", a.skipped_reason or "?")
    console.print(table)


# ---------------------------------------------------------------------------
# Totals (over/under) rendering -- a PARALLEL, PAPER-ONLY section.
# ---------------------------------------------------------------------------
def _render_totals_table(totals: list, threshold: float, bankroll: float, game_date: str) -> None:
    paper = any(t.paper for t in totals) if totals else True
    tag = "  [yellow](PAPER / SIMULATED)[/]" if paper else ""
    table = Table(title=f"TOTALS slate — {game_date}{tag}", header_style="bold magenta", expand=False)
    for col in ("Matchup", "Pick", "Line", "Price", "ProjTot", "Mdl o/u%", "Mkt o/u%", "EV%",
                "Kelly%", "Stake", "Conf", "Stab"):
        table.add_column(col)
    for t in totals:
        be = t.best_eval
        if be is None:
            continue
        is_value = t.is_value(threshold)
        style = "bold green" if is_value else "white"
        held = "  [yellow]⚠wx[/]" if t.weather_held else ""
        model_side = t.blended_over if t.pick_side == "over" else (
            1 - t.blended_over if t.blended_over is not None else None)
        mkt_side = t.market_devig_over if t.pick_side == "over" else (
            1 - t.market_devig_over if t.market_devig_over is not None else None)
        table.add_row(
            f"{t.away_team} @ {t.home_team}",
            f"[{style}]{t.pick_side}{held}[/]",
            _fmt_num(t.market_total, 1),
            _american(be.american_odds),
            _fmt_num(t.rd.expected_total, 1) if t.rd else "-",
            _fmt_pct(model_side),
            _fmt_pct(mkt_side),
            f"[{style}]{be.ev_pct * 100:+.1f}%[/]",
            _fmt_pct(be.kelly_stake),
            f"{be.kelly_stake * bankroll:,.0f}",
            _fmt_num(t.confidence, 0),
            (t.stability.label[:4] if t.stability else "-"),
        )
    console.print(table)


def _render_totals_breakdowns(totals: list) -> None:
    for t in totals:
        be = t.best_eval
        if be is None or t.rd is None:
            continue
        rd = t.rd
        lines = [
            f"[bold]{t.away_team} @ {t.home_team}[/]  pick [green]{t.pick_side} {t.market_total}[/] "
            f"@ {be.american_odds:+d}   [yellow]{'PAPER' if t.paper else 'LIVE'}[/]",
            f"  proj runs: away {rd.away_runs} – home {rd.home_runs}  | raw total {rd.raw_model_total} "
            f"-> expected {rd.expected_total} (var {rd.variance})",
        ]
        for c in rd.components:
            extra = c.get("runs", c.get("mult", ""))
            lines.append(f"  - {c['name']:<12} {c.get('value','')}  {extra}")
        if t.market_devig_over is not None:
            lines.append(
                f"  = blend: model P(over) {t.model_p_over} × {t.blend:g} + market {t.market_devig_over} × "
                f"{1 - t.blend:g} -> [bold]{t.blended_over}[/]  (P(over), {t.blend_tier} conf {t.confidence:.0f})"
            )
        lines.append(
            f"  EV {be.ev_pct * 100:+.1f}%  tier [bold]{t.tier}[/]  "
            f"stability [bold]{t.stability.label.upper() if t.stability else '?'}[/]"
            + (f"  signals: {', '.join(t.stability.hard_fragile_signals)}"
               if t.stability and t.stability.hard_fragile_signals else "")
        )
        if t.sharp_close is not None:
            lines.append(
                f"  sharp close: {t.sharp_close.book} {t.sharp_close.line} "
                f"(P(over) {t.sharp_close.devig_over}) — CLV graded vs this"
            )
        if t.flags:
            lines.append(f"  [yellow]flags: {'; '.join(t.flags)}[/]")
        console.print(Panel("\n".join(lines), border_style="magenta", expand=False))


def _render_totals(analyses: list[GameAnalysis], config: dict, bankroll: float,
                   game_date: str, save: bool, show_all: bool) -> None:
    """Render + persist the totals slate (PAPER). No-op if totals disabled."""
    if not config.get("totals", {}).get("enabled", False):
        return
    totals = [a.totals for a in analyses if getattr(a, "totals", None) is not None]
    evaluable = [t for t in totals if t.best_eval is not None and not t.skipped_reason]
    threshold = float(config["totals"].get("ev_threshold", 0.03))
    value = [t for t in evaluable if t.is_value(threshold)]
    skipped = [t for t in totals if t.best_eval is None or t.skipped_reason]

    console.print()
    _render_totals_table(evaluable if show_all else value, threshold, bankroll, game_date)
    _render_totals_breakdowns(value)
    if skipped:
        st = Table(title="Skipped totals", header_style="dim", expand=False)
        st.add_column("Matchup"); st.add_column("Reason")
        for t in skipped:
            st.add_row(f"{t.away_team} @ {t.home_team}", t.skipped_reason or "no totals market")
        console.print(st)

    paper = config["totals"].get("paper_only", True)
    console.print(
        f"[bold]{len(value)}[/] totals pick(s) at EV >= {threshold * 100:.1f}% "
        f"out of {len(evaluable)} evaluable — [yellow]{'PAPER/SIMULATED' if paper else 'LIVE'}[/] "
        f"(graded on CLV vs the totals close)."
    )

    if save and evaluable:
        from mlb_value_bot.pipeline_totals import refresh_skipped_totals_closing, save_totals_slate
        total, n_value = save_totals_slate(evaluable, threshold, game_date)
        console.print(f"[green]Saved/updated {total} totals row(s) ({n_value} flagged value) to the tracking DB.[/]")
        n_ref = refresh_skipped_totals_closing(totals, game_date)
        if n_ref:
            console.print(f"[dim]Refreshed totals close on {n_ref} committed paper pick(s) on skipped games.[/]")


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--date", "date_", default=None,
              help="Game date YYYY-MM-DD (default: sweep ALL past dates with open bets).")
def results(date_: str | None) -> None:
    """Fetch final scores and settle open bets.

    With --date, grades that single date. Without it, sweeps every past date
    that still has pending bets, so a missed grading run (failed pipeline,
    suspended game, pre-grading rows) self-heals on the next run.
    """
    if date_:
        summaries = [results_mod.grade_date(date_)]
    else:
        summaries = results_mod.grade_all_open(before=date.today().isoformat())
        if not summaries:
            console.print("[dim]No past dates with open bets.[/]")
            return

    for summary in summaries:
        table = Table(title=f"Results — {summary.date}", header_style="bold cyan")
        for col in ("Matchup", "Pick", "Result", "P/L (units)"):
            table.add_column(col)
        for b in summary.bets:
            color = "green" if b.result == "win" else ("red" if b.result == "loss" else "yellow")
            table.add_row(b.matchup, b.side, f"[{color}]{b.result}[/]", f"[{color}]{b.profit_loss:+.4f}[/]")
        console.print(table)

        roi_txt = _fmt_pct(summary.roi) if summary.staked > 0 else "-"
        console.print(Panel(
            f"Settled: [bold]{summary.graded}[/]  ({summary.wins}W-{summary.losses}L)   "
            f"Void: {summary.voids}   Pending: {summary.pending}\n"
            f"Staked: {summary.staked:.4f} units   "
            f"P/L: [bold]{summary.profit_loss:+.4f}[/] units   ROI: [bold]{roi_txt}[/]",
            title=f"Daily P/L — {summary.date}", border_style="cyan", expand=False,
        ))

    if len(summaries) > 1:
        graded = sum(s.graded for s in summaries)
        wins = sum(s.wins for s in summaries)
        losses = sum(s.losses for s in summaries)
        pl = sum(s.profit_loss for s in summaries)
        pending = sum(s.pending for s in summaries)
        console.print(Panel(
            f"Dates swept: [bold]{len(summaries)}[/]   Settled: [bold]{graded}[/]  "
            f"({wins}W-{losses}L)   Still pending: {pending}   "
            f"P/L: [bold]{pl:+.4f}[/] units",
            title="Backfill total", border_style="magenta", expand=False,
        ))

    # Totals (PAPER) grading -- separate table, separate ledger.
    if load_config().get("totals", {}).get("enabled", False):
        if date_:
            t_summaries = [results_mod.grade_totals_date(date_)]
        else:
            t_summaries = results_mod.grade_all_open_totals(before=date.today().isoformat())
        t_graded = sum(s.graded for s in t_summaries)
        if t_graded or any(s.bets for s in t_summaries):
            t_wins = sum(s.wins for s in t_summaries)
            t_losses = sum(s.losses for s in t_summaries)
            t_pl = sum(s.profit_loss for s in t_summaries)
            console.print(Panel(
                f"Totals settled (PAPER): [bold]{t_graded}[/]  ({t_wins}W-{t_losses}L)   "
                f"Paper P/L: [bold]{t_pl:+.4f}[/] units\n"
                f"[dim]Totals are graded on CLV vs the totals close, not record — see `performance`.[/]",
                title="Totals — PAPER results", border_style="magenta", expand=False,
            ))


# ---------------------------------------------------------------------------
# performance
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--since", default=None, help="Only include games on/after this date (YYYY-MM-DD).")
def performance(since: str | None) -> None:
    """ROI, hit rate, and CLV - overall and segmented."""
    report = perf.compute_performance(since=since)
    o = report.overall
    if o.get("bets", 0) == 0:
        console.print("[yellow]No recommendations recorded yet. Run `today` first.[/]")
        return

    console.print(Panel(
        f"Bets: [bold]{o['bets']}[/]   Settled: [bold]{o['settled']}[/]  "
        f"({o['wins']}W-{o['losses']}L)\n"
        f"Hit rate: [bold]{_fmt_pct(o['hit_rate'])}[/]   Avg EV: {_fmt_num(o['avg_ev_pct'])}%\n"
        f"Kelly ROI: [bold]{_fmt_pct(o['kelly_roi'])}[/]  (P/L {o['kelly_pl_units']:+.4f} units)\n"
        f"Flat ROI:  [bold]{_fmt_pct(o['flat_roi'])}[/]  (P/L {o['flat_pl_units']:+.2f}u over {o['settled']} bets)\n"
        f"Avg CLV: [bold]{_fmt_num(o['avg_clv_pct'])}%[/]  (tracked on {o['clv_tracked']} bets)",
        title=f"Overall performance{' since ' + since if since else ''}", border_style="cyan", expand=False,
    ))

    for title, seg_df in report.segments.items():
        if seg_df.empty:
            continue
        _render_segment(title, seg_df)

    _render_totals_performance(since)


def _render_totals_performance(since: str | None) -> None:
    """Totals (PAPER) performance -- CLV vs the totals close is the headline."""
    if not load_config().get("totals", {}).get("enabled", False):
        return
    from mlb_value_bot.tracking import totals_performance as tperf

    report = tperf.compute_totals_performance(since=since)
    o = report.overall
    if o.get("bets", 0) == 0:
        return
    console.print(Panel(
        f"Paper bets: [bold]{o['bets']}[/]   Settled: [bold]{o['settled']}[/]  "
        f"({o['wins']}W-{o['losses']}L, {o.get('pushes', 0)} push)\n"
        f"[bold]Avg CLV: {_fmt_num(o['avg_clv_pp'])} pp[/]  (vs sharp totals close, on {o['clv_tracked']} picks) "
        f"— [dim]the gate to go live[/]\n"
        f"CLV+ rate: {_fmt_pct(o.get('clv_positive_rate'))}   "
        f"Paper Kelly ROI: {_fmt_pct(o['kelly_roi'])}   Hit rate: {_fmt_pct(o['hit_rate'])}",
        title=f"TOTALS performance (PAPER){' since ' + since if since else ''}",
        border_style="magenta", expand=False,
    ))
    for title, seg_df in report.segments.items():
        if seg_df.empty:
            continue
        _render_totals_segment(title, seg_df)


def _render_totals_segment(title: str, df) -> None:
    table = Table(title=f"[totals] {title}", header_style="bold magenta")
    seg_col = df.columns[0]
    for col in (seg_col, "bets", "settled", "wins", "losses", "avg_clv_pp", "kelly_roi", "avg_ev_pct"):
        if col in df.columns:
            table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(
            str(row[seg_col]), str(int(row["bets"])), str(int(row["settled"])),
            str(int(row["wins"])), str(int(row["losses"])),
            _fmt_num(row["avg_clv_pp"]), _fmt_pct(row["kelly_roi"]), _fmt_num(row["avg_ev_pct"]),
        )
    console.print(table)


def _render_segment(title: str, df) -> None:
    table = Table(title=title, header_style="bold magenta")
    seg_col = df.columns[0]
    for col in (seg_col, "bets", "settled", "hit_rate", "kelly_roi", "flat_roi", "avg_clv_pct", "avg_ev_pct"):
        table.add_column(col)
    for _, row in df.iterrows():
        table.add_row(
            str(row[seg_col]),
            str(int(row["bets"])),
            str(int(row["settled"])),
            _fmt_pct(row["hit_rate"]),
            _fmt_pct(row["kelly_roi"]),
            _fmt_pct(row["flat_roi"]),
            _fmt_num(row["avg_clv_pct"]),
            _fmt_num(row["avg_ev_pct"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--start", required=True, help="Start date YYYY-MM-DD.")
@click.option("--end", required=True, help="End date YYYY-MM-DD.")
@click.option("--csv", "csv_path", required=True, help="CSV of historical odds (see README).")
@click.option("--market-blend", type=float, default=None,
              help="Override the model/market blend weight for this backtest (e.g. 0.25 vs 0.35).")
def backtest(start: str, end: str, csv_path: str, market_blend: float | None) -> None:
    """Re-run the model over historical games using a CSV of odds."""
    from mlb_value_bot.backtest.backtester import run_backtest

    config = load_config()
    if market_blend is not None:
        config = {**config, "model": {**config["model"], "market_blend": market_blend}}
    result = run_backtest(start, end, csv_path, config=config)
    s = result.summary
    if s.get("bets", 0) == 0:
        console.print("[yellow]No qualifying +EV bets in the backtest window.[/]")
        return

    console.print(Panel(
        f"Bets: [bold]{s['bets']}[/]  ({s['wins']}W-{s['losses']}L)  "
        f"Hit rate: [bold]{_fmt_pct(s['hit_rate'])}[/]\n"
        f"Avg EV: {_fmt_num(s['avg_ev_pct'])}%\n"
        f"Kelly ROI: [bold]{_fmt_pct(s['kelly_roi'])}[/]  (P/L {s['kelly_pl_units']:+.4f} units)\n"
        f"Flat ROI:  [bold]{_fmt_pct(s['flat_roi'])}[/]  (P/L {s['flat_pl_units']:+.2f}u)\n"
        f"Avg CLV: [bold]{_fmt_num(s['avg_clv_pct'])}%[/]  (on {s['clv_tracked']} bets)",
        title=f"Backtest {start} .. {end}", border_style="cyan", expand=False,
    ))
    # Show the first chunk of individual bets for inspection.
    head = result.bets.head(25)
    table = Table(title="Backtest bets (first 25)", header_style="bold magenta")
    for col in ("date", "matchup", "side", "american_odds", "ev_pct", "confidence", "result", "clv_pct"):
        table.add_column(col)
    for _, r in head.iterrows():
        color = "green" if r["result"] == "win" else "red"
        table.add_row(
            str(r["date"]), str(r["matchup"]), str(r["side"]), f"{int(r['american_odds']):+d}",
            f"{r['ev_pct']:+.1f}%", _fmt_num(r["confidence"], 0),
            f"[{color}]{r['result']}[/]", _fmt_num(r["clv_pct"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# clear-cache
# ---------------------------------------------------------------------------
@cli.command("clear-cache")
def clear_cache_cmd() -> None:
    """Delete cached pybaseball/Statcast parquet files."""
    from mlb_value_bot.data.cache import clear_cache

    n = clear_cache()
    console.print(f"[green]Removed {n} cached file(s).[/]")


@cli.command("data-status")
@click.option("--season", type=int, default=None, help="Season (default: current year).")
def data_status_cmd(season: int | None) -> None:
    """Show whether FanGraphs CSV exports are present and readable."""
    from mlb_value_bot.data.fangraphs_csv import ensure_dir, status

    season = season or date.today().year
    ensure_dir()  # make sure the drop folder exists
    st = status(season)

    def _row(label: str, info: dict) -> str:
        if info["loaded"]:
            return (f"[green]OK[/]   {label}: {info['file']}  "
                    f"({info['rows']} rows)  cols: {', '.join(info['key_cols'])}")
        found = f"found '{info['file']}' but unreadable/missing required columns" if info["file"] else "not found"
        return f"[yellow]--[/]   {label}: {found}"

    console.print(Panel(
        f"FanGraphs CSV fallback {'[green]enabled[/]' if st['enabled'] else '[red]disabled[/]'}\n"
        f"Drop folder: [bold]{st['dir']}[/]\n\n"
        f"{_row('pitching', st['pitching'])}\n"
        f"{_row('batting ', st['batting'])}",
        title=f"data-status — season {season}", border_style="cyan", expand=False,
    ))
    if not st["pitching"]["loaded"]:
        console.print(
            "[dim]How to populate (FanGraphs is Cloudflare-blocked to scrapers):\n"
            f"  1. In a browser open FanGraphs > Leaderboards > Pitching, set the season and Min IP = 0.\n"
            "     Make sure the table shows Name, Team, IP, G, GS, FIP and xFIP and/or SIERA (add K-BB%, Stuff+ if you like).\n"
            f"  2. Click 'Export Data' and save it as '{st['dir']}\\pitching_{season}.csv'.\n"
            f"  3. (Optional) Export Team Batting as '{st['dir']}\\batting_{season}.csv' for wRC+.\n"
            "  Then re-run. Without it, the starter component uses the Statcast-derived rate.[/]"
        )


@cli.command("pull")
def pull_cmd() -> None:
    """Rebuild the local tracking DB from Supabase (used by the hosted pipeline)."""
    from mlb_value_bot.sync.supabase_sync import (
        SupabaseConfigError,
        pull_recommendations,
        pull_totals_recommendations,
    )

    try:
        n = pull_recommendations()
    except SupabaseConfigError as exc:
        console.print(f"[bold red]Supabase not configured:[/] {exc}")
        raise SystemExit(1)
    except Exception as exc:
        console.print(f"[bold red]Pull failed:[/] {exc}")
        log.exception("supabase pull failed")
        raise SystemExit(1)
    console.print(f"[green]Pulled {n} recommendation(s) from Supabase into local DB.[/]")

    # Totals (paper) restore -- required so the ephemeral CI box can grade prior
    # totals picks + keep their CLV moving. Tolerant: skips if disabled or the
    # totals table isn't there yet, without failing the moneyline pull.
    if load_config().get("totals", {}).get("enabled", False):
        try:
            n_t = pull_totals_recommendations()
            console.print(f"[green]Pulled {n_t} totals (paper) row(s) from Supabase into local DB.[/]")
        except Exception as exc:  # noqa: BLE001
            log.warning("totals pull skipped (%s)", exc)
            console.print("[dim]Totals pull skipped (table not present or unreadable).[/]")


@cli.command("sync")
@click.option("--since", default=None, help="Only sync recommendations on/after this date (YYYY-MM-DD).")
def sync_cmd(since: str | None) -> None:
    """Push recommendations + performance to Supabase (for the public web site)."""
    from mlb_value_bot.sync.supabase_sync import SupabaseConfigError, push_all

    try:
        result = push_all(since=since)
    except SupabaseConfigError as exc:
        console.print(f"[bold red]Supabase not configured:[/] {exc}")
        raise SystemExit(1)
    except Exception as exc:
        console.print(f"[bold red]Sync failed:[/] {exc}")
        log.exception("supabase sync failed")
        raise SystemExit(1)

    console.print(Panel(
        f"Recommendations pushed: [bold]{result.recommendations}[/]\n"
        f"Totals (paper) pushed:  [bold]{result.totals_recommendations}[/]\n"
        f"Performance snapshot(s): [bold]{result.performance_scopes}[/]",
        title="Synced to Supabase", border_style="green", expand=False,
    ))


@cli.command()
@click.option("--port", default=8501, show_default=True, help="Port to serve on.")
@click.option("--headless", is_flag=True, help="Don't auto-open a browser.")
def serve(port: int, headless: bool) -> None:
    """Launch the Streamlit web viewer (Today / Results / Performance)."""
    import subprocess

    from mlb_value_bot.utils import PACKAGE_DIR

    app_path = PACKAGE_DIR / "web" / "app.py"
    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print("[red]Streamlit isn't installed.[/] Run: pip install streamlit")
        raise SystemExit(1)

    args = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    if headless:
        args += ["--server.headless", "true"]
    console.print(f"[cyan]Starting web viewer at http://localhost:{port}  (Ctrl+C to stop)[/]")
    subprocess.run(args)


if __name__ == "__main__":
    cli()
