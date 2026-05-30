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
from datetime import date, timedelta

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

from mlb_value_bot.pipeline import GameAnalysis, analyze_slate, save_slate, save_value_bets  # save_value_bets kept for back-compat imports
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
# results
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--date", "date_", default=None, help="Game date YYYY-MM-DD (default: yesterday).")
def results(date_: str | None) -> None:
    """Fetch final scores and settle open bets for a date."""
    game_date = date_ or (date.today() - timedelta(days=1)).isoformat()
    summary = results_mod.grade_date(game_date)

    table = Table(title=f"Results — {game_date}", header_style="bold cyan")
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
        title=f"Daily P/L — {game_date}", border_style="cyan", expand=False,
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
    from mlb_value_bot.sync.supabase_sync import SupabaseConfigError, pull_recommendations

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
