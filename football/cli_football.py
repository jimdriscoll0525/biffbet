"""Football CLI — `python -m mlb_value_bot.football <command>`.

Mirrors the daily-command shape of BiffBet/GriffBet on football's independent
config + store:
  data-status | matchups | today | results | pull | sync

Commands land milestone by milestone; each defers heavy imports so `--help`
stays fast and tests never pay for nflreadpy.
"""
from __future__ import annotations

from datetime import date

import click

from mlb_value_bot.football import football_in_season, load_football_config, season_for_date
from mlb_value_bot.utils import get_logger

log = get_logger("football.cli")


@click.group()
def cli() -> None:
    """Football - the NFL + college FBS matchup-exploitation model (paper)."""


@cli.command(name="data-status")
@click.option("--season", type=int, default=None, help="Season year (default: inferred from today).")
@click.option("--odds/--no-odds", default=False,
              help="Also probe The Odds API for both leagues (BURNS ~8 credits).")
def data_status(season: int | None, odds: bool) -> None:
    """Report row counts / availability for every football data feed."""
    config = load_football_config()
    yr = season or season_for_date(date.today().isoformat())
    click.echo(f"Football data status (season {yr}):")

    # -- nflverse (free) ------------------------------------------------------
    from mlb_value_bot.football.data import nfl_client

    for label, fetch in (
        ("NFL schedules", lambda: nfl_client.schedules(yr, config)),
        ("NFL play-by-play", lambda: nfl_client.pbp(yr, config)),
        ("NFL snap counts", lambda: nfl_client.snap_counts(yr, config)),
        ("NFL FTN charting", lambda: nfl_client.ftn_charting(yr, config)),
    ):
        try:
            df = fetch()
            note = "" if not df.empty else "  (empty — pre-season or feed gap)"
            click.echo(f"  {label:22s} {len(df):>7d} rows{note}")
        except Exception as exc:  # noqa: BLE001 -- status must report, not crash
            click.echo(f"  {label:22s} ERROR: {exc}")

    # -- CFBD (key + monthly quota) --------------------------------------------
    from mlb_value_bot.football.data.cfbd_client import CfbdClient, CfbdError

    cfbd = CfbdClient(config)
    if not cfbd.configured:
        click.echo("  CFBD                   NO KEY -- set CFBD_API_KEY in .env "
                   "(free key at collegefootballdata.com)")
    else:
        for label, fetch in (
            ("CFBD FBS teams", lambda: cfbd.fbs_teams(yr)),
            ("CFBD season stats", lambda: cfbd.season_stats(yr)),
            ("CFBD team PPA", lambda: cfbd.ppa_teams(yr)),
            ("CFBD SP+ ratings", lambda: cfbd.sp_ratings(yr)),
            ("CFBD venues", lambda: cfbd.venues()),
        ):
            try:
                df = fetch()
                note = "" if not df.empty else "  (empty — pre-season or feed gap)"
                click.echo(f"  {label:22s} {len(df):>7d} rows{note}")
            except CfbdError as exc:
                click.echo(f"  {label:22s} ERROR: {exc}")

    # -- Weather (free, no key): probe one outdoor stadium ----------------------
    from mlb_value_bot.football.data.football_weather import game_weather
    from mlb_value_bot.football.data.stadiums import nfl_coords

    lat, lon = nfl_coords("GB")
    env = game_weather(lat, lon, f"{date.today().isoformat()}T18:00:00Z", indoor=False, config=config)
    click.echo(f"  Weather (GB probe)     {'OK' if env.available else 'UNAVAILABLE'}: {env.note}")

    # -- Odds (opt-in: costs credits) -------------------------------------------
    if odds:
        from mlb_value_bot.football.data.football_odds import fetch_league_odds

        for league in ("nfl", "cfb"):
            try:
                games = fetch_league_odds(league, config)
                click.echo(f"  Odds ({league.upper():3s})             {len(games):>7d} events with lines")
            except Exception as exc:  # noqa: BLE001
                click.echo(f"  Odds ({league.upper():3s})             ERROR: {exc}")
    else:
        click.echo("  Odds                   skipped (use --odds to probe; costs ~8 credits)")


@cli.command()
@click.option("--date", "date_", default=None, help="Slate date YYYY-MM-DD (default: today).")
@click.option("--league", "league_", type=click.Choice(["nfl", "cfb", "all"]), default="all")
@click.option("--save/--no-save", default=True, help="Persist the slate to football's DB.")
def today(date_: str | None, league_: str, save: bool) -> None:
    """Analyze the current football board and print the paper slate."""
    from mlb_value_bot.football.pipeline_football import evaluate_league_slate

    config = load_football_config()
    game_date = date_ or date.today().isoformat()
    if not football_in_season(game_date, config):
        click.echo(f"Football off-season ({game_date}): Odds API pull skipped "
                   "(see season_window in config_football.yaml).")
        return
    leagues = ["nfl", "cfb"] if league_ == "all" else [league_]
    paper = config.get("betting", {}).get("paper_only", True)

    all_analyses = []
    for lg in leagues:
        if not config.get("leagues", {}).get(lg, {}).get("enabled", True):
            continue
        try:
            analyses = evaluate_league_slate(lg, game_date, config)
        except Exception as exc:  # noqa: BLE001 -- one league failing must not kill the other
            log.warning("%s slate failed: %s", lg, exc)
            click.echo(f"{lg.upper()}: slate unavailable ({exc})")
            continue
        all_analyses.extend(analyses)

        bets = [(a, p) for a in analyses for p in a.bets]
        n_markets = sum(len(a.picks) for a in analyses)
        tag = " [PAPER]" if paper else ""
        click.echo(f"{lg.upper()} board{tag}: {len(bets)} pick(s) from "
                   f"{n_markets} priced market(s) across {len(analyses)} game(s)")
        for a, p in sorted(bets, key=lambda t: -t[1].adjusted_ev):
            line_txt = (f"{p.side} {p.line:+g}" if p.market == "spread"
                        else f"{p.side} {p.line:g}")
            click.echo(
                f"  {a.away} @ {a.home}: {p.market.upper()} {line_txt} {p.american_odds:+d} "
                f"EV {p.raw_ev * 100:+.1f}% adjEV {p.adjusted_ev * 100:+.1f}% "
                f"conf {p.confidence:.0f} tier={p.tier} stake {p.stake_pct * 100:.1f}u "
                f"[{p.stability_label}]")
            arch = p.reasoning["matchup"]["archetype"]
            if arch != "neutral":
                click.echo(f"      {arch}: {'; '.join(p.reasoning['matchup']['notes'][:2])}")

    if save and all_analyses:
        from mlb_value_bot.football.tracking.football_store import save_slate

        total, n_value = save_slate(all_analyses, config)
        click.echo(f"Saved {total} football market row(s) ({n_value} paper bets).")


@cli.command()
@click.option("--before", default=None, help="Grade bets dated before this (default: today).")
def results(before: str | None) -> None:
    """Grade open football bets (self-healing sweep of all past dates)."""
    from mlb_value_bot.football.tracking.football_results import grade_open

    config = load_football_config()
    summaries = grade_open(config, before=before)
    if not summaries:
        click.echo("No past dates with open football bets.")
        return
    for s in summaries:
        click.echo(f"{s.league.upper()}: {s.graded} settled "
                   f"({s.wins}W-{s.losses}L-{s.pushes}P), {s.voids} void, "
                   f"{s.pending} pending, P/L {s.profit_loss:+.4f}u "
                   f"across {len(s.dates)} date(s)")


@cli.command(name="pull")
def pull_cmd() -> None:
    """Rebuild football's local DB from Supabase."""
    from mlb_value_bot.football.sync_football import pull_recommendations

    n = pull_recommendations()
    click.echo(f"Pulled {n} football recommendation row(s) from Supabase.")


@cli.command(name="sync")
@click.option("--since", default=None, help="Only sync rows on/after this date.")
def sync_cmd(since: str | None) -> None:
    """Push football rows + the snapshot (records/distribution/CLV/calibration)."""
    from mlb_value_bot.football.sync_football import push_all

    config = load_football_config()
    out = push_all(config, since=since)
    click.echo(f"Synced football: {out}")


@cli.command()
@click.option("--league", type=click.Choice(["nfl", "cfb"]), default="nfl")
@click.option("--season", type=int, default=None, help="Season year (default: inferred).")
@click.option("--week", type=int, required=True, help="Week whose games to score.")
@click.option("--top", type=int, default=15, help="Show the N biggest edges.")
def matchups(league: str, season: int | None, week: int, top: int) -> None:
    """Debug board: matchup edge scores for one week's games (no odds, no
    picks — pure matchup model output for hand-checking)."""
    config = load_football_config()
    yr = season or season_for_date(date.today().isoformat())

    from mlb_value_bot.football.pipeline_football import build_league_context, score_game

    ctx = build_league_context(league, yr, week, config)
    if ctx.percentiles.empty:
        click.echo(f"No {league.upper()} unit stats available for season {yr}.")
        return

    rows = []
    for _, g in ctx.week_games(week).iterrows():
        home, away = g["home_team"], g["away_team"]
        scored = score_game(ctx, home, away)
        if scored is None:
            continue
        rows.append((scored.matchup.home_edge, home, away, scored))
    rows.sort(key=lambda r: -abs(r[0]))

    click.echo(f"{league.upper()} week {week} matchup edges (home-positive, pct points):")
    for edge, home, away, scored in rows[:top]:
        m = scored.matchup
        tags = []
        if m.dual_edge_side:
            tags.append(f"DUAL:{m.dual_edge_side}")
        if m.turnover_flag_side:
            tags.append(f"TO:{m.turnover_flag_side}")
        ol_h, ol_a = scored.ol_home.points, scored.ol_away.points
        click.echo(f"  {away:>18s} @ {home:<18s} edge {edge:+7.1f}  "
                   f"OL {ol_h:+.1f}/{ol_a:+.1f}  {m.archetype}"
                   + (f"  [{' '.join(tags)}]" if tags else ""))
        for note in m.notes[:3]:
            click.echo(f"      - {note}")


if __name__ == "__main__":
    cli()
