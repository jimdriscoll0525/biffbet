"""GriffBet CLI — `python -m mlb_value_bot.griffbet <command>`.

Mirrors BiffBet's daily commands on GriffBet's independent config + store:
  today | results | pull | sync | referee | investigate-base
"""
from __future__ import annotations

import json
from datetime import date

import click

from mlb_value_bot.griffbet import load_griff_config
from mlb_value_bot.utils import get_logger

log = get_logger("griffbet.cli")


@click.group()
def cli() -> None:
    """GriffBet - the challenger model (independent config + store)."""


@cli.command()
@click.option("--date", "date_", default=None, help="Game date YYYY-MM-DD (default: today).")
@click.option("--save/--no-save", default=True, help="Persist the slate to GriffBet's DB.")
def today(date_: str | None, save: bool) -> None:
    """Analyze today's slate with GriffBet and (optionally) save it."""
    from mlb_value_bot.griffbet.pipeline import (
        analyze_slate_griff,
        refresh_skipped_closing_lines_griff,
        save_slate_griff,
    )

    config = load_griff_config()
    game_date = date_ or date.today().isoformat()
    threshold = float(config["ev"]["threshold"])
    analyses = analyze_slate_griff(game_date, config=config)
    evaluable = [a for a in analyses if a.best_eval is not None]
    value = [a for a in evaluable if a.is_value(threshold)]

    click.echo(f"GriffBet slate {game_date}: {len(value)} +EV of {len(evaluable)} evaluable")
    for a in value:
        be = a.best_eval
        disc = f" [{'; '.join(a.discipline_reasons)}]" if a.discipline_reasons else ""
        click.echo(
            f"  {a.away_team} @ {a.home_team}: {a.best_side} {be.american_odds:+d} "
            f"EV {be.ev_pct * 100:+.1f}% adjEV {(a.adjusted_ev_pct or 0) * 100:+.1f}% "
            f"tier={a.tier} kelly {be.kelly_stake * 100:.2f}% conf {a.confidence:.0f}"
            f" | raw pick {a.raw_pick_side} {a.raw_pick_price:+d}{disc}"
        )

    if save and evaluable:
        total, n_value = save_slate_griff(evaluable, threshold, game_date)
        n_ref = refresh_skipped_closing_lines_griff(analyses, game_date)
        click.echo(f"Saved {total} GriffBet row(s) ({n_value} bets); "
                   f"refreshed {n_ref} skipped-game closing line(s).")


@cli.command()
@click.option("--date", "date_", default=None, help="Grade one date (default: sweep all past open).")
def results(date_: str | None) -> None:
    """Grade GriffBet's open bets (self-healing sweep by default)."""
    from mlb_value_bot.griffbet import results as gresults

    if date_:
        summaries = [gresults.grade_date(date_)]
    else:
        summaries = gresults.grade_all_open(before=date.today().isoformat())
    if not summaries:
        click.echo("No past dates with open GriffBet bets.")
        return
    for s in summaries:
        click.echo(f"{s.date}: {s.graded} settled ({s.wins}W-{s.losses}L), "
                   f"{s.voids} void, {s.pending} pending, P/L {s.profit_loss:+.4f}")


@cli.command(name="pull")
def pull_cmd() -> None:
    """Rebuild GriffBet's local DB from Supabase."""
    from mlb_value_bot.griffbet.sync_griff import pull_recommendations

    n = pull_recommendations()
    click.echo(f"Pulled {n} GriffBet recommendation(s) from Supabase.")


@cli.command(name="sync")
@click.option("--since", default=None, help="Only sync rows on/after this date.")
def sync_cmd(since: str | None) -> None:
    """Compute the referee snapshot and push GriffBet data + referee to Supabase."""
    from mlb_value_bot.griffbet.sync_griff import push_all

    referee = _build_referee()
    out = push_all(referee=referee, since=since)
    click.echo(f"Synced GriffBet: {out}")


@cli.command()
@click.option("--sync/--no-sync", default=False, help="Also push the referee snapshot to Supabase.")
def referee(sync: bool) -> None:
    """Compute (and optionally push) the cross-model referee report."""
    snap = _build_referee()
    click.echo(json.dumps(snap, indent=2, default=str))
    if sync:
        from mlb_value_bot.griffbet.sync_griff import _credentials, push_referee  # type: ignore
        url, key = _credentials()
        push_referee(url, key, snap)
        click.echo("Referee snapshot synced.")


@cli.command(name="investigate-base")
@click.option("--date", "date_", default=None, help="Season inferred from this date (default: today).")
@click.option("--teams", default="New York Yankees,Los Angeles Dodgers,Colorado Rockies,Miami Marlins",
              help="Comma-separated sample teams.")
def investigate_base(date_: str | None, teams: str) -> None:
    """Report base_wp / win-prob movement OFF vs ON (neutralization) on an
    ace-start vs back-end-start day for a sample of teams (plan §B)."""
    from mlb_value_bot.analysis.pitcher_metrics import PitcherProfile
    from mlb_value_bot.analysis.team_metrics import TeamMetricsProvider
    from mlb_value_bot.griffbet.win_probability import base_off_on

    config = load_griff_config()
    game_date = date_ or date.today().isoformat()
    season = int(game_date[:4])
    provider = TeamMetricsProvider(season=season, config=config)
    opp = provider.build_team_profile("Athletics", is_home=False)  # ~league-average opponent stand-in
    ace = PitcherProfile(player_id=1, name="ACE", xfip=3.00)
    backend = PitcherProfile(player_id=2, name="BACKEND", xfip=5.00)
    avg_opp_sp = PitcherProfile(player_id=3, name="AVG", xfip=4.00)

    click.echo(f"Neutralization investigation (season {season}); home team starts ACE vs BACK-END:")
    for team in [t.strip() for t in teams.split(",") if t.strip()]:
        tp = provider.build_team_profile(team, is_home=True)
        ace_day = base_off_on(tp, opp, ace, avg_opp_sp, season, config)
        be_day = base_off_on(tp, opp, backend, avg_opp_sp, season, config)
        neut = ace_day["neutralization"]
        deg = " (DEGRADED: rotation rate unavailable)" if neut and neut.get("degraded") else ""
        click.echo(
            f"  {team}: base shift {neut['base_shift']:+.4f}{deg} | "
            f"ace-day home prob {ace_day['home_prob_off']:.3f}->{ace_day['home_prob_on']:.3f} "
            f"({ace_day['prob_shift']:+.3f}); back-end-day "
            f"{be_day['home_prob_off']:.3f}->{be_day['home_prob_on']:.3f} ({be_day['prob_shift']:+.3f})"
        )


def _build_referee() -> dict:
    from mlb_value_bot.tracking import recommendations as biff_recs
    from mlb_value_bot.griffbet import tracking as gtrack
    from mlb_value_bot.griffbet.referee import compute_referee

    config = load_griff_config()
    biff_df = biff_recs.to_dataframe()
    griff_df = gtrack.to_dataframe()
    return compute_referee(biff_df, griff_df, config)


if __name__ == "__main__":
    cli()
