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

    from mlb_value_bot.data import fetch_ledger
    fetch_ledger.reset()   # output-neutral fetch ledger for provenance (below)

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
        # Data-provenance: label WHY components are unavailable (additive only).
        try:
            from mlb_value_bot.diagnostics.provenance import annotate_slate
            from mlb_value_bot.griffbet import GRIFF_DB_PATH
            health = annotate_slate(evaluable, GRIFF_DB_PATH, "griff_recommendations", game_date, config)
            if health["fetch_failures"] or health["stale"]:
                click.echo(f"Data health: {health['fetch_failures']} fetch failure(s), "
                           f"{health['stale']} stale. {health['details']}")
        except Exception as exc:  # noqa: BLE001 -- diagnostics must never break a run
            log.warning("griff provenance annotation failed (%s)", exc)


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


@cli.command()
def train() -> None:
    """Retrain the residual market-error model on accumulated history, store it,
    and report the out-of-sample verdict + feature ablation."""
    from mlb_value_bot.griffbet.residual_model import train_and_save

    config = load_griff_config()
    report = train_and_save(config)
    click.echo(f"Trained residual model on {report['n_train']} graded games (l2={report['l2']}).")
    oos = report["oos"]
    if oos.get("sufficient"):
        verdict = "BEATS market" if oos.get("beats_market_log_loss") else "does NOT beat market"
        click.echo(f"  OOS (n_test={oos.get('n')}): model log loss {oos.get('model_log_loss')} vs "
                   f"market {oos.get('market_log_loss')} -> {verdict}")
    else:
        click.echo(f"  OOS: {oos.get('note', 'insufficient data')}")
    if report["ablation"]:
        click.echo("  Top features by out-of-sample value (positive = helps):")
        for a in report["ablation"][:5]:
            click.echo(f"    {a['feature']:18s} delta_log_loss {a['delta_vs_full']:+.4f}")


@cli.command()
@click.option("--limit", type=int, default=None, help="Cap games (for a quick test).")
def backfill(limit: int | None) -> None:
    """Backfill Stage-4 features for historical graded games (one-time)."""
    from mlb_value_bot.griffbet.backfill import backfill as run_backfill

    config = load_griff_config()
    n = run_backfill(config, limit=limit)
    click.echo(
        f"Backfilled {n['games']} graded game(s): "
        f"pitcher pitch-quality on {n['pitcher_ok']}, "
        f"confirmed lineups on {n['lineup_confirmed']}, weather on {n['weather_ok']}."
    )
    click.echo("Run `griff train` to fold the backfilled features into the model.")


@cli.command(name="fetch-history")
def fetch_history() -> None:
    """Download a free historical MLB odds dataset (pwu97/bettingtools, 2014-2019)
    and convert it to the ingester CSV at storage/hist_odds/mlb_2014_2019.csv.
    One-time; needs `pip install pyreadr` (R .rda reader, not a runtime dep)."""
    import os
    import urllib.request

    import pandas as pd
    try:
        import pyreadr
    except ImportError:
        raise SystemExit("Install the one-time converter dep: pip install pyreadr")

    out_dir = "storage/hist_odds"
    os.makedirs(out_dir, exist_ok=True)
    base = "https://raw.githubusercontent.com/pwu97/bettingtools/master/data/mlb_odds_{yr}.rda"
    frames = []
    for yr in range(2014, 2020):
        tmp = os.path.join(out_dir, f"_{yr}.rda")
        urllib.request.urlretrieve(base.format(yr=yr), tmp)
        frames.append(list(pyreadr.read_r(tmp).values())[0])
        os.remove(tmp)
        click.echo(f"  {yr}: {len(frames[-1])} games")
    raw = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "date": raw["date"].astype(str),
        "home_team": raw["home_name"], "away_team": raw["away_name"],
        "home_open": raw["home_open_ml"], "away_open": raw["away_open_ml"],
        "home_close": raw["home_close_ml"], "away_close": raw["away_close_ml"],
        "home_score": raw["home_score"], "away_score": raw["away_score"],
    }).dropna()
    for c in ("home_open", "away_open", "home_close", "away_close", "home_score", "away_score"):
        out[c] = out[c].astype(int)
    path = os.path.join(out_dir, "mlb_2014_2019.csv")
    out.to_csv(path, index=False)
    click.echo(f"Wrote {len(out)} games -> {path}. Now run `griff hist-train`.")


@cli.command(name="fetch-pinnacle")
def fetch_pinnacle() -> None:
    """Download a free SHARP dataset -- actual Pinnacle MLB 2016 lines
    (marcoblume/pinnacle.data) -- and convert open+close moneylines per game to
    the ingester CSV at storage/hist_odds/pinnacle_mlb2016.csv. The team1/team2
    -> home/away mapping is resolved by calibration. One-time; needs
    `pip install rdata` (R reader, not a runtime dep)."""
    import os
    import urllib.request

    import numpy as np
    import pandas as pd
    try:
        import rdata
    except ImportError:
        raise SystemExit("Install the one-time converter dep: pip install rdata")
    from mlb_value_bot.analysis.ev_calculator import devigged_market_probs

    out_dir = "storage/hist_odds"
    os.makedirs(out_dir, exist_ok=True)
    rda = os.path.join(out_dir, "MLB2016.rda")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/marcoblume/pinnacle.data/master/data/MLB2016.rda", rda)
    df = rdata.read_rda(rda)["MLB2016"]
    os.remove(rda)

    recs = []
    for _, g in df.iterrows():
        sh, sa = g["FinalScoreHome"], g["FinalScoreAway"]
        if pd.isna(sh) or pd.isna(sa) or int(sh) == int(sa):
            continue
        L = g["Lines"]
        if not isinstance(L, pd.DataFrame) or L.empty:
            continue
        pre = L[L["EnteredDateTimeUTC"] <= g["EventDateTimeUTC"]]
        pre = (pre if not pre.empty else L).dropna(subset=["MoneyUS1", "MoneyUS2"]).sort_values("EnteredDateTimeUTC")
        if pre.empty:
            continue
        o, c = pre.iloc[0], pre.iloc[-1]
        recs.append({"date": pd.to_datetime(g["EventDateTimeUTC"], unit="s").strftime("%Y-%m-%d"),
                     "home_team": g["HomeTeam"], "away_team": g["AwayTeam"],
                     "m1_open": int(o["MoneyUS1"]), "m2_open": int(o["MoneyUS2"]),
                     "m1_close": int(c["MoneyUS1"]), "m2_close": int(c["MoneyUS2"]),
                     "home_score": int(sh), "away_score": int(sa)})
    raw = pd.DataFrame(recs)
    raw["home_won"] = (raw["home_score"] > raw["away_score"]).astype(int)

    def brier(team2_home):
        e = []
        for _, r in raw.iterrows():
            h, a = (r["m2_close"], r["m1_close"]) if team2_home else (r["m1_close"], r["m2_close"])
            try:
                dh, _ = devigged_market_probs(h, a)
            except ValueError:
                continue
            e.append((dh - r["home_won"]) ** 2)
        return float(np.mean(e))

    team2_home = brier(True) < brier(False)

    def side(r, which, is_home):
        t1, t2 = r[f"m1_{which}"], r[f"m2_{which}"]
        return (t2 if is_home else t1) if team2_home else (t1 if is_home else t2)

    out = pd.DataFrame({
        "date": raw["date"], "home_team": raw["home_team"], "away_team": raw["away_team"],
        "home_open": raw.apply(lambda r: side(r, "open", True), axis=1),
        "away_open": raw.apply(lambda r: side(r, "open", False), axis=1),
        "home_close": raw.apply(lambda r: side(r, "close", True), axis=1),
        "away_close": raw.apply(lambda r: side(r, "close", False), axis=1),
        "home_score": raw["home_score"], "away_score": raw["away_score"]})
    path = os.path.join(out_dir, "pinnacle_mlb2016.csv")
    out.to_csv(path, index=False)
    click.echo(f"Wrote {len(out)} Pinnacle games (home={'team2' if team2_home else 'team1'}) "
               f"-> {path}. Run `griff hist-train --file {path}`.")


@cli.command(name="hist-train")
@click.option("--file", "path", default="storage/hist_odds/mlb_2014_2019.csv",
              help="Historical odds CSV/XLSX (see griffbet/hist_odds.py schema).")
@click.option("--l2", default=20.0, help="L2 shrinkage.")
def hist_train(path: str, l2: float) -> None:
    """Train + out-of-sample-test the market-microstructure model on a historical
    odds dataset: does line movement beat the closing line?"""
    from mlb_value_bot.griffbet.hist_odds import build_market_rows, load_history
    from mlb_value_bot.griffbet.market_model import save_model, train_from_history

    config = load_griff_config()
    df = build_market_rows(load_history(path), config["ev"]["devig_method"])
    click.echo(f"Ingested {len(df)} usable historical games.")
    if df.empty:
        return
    model, oos = train_from_history(df, l2)
    save_model(model)
    if oos.get("sufficient"):
        verdict = "BEATS the close" if oos.get("beats_close_log_loss") else "does NOT beat the close"
        click.echo(f"OOS (n_test={oos['n']}): model log loss {oos.get('model_log_loss')} vs "
                   f"close {oos.get('close_log_loss')} -> {verdict}")
        click.echo(f"Coefficients (standardized): {oos.get('coefficients')}")
    else:
        click.echo(f"OOS: {oos.get('note')}")


@cli.command(name="pull")
def pull_cmd() -> None:
    """Rebuild GriffBet's local DB from Supabase."""
    from mlb_value_bot.griffbet.sync_griff import pull_features, pull_recommendations

    n = pull_recommendations()
    nf = pull_features()
    click.echo(f"Pulled {n} GriffBet recommendation(s) and {nf} feature row(s) from Supabase.")


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
