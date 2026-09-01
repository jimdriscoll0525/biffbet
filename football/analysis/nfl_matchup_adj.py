"""NFL conditional matchup adjustments — PURE (M3 of the ATS module).

The core rating (M2) already contains each team's overall EPA+SR quality, so
an adjustment fires ONLY on a meaningful mismatch — top-`edge_n` unit vs
bottom-`edge_n` unit in the 32-team pool (spec: top-10 vs bottom-10) — and
neutral matchups contribute exactly zero, to avoid double-counting. Every
fired adjustment is a fixed, config-sized number of points with a note, and
the SUM is clamped by projections.margin_tilt_max_nfl downstream.

Channels (config nfl_matchup.*):
  rush   — rush EPA+SR composite vs opponent rush defense composite
  pass   — pass EPA vs opponent pass defense EPA
  qb_pressure — a pressure-sensitive QB (bottom-quartile EPA drop on qb-hit
           dropbacks) facing a top-`edge_n` pressure defense
  ol_pass / ol_run — sack-rate and stuff-rate mismatches

Ranks use the caller-supplied league table (all 32 teams' weighted stats),
so tests drive them with fixtures.
"""
from __future__ import annotations

import pandas as pd


def _rank_ok(series: pd.Series, team: str, top: bool, edge_n: int,
             higher_is_better: bool) -> bool:
    s = series.dropna()
    if team not in s.index or len(s) < edge_n * 2:
        return False
    ranked = s.rank(ascending=not higher_is_better)   # 1 = best
    r = ranked[team]
    return r <= edge_n if top else r > len(s) - edge_n


def qb_pressure_sensitivity(qb_ws: pd.DataFrame, primary: dict[str, str],
                            min_hit_dropbacks: int = 20) -> pd.Series:
    """team -> EPA/dropback drop when hit (negative = degrades under
    pressure), for the team's primary passer. NaN under the sample floor."""
    if qb_ws is None or qb_ws.empty or not primary:
        return pd.Series(dtype=float)
    g = qb_ws.groupby(["passer", "team"]).sum(numeric_only=True)
    out = {}
    for team, passer in primary.items():
        if (passer, team) not in g.index:
            continue
        row = g.loc[(passer, team)]
        clean_db = row["dropbacks"] - row["hit_dropbacks"]
        if row["hit_dropbacks"] < min_hit_dropbacks or clean_db <= 0:
            continue
        clean_epa = (row["epa_sum"] - row["hit_epa_sum"]) / clean_db
        hit_epa = row["hit_epa_sum"] / row["hit_dropbacks"]
        out[team] = hit_epa - clean_epa
    return pd.Series(out, dtype=float)


def matchup_adjustments(home: str, away: str, table: pd.DataFrame,
                        qb_sens: pd.Series, config: dict) -> tuple[float, list[str]]:
    """Signed home-positive adjustment points + human-readable notes."""
    cfg = config.get("nfl_matchup", {})
    edge_n = int(cfg.get("edge_n", 10))
    pts = {"rush": float(cfg.get("rush_adj_pts", 1.0)),
           "pass": float(cfg.get("pass_adj_pts", 1.5)),
           "qb_pressure": float(cfg.get("qb_pressure_adj_pts", 1.0)),
           "ol_pass": float(cfg.get("ol_pass_adj_pts", 1.0)),
           "ol_run": float(cfg.get("ol_run_adj_pts", 0.5))}
    sens_q = float(cfg.get("qb_pressure_sens_quantile", 0.25))
    if table is None or table.empty or home not in table.index or away not in table.index:
        return 0.0, []

    t = table
    t = t.assign(
        off_rush_comp=t["off_rush_epa"] + 1.6 * t["off_rush_sr"],
        def_rush_comp=t["def_rush_epa"] + 1.6 * t["def_rush_sr"],
    )
    total, notes = 0.0, []

    def channel(off_team, def_team, off_col, def_col, name, p, off_high_good=True):
        nonlocal total
        # strong offense vs weak defense (defense col: HIGHER allowed = worse)
        if _rank_ok(t[off_col], off_team, True, edge_n, off_high_good)                 and _rank_ok(t[def_col], def_team, True, edge_n, True):
            sign = 1.0 if off_team == home else -1.0
            total += sign * p
            notes.append(f"{name}: {off_team} top-{edge_n} O vs {def_team} bottom-{edge_n} D ({sign * p:+.1f})")
        elif _rank_ok(t[off_col], off_team, False, edge_n, off_high_good)                 and _rank_ok(t[def_col], def_team, False, edge_n, True):
            sign = -1.0 if off_team == home else 1.0
            total += sign * p
            notes.append(f"{name}: {off_team} bottom-{edge_n} O vs {def_team} top-{edge_n} D ({sign * p:+.1f})")

    for off_team, def_team in ((home, away), (away, home)):
        channel(off_team, def_team, "off_rush_comp", "def_rush_comp", "rush", pts["rush"])
        channel(off_team, def_team, "off_pass_epa", "def_pass_epa", "pass", pts["pass"])
        # OL/DL: rates where LOWER is better for the offense side.
        channel(off_team, def_team, "sack_rate_taken", "sack_rate_made", "OL pass-pro",
                pts["ol_pass"], off_high_good=False)
        channel(off_team, def_team, "stuff_rate_taken", "stuff_rate_made", "run blocking",
                pts["ol_run"], off_high_good=False)

    # QB under pressure: fires against the pressure-sensitive QB's team.
    if qb_sens is not None and len(qb_sens.dropna()) >= 8:
        cutoff = qb_sens.quantile(sens_q)
        for team, opp in ((home, away), (away, home)):
            sens = qb_sens.get(team)
            if sens is not None and sens == sens and sens <= cutoff                     and _rank_ok(t["pressure_rate_made"], opp, True, edge_n, True):
                sign = -1.0 if team == home else 1.0
                total += sign * pts["qb_pressure"]
                notes.append(f"QB pressure: {team} QB degrades under pressure "
                             f"({sens:+.2f} EPA/db) vs top-{edge_n} pressure D ({sign * pts['qb_pressure']:+.1f})")
    return round(total, 2), notes
