"""Per-(team, week) NFL aggregates from trimmed nflfastR pbp — PURE (M1 of
the ATS module, approved 2026-09-01).

Design: aggregates are stored SPLIT by garbage time (`_ngt` = win prob in
[0.05, 0.95], `_gt` = outside it) as raw sums+counts, so the rating layer
can re-weight garbage time with any `gt_weight` without re-aggregating pbp.
Same idea for pass/rush phase splits (the M3 matchup layer's inputs) and the
luck-lens counting stats (fumbles, red zone, turnovers). One row per
(team, week); offense (`off_`) from posteam rows, defense (`def_`) from
defteam rows of the same plays.
"""
from __future__ import annotations

import pandas as pd

GT_LOW, GT_HIGH = 0.05, 0.95


def _side_agg(df: pd.DataFrame, team_col: str, prefix: str) -> pd.DataFrame:
    """Sums+counts per (team, week) for one side of the ball."""
    d = df[df[team_col].notna() & (df[team_col] != "")].copy()
    d["team"] = d[team_col]
    is_play = (d["pass"] == 1) | (d["rush"] == 1)
    d = d[is_play]
    gt = d["wp"].notna() & ((d["wp"] < GT_LOW) | (d["wp"] > GT_HIGH))
    is_pass, is_rush, is_sack = d["pass"] == 1, d["rush"] == 1, d["sack"] == 1

    g = d.groupby(["team", "week"])
    out = pd.DataFrame(index=g.size().index)

    def add(name, mask, col=None):
        sel = d[mask]
        grp = sel.groupby([sel["team"], sel["week"]])
        series = grp.size() if col is None else grp[col].sum()
        out[name] = series.reindex(out.index).fillna(0.0)

    for phase, pmask in (("all", is_play.loc[d.index]), ("pass", is_pass), ("rush", is_rush)):
        for bucket, bmask in (("ngt", ~gt), ("gt", gt)):
            m = pmask & bmask
            add(f"{prefix}_{phase}_{bucket}_plays", m)
            add(f"{prefix}_{phase}_{bucket}_epa", m, "epa")
            add(f"{prefix}_{phase}_{bucket}_succ", m, "success")

    # Proxies + luck-lens counting stats (not gt-split; small samples).
    add(f"{prefix}_dropbacks", d["qb_dropback"] == 1)
    add(f"{prefix}_sacks", is_sack)
    add(f"{prefix}_qb_hits", d["qb_hit"] == 1)
    add(f"{prefix}_stuffs", is_rush & (d["yards_gained"] <= 0))
    add(f"{prefix}_rush_att", d["rush_attempt"] == 1)
    add(f"{prefix}_int", d["interception"] == 1)
    add(f"{prefix}_fumbles", d["fumble"] == 1)
    add(f"{prefix}_fumbles_lost", d["fumble_lost"] == 1)
    add(f"{prefix}_rz_plays", d["yardline_100"] <= 20)
    add(f"{prefix}_rz_td", (d["yardline_100"] <= 20) & (d["touchdown"] == 1))
    add(f"{prefix}_games", d["game_id"].notna())  # placeholder; fixed below
    out[f"{prefix}_games"] = g["game_id"].nunique().reindex(out.index)
    return out


def week_team_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, week): off_* from posteam plays, def_* from defteam
    plays. Empty frame in -> empty frame out (degrade contract)."""
    need = {"posteam", "defteam", "week", "pass", "rush", "epa", "success",
            "wp", "sack", "qb_dropback", "qb_hit", "yards_gained",
            "rush_attempt", "interception", "fumble", "fumble_lost",
            "yardline_100", "touchdown", "game_id"}
    if pbp.empty or not need <= set(pbp.columns):
        return pd.DataFrame()
    off = _side_agg(pbp, "posteam", "off")
    de = _side_agg(pbp, "defteam", "def")
    out = off.join(de, how="outer").fillna(0.0).reset_index()
    out = out.rename(columns={"level_0": "team", "level_1": "week"})
    if "team" not in out.columns:   # pandas names the levels when set
        out = out.rename(columns={out.columns[0]: "team", out.columns[1]: "week"})
    return out


def week_qb_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (passer, team, week): dropback EPA overall and on qb-hit plays —
    the M3 pressure-sensitivity input."""
    need = {"posteam", "week", "qb_dropback", "qb_hit", "epa", "passer_player_name"}
    if pbp.empty or not need <= set(pbp.columns):
        return pd.DataFrame()
    d = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_name"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    hit = d["qb_hit"] == 1
    g = d.groupby(["passer_player_name", "posteam", "week"])
    out = pd.DataFrame({
        "dropbacks": g.size(),
        "epa_sum": g["epa"].sum(),
    })
    gh = d[hit].groupby([d.loc[hit, "passer_player_name"],
                         d.loc[hit, "posteam"], d.loc[hit, "week"]])
    out["hit_dropbacks"] = gh.size().reindex(out.index).fillna(0.0)
    out["hit_epa_sum"] = gh["epa"].sum().reindex(out.index).fillna(0.0)
    out.index.names = ["passer", "team", "week"]
    return out.reset_index()
