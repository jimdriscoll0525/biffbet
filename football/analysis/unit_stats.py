"""Team unit stats — PURE aggregation from raw feed frames (no I/O).

NFL: everything derives from trimmed nflverse play-by-play (the EPA source of
truth). Defense-allowed stats are the same masks grouped by `defteam`.

CFB: CFBD's long-form season stats + team PPA. CFBD's basic season stats don't
carry opponent-allowed volume, so CFB defensive quality leans on PPA-allowed
(EPA-like) — the percentile layer averages whatever columns exist, so the two
leagues can carry different stat sets without special-casing downstream.

Column vocabulary (shared with percentiles.UNIT_SPECS): offense higher-better
stats plus *_allowed / *-created mirrors. A missing column simply drops that
stat from the unit's percentile mean.
"""
from __future__ import annotations

import pandas as pd

from mlb_value_bot.utils import get_logger

log = get_logger("football.analysis.unit_stats")


def nfl_unit_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-team season unit stats from trimmed pbp. Index = team abbr."""
    if pbp.empty:
        return pd.DataFrame()
    df = pbp.copy()
    for col in ("pass", "rush", "qb_dropback", "pass_attempt", "rush_attempt", "sack",
                "qb_hit", "interception", "fumble_lost", "pass_touchdown", "rush_touchdown",
                "touchdown", "yards_gained", "epa", "qtr", "yardline_100"):
        if col not in df.columns:
            df[col] = pd.NA
    df = df[df["posteam"].notna() & (df["posteam"] != "")]

    is_pass = df["pass"] == 1
    is_rush = df["rush"] == 1
    is_dropback = df["qb_dropback"] == 1
    is_sack = df["sack"] == 1
    in_rz = df["yardline_100"] <= 20

    def _agg(group_col: str) -> pd.DataFrame:
        g = df.groupby(group_col)
        out = pd.DataFrame(index=g.size().index)
        out["games"] = g["game_id"].nunique()

        def rate(mask_num, mask_den):
            num = df[mask_num].groupby(df[group_col]).size().reindex(out.index, fill_value=0)
            den = df[mask_den].groupby(df[group_col]).size().reindex(out.index, fill_value=0)
            return num / den.astype(float).replace(0.0, float("nan"))

        def total(col, mask):
            return df[mask].groupby(df[group_col])[col].sum().reindex(out.index, fill_value=0.0)

        def mean(col, mask):
            return df[mask].groupby(df[group_col])[col].mean().reindex(out.index)

        pass_yards = total("yards_gained", is_pass & ~is_sack)
        rush_yards = total("yards_gained", is_rush)
        out["pass_ypg"] = pass_yards / out["games"]
        out["pass_td_pg"] = total("pass_touchdown", is_pass) / out["games"]
        out["ypa"] = pass_yards / total("pass_attempt", is_pass).astype(float).replace(0.0, float("nan"))
        out["sack_rate"] = rate(is_sack, is_dropback)
        out["qb_hit_rate"] = rate(df["qb_hit"] == 1, is_dropback)
        # The pass-protection PROXY: sacks + QB hits per dropback (real pressure
        # data isn't free in-season; labeled a proxy everywhere it surfaces).
        out["pressure_proxy_rate"] = rate(is_sack | (df["qb_hit"] == 1), is_dropback)
        out["int_rate"] = rate(df["interception"] == 1, is_dropback)
        out["epa_dropback"] = mean("epa", is_dropback)
        out["rush_ypg"] = rush_yards / out["games"]
        out["ypc"] = rush_yards / total("rush_attempt", is_rush).astype(float).replace(0.0, float("nan"))
        out["rush_epa"] = mean("epa", is_rush)
        out["turnovers_pg"] = (total("interception", is_dropback)
                               + total("fumble_lost", is_pass | is_rush)) / out["games"]
        out["plays_pg"] = (df[is_pass | is_rush].groupby(df[group_col]).size()
                           .reindex(out.index, fill_value=0) / out["games"])
        out["rz_td_rate"] = rate((df["touchdown"] == 1) & in_rz, in_rz & (is_pass | is_rush))
        out["q4_epa"] = mean("epa", (df["qtr"] == 4) & (is_pass | is_rush))
        return out

    off = _agg("posteam")
    def_ = _agg("defteam")

    stats = off.rename(columns={
        "sack_rate": "sack_rate_allowed",
        "qb_hit_rate": "qb_hit_rate_allowed",
        "pressure_proxy_rate": "pressure_proxy_rate",   # offense's protection
        "turnovers_pg": "giveaway_pg",
    })
    for src, dst in (
        ("pass_ypg", "pass_ypg_allowed"), ("ypa", "ypa_allowed"),
        ("sack_rate", "sack_rate_made"), ("int_rate", "int_created_rate"),
        ("epa_dropback", "epa_dropback_allowed"), ("rush_ypg", "rush_ypg_allowed"),
        ("ypc", "ypc_allowed"), ("rush_epa", "rush_epa_allowed"),
        ("turnovers_pg", "takeaway_pg"),
    ):
        stats[dst] = def_[src]
    stats.index.name = "team"
    return stats


# Candidate CFBD statNames per derived stat (v2 field names vary by endpoint
# era; the first present candidate wins, absent -> stat skipped).
_CFBD_CANDIDATES = {
    "games": ["games"],
    "net_pass_yards": ["netPassingYards", "netPassingYds"],
    "pass_attempts": ["passAttempts"],
    "pass_tds": ["passingTDs"],
    "rush_yards": ["rushingYards"],
    "rush_attempts": ["rushingAttempts"],
    "turnovers": ["turnovers"],
    "interceptions_made": ["interceptions"],          # CFBD: defensive INTs
    "fumbles_recovered": ["fumblesRecovered"],
    "sacks_made": ["sacks"],                          # CFBD: defensive sacks
    "penalty_yards": ["penaltyYards"],
}


def cfb_unit_stats(stats_long: pd.DataFrame, ppa: pd.DataFrame) -> pd.DataFrame:
    """Per-school unit stats from CFBD long stats + team PPA. Index = school."""
    if stats_long.empty and ppa.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    if not stats_long.empty and {"team", "statName", "statValue"} <= set(stats_long.columns):
        wide = stats_long.pivot_table(index="team", columns="statName",
                                      values="statValue", aggfunc="first")

        def pick(name: str) -> pd.Series | None:
            for cand in _CFBD_CANDIDATES[name]:
                if cand in wide.columns:
                    return pd.to_numeric(wide[cand], errors="coerce")
            return None

        games = pick("games")
        if games is not None:
            out = pd.DataFrame(index=wide.index)
            out["games"] = games
            g = games.replace(0, pd.NA)
            npy, att = pick("net_pass_yards"), pick("pass_attempts")
            if npy is not None:
                out["pass_ypg"] = npy / g
                if att is not None:
                    out["ypa"] = npy / att.replace(0, pd.NA)
            ptd = pick("pass_tds")
            if ptd is not None:
                out["pass_td_pg"] = ptd / g
            ry, ra = pick("rush_yards"), pick("rush_attempts")
            if ry is not None:
                out["rush_ypg"] = ry / g
                if ra is not None:
                    out["ypc"] = ry / ra.replace(0, pd.NA)
            to = pick("turnovers")
            if to is not None:
                out["giveaway_pg"] = to / g
            ints, fr = pick("interceptions_made"), pick("fumbles_recovered")
            if ints is not None:
                out["takeaway_pg"] = (ints + (fr if fr is not None else 0)) / g
                out["int_created_rate"] = ints / g   # per-game proxy (no opp dropbacks)
            sk = pick("sacks_made")
            if sk is not None:
                out["sack_rate_made"] = sk / g       # per-game proxy

    if not ppa.empty and "team" in ppa.columns:
        p = ppa.set_index("team")
        ppa_map = {
            "offense.passing": "epa_dropback", "offense.rushing": "rush_epa",
            "defense.passing": "epa_dropback_allowed", "defense.rushing": "rush_epa_allowed",
        }
        cols = {}
        for src, dst in ppa_map.items():
            if src in p.columns:
                cols[dst] = pd.to_numeric(p[src], errors="coerce")
        if cols:
            ppa_frame = pd.DataFrame(cols)
            out = ppa_frame if out.empty else out.join(ppa_frame, how="outer")

    out.index.name = "team"
    return out
