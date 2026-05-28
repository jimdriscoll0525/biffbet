"""Projected score per team -- display-only run-environment estimate.

The middle-ground from docs/upgrades-roadmap.md #6: gives the user a tangible
"4.6 - 5.2" projection on every game card without replacing the transparent
weighted-sum win-probability model. The moneyline EV computation is
UNCHANGED -- this is purely additive output for the breakdown.

Formula (deliberately simple, no run-distribution machinery):

    home_RS = league_rpg
              × (home_offense_index / 100)         offense relative to league
              × (away_pitcher_rate / lg_pitch_rate) opp pitcher relative to league
              × (home_park_factor / 100)            park multiplier (both teams)

Pitcher rate blends starter (~65% of innings) + bullpen FIP (~35%) when
bullpen data is available, else starter-only. Park factor applies to BOTH
teams (they share the field).

Output is clamped to [1.0, 12.0] runs per team -- a hard sanity bound
for an MLB game; outside that range almost always means an input is
miscalibrated. Caller can read `available` to skip rendering when any
input was missing.
"""
from __future__ import annotations

from dataclasses import dataclass

from mlb_value_bot.analysis.pitcher_metrics import PitcherProfile
from mlb_value_bot.analysis.team_metrics import TeamProfile


@dataclass
class ProjectedScore:
    """Per-team expected runs scored."""

    home_runs: float | None
    away_runs: float | None
    pitcher_basis: str = ""        # "SP+RP blend" | "SP-only" | ""

    @property
    def available(self) -> bool:
        return self.home_runs is not None and self.away_runs is not None

    @property
    def total(self) -> float | None:
        if not self.available:
            return None
        return round(self.home_runs + self.away_runs, 1)

    def short_label(self) -> str:
        """Compact UI label: 'away X.X — home Y.Y' (matches matchup header)."""
        if not self.available:
            return ""
        return f"{self.away_runs:.1f} – {self.home_runs:.1f}"


def projected_score(
    home_team: TeamProfile,
    away_team: TeamProfile,
    home_pitcher: PitcherProfile,
    away_pitcher: PitcherProfile,
    config: dict,
) -> ProjectedScore:
    """Compute the projected score; returns a ProjectedScore (possibly empty).

    Degrades cleanly: any missing input -> available=False, the UI just
    doesn't render the projection line. Never raises.
    """
    lg = config.get("league", {})
    league_rpg = float(lg.get("runs_per_game", 4.5))
    lg_pitch_rate = float(lg.get("avg_xfip", 4.0))

    home_offense = home_team.offense_wrc_plus     # 100 = league avg
    away_offense = away_team.offense_wrc_plus
    home_park = float(home_team.park_factor or 100.0)

    starter_share = float(config.get("model", {}).get("projected_score_starter_share", 0.65))
    home_pitch_rate = _blended_pitcher_rate(
        home_pitcher.primary_rate(),
        home_team.bullpen_fip,
        starter_share,
    )
    away_pitch_rate = _blended_pitcher_rate(
        away_pitcher.primary_rate(),
        away_team.bullpen_fip,
        starter_share,
    )

    if (
        home_offense is None or away_offense is None
        or home_pitch_rate is None or away_pitch_rate is None
    ):
        return ProjectedScore(home_runs=None, away_runs=None)

    park_mult = home_park / 100.0
    # Home batting vs the AWAY team's pitching staff (and vice versa).
    home_rs = league_rpg * (home_offense / 100.0) * (away_pitch_rate / lg_pitch_rate) * park_mult
    away_rs = league_rpg * (away_offense / 100.0) * (home_pitch_rate / lg_pitch_rate) * park_mult

    # MLB sanity clamp. Real outliers (Coors blowouts, 1-0 pitchers' duels)
    # live mostly inside [1, 12]. Outside means an input is miscalibrated.
    home_rs = max(1.0, min(12.0, home_rs))
    away_rs = max(1.0, min(12.0, away_rs))

    basis = (
        "SP+RP blend"
        if home_team.bullpen_fip is not None and away_team.bullpen_fip is not None
        else "SP-only"
    )

    return ProjectedScore(
        home_runs=round(home_rs, 1),
        away_runs=round(away_rs, 1),
        pitcher_basis=basis,
    )


def _blended_pitcher_rate(
    starter: float | None,
    bullpen: float | None,
    starter_share: float,
) -> float | None:
    """Combined runs/9 from typical SP + RP workload. None if both missing."""
    if starter is None and bullpen is None:
        return None
    if starter is None:
        return bullpen
    if bullpen is None:
        return starter
    return starter_share * starter + (1.0 - starter_share) * bullpen
