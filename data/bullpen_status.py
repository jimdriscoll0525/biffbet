"""Bullpen availability / fatigue from MLB Stats API game logs.

Replaces the static "season bullpen FIP" view of a team's pen with a
fatigue-aware status: who pitched yesterday, who's on back-to-back days,
who's likely unavailable today, and how many high-leverage arms each team
has free. The model uses this as an *additive* component on top of the
existing bullpen FIP delta -- it's a tilt, not a replacement.

Data source: MLB Stats API (free, key-less), same source the rest of the
engine already uses (data.mlb_client). No FanGraphs dependency.

Definitions (all tunable in config.bullpen_fatigue):
  * leverage arm     each team's top N relievers by season ERA (default 3),
                     among relievers with >= min_appearances on the year so
                     a single-game stat fluke can't elevate a fringe arm.
  * unavailable      threw >= pitch_threshold pitches yesterday, OR was on
                     back-to-back days yesterday + two-days-ago, OR has
                     pitched in >= appearance_threshold games over the last
                     3 days.
  * b2b_arm          pitched on the previous two consecutive calendar days.

Graceful degradation: if any of (recent games / boxscores / per-pitcher
season stats) come back empty, get_bullpen_status returns a status with
`available=False`. The model component then contributes 0 with the note
"bullpen status unavailable" rather than crashing the slate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_cls, timedelta

from mlb_value_bot.data.mlb_client import MLBClient
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("data.bullpen_status")


@dataclass
class RelieverUsage:
    """One reliever's recent-usage snapshot."""

    player_id: int | None
    name: str
    season_era: float | None = None
    is_leverage: bool = False
    # Pitches thrown on each of the last 3 days (yesterday, -2d, -3d).
    pitches_by_day: list[int] = field(default_factory=lambda: [0, 0, 0])
    appearances_3d: int = 0
    consecutive_days: int = 0   # streak ending yesterday (0 = didn't pitch yesterday)

    @property
    def total_pitches_3d(self) -> int:
        return sum(self.pitches_by_day)

    def is_unavailable(self, pitch_threshold: int, appearance_threshold: int) -> bool:
        """True if this reliever is unlikely to be available today."""
        if self.pitches_by_day[0] >= pitch_threshold:    # heavy outing yesterday
            return True
        if self.consecutive_days >= 2:                    # already on B2B
            return True
        if self.appearances_3d >= appearance_threshold:   # used up over 3 days
            return True
        return False


@dataclass
class BullpenStatus:
    """A team's bullpen state for an as-of date."""

    team: str
    available: bool                      # False = data unavailable; component must skip
    relievers: list[RelieverUsage] = field(default_factory=list)
    leverage_total: int = 0
    leverage_unavailable: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def leverage_available(self) -> int:
        return self.leverage_total - self.leverage_unavailable

    @property
    def fatigue_score(self) -> int:
        """Higher = more leverage arms down. Used by the win-prob component."""
        return self.leverage_unavailable

    def short_label(self) -> str:
        """Compact UI string: '2/3 leverage arms available'."""
        if not self.available or self.leverage_total == 0:
            return "data unavailable"
        return f"{self.leverage_available}/{self.leverage_total} leverage arms available"


def get_bullpen_status(
    team_name: str,
    team_id: int,
    as_of: date_cls,
    mlb: MLBClient,
    per_pitcher_relievers: dict[str, list[dict]],
    config: dict | None = None,
) -> BullpenStatus:
    """Build a BullpenStatus for `team_name` as of `as_of`.

    `per_pitcher_relievers` is the output of MLBClient.get_per_pitcher_reliever_stats
    -- shared across teams in a single run so we only hit the all-pitchers
    endpoint once per slate. `team_id` is required because the recent-games
    endpoint needs the numeric MLB team ID, not the canonical name.

    Any individual data gap returns a status with available=False rather than
    raising. This component is a tilt, never a hard requirement.
    """
    config = config or load_config()
    cfg = config.get("bullpen_fatigue", {})
    pitch_threshold = int(cfg.get("pitch_threshold_unavailable", 35))
    appearance_threshold = int(cfg.get("appearance_threshold", 3))
    leverage_top_n = int(cfg.get("leverage_top_n", 3))
    min_appearances = int(cfg.get("min_appearances_for_leverage", 8))
    lookback_days = int(cfg.get("lookback_days", 5))

    # Resolve the team's leverage-arm list (top N by season ERA, with a sample-
    # size floor). Sorted asc by ERA in per_pitcher_relievers already.
    team_rp = per_pitcher_relievers.get(team_name, [])
    leverage_candidates = [r for r in team_rp if r["appearances"] >= min_appearances]
    leverage_ids: set[int] = {
        r["player_id"] for r in leverage_candidates[:leverage_top_n]
        if r["player_id"] is not None
    }

    if not team_id:
        return BullpenStatus(
            team=team_name, available=False,
            notes=["no team_id resolved (can't fetch recent games)"],
        )

    # Pull the team's last ~5 days of finalized games.
    recent = mlb.get_recent_games_for_team(team_id, end_date=as_of, days_back=lookback_days)
    if not recent:
        return BullpenStatus(
            team=team_name, available=False,
            notes=["no recent finalized games found"],
        )

    # Aggregate pitches by (player_id, day_offset). day_offset 0 = yesterday,
    # 1 = two days ago, 2 = three days ago. Anything older doesn't count for
    # standard one-day-of-rest rules. We accept multiple appearances per day
    # (rare for relievers in MLB but possible).
    by_pid: dict[int, dict] = {}
    days_seen: dict[int, set[str]] = {}  # player_id -> set of dates pitched
    for g in recent[:lookback_days]:
        game_date = g["date"]
        try:
            d = date_cls.fromisoformat(game_date)
        except ValueError:
            continue
        offset = (as_of - d).days - 1  # 0 = yesterday
        if offset < 0 or offset > 2:
            continue
        log_data = mlb.get_pitching_log(g["gamePk"])
        if not log_data:
            continue
        # The team can be home OR away in any given game -- find the side.
        for side in ("home", "away"):
            for p in log_data.get(side, []):
                pid = p["player_id"]
                if pid is None or p["is_starter"]:
                    continue
                # Side-attribution: we only care about pitchers belonging to
                # THIS team. The boxscore lists both sides under one game; we
                # use season-stats membership in per_pitcher_relievers to filter.
                # (A pitcher might be in BOTH teams' rosters across a trade,
                # but within a single game they appear under one side, and the
                # season stats are typically aggregated to the latest team.)
                if pid not in {r["player_id"] for r in team_rp if r["player_id"] is not None}:
                    continue
                entry = by_pid.setdefault(pid, {
                    "name": p["name"],
                    "pitches_by_day": [0, 0, 0],
                    "dates": set(),
                })
                entry["pitches_by_day"][offset] += int(p.get("pitches") or 0)
                entry["dates"].add(game_date)
                days_seen.setdefault(pid, set()).add(game_date)

    # Build RelieverUsage rows, but include ALL leverage arms even if they
    # didn't pitch in the window -- a leverage arm with zero recent usage IS
    # available, and we still want to count them toward leverage_total.
    relievers: list[RelieverUsage] = []
    era_by_pid = {r["player_id"]: r["era"] for r in team_rp if r["player_id"]}
    name_by_pid = {r["player_id"]: r["name"] for r in team_rp if r["player_id"]}

    all_pids = set(leverage_ids) | set(by_pid.keys())
    for pid in all_pids:
        entry = by_pid.get(pid, {"name": name_by_pid.get(pid, "?"),
                                 "pitches_by_day": [0, 0, 0], "dates": set()})
        # Consecutive-days streak ending yesterday: walk back from yesterday
        # while the date is in entry["dates"].
        streak = 0
        for offset_back in range(3):
            d = (as_of - timedelta(days=offset_back + 1)).isoformat()
            if d in entry["dates"]:
                streak += 1
            else:
                break
        relievers.append(RelieverUsage(
            player_id=pid,
            name=entry["name"],
            season_era=era_by_pid.get(pid),
            is_leverage=pid in leverage_ids,
            pitches_by_day=entry["pitches_by_day"],
            appearances_3d=len(entry["dates"]),
            consecutive_days=streak,
        ))

    leverage_total = sum(1 for r in relievers if r.is_leverage)
    leverage_unavailable = sum(
        1 for r in relievers
        if r.is_leverage and r.is_unavailable(pitch_threshold, appearance_threshold)
    )

    notes: list[str] = []
    if leverage_total == 0:
        notes.append("no leverage arms identified (insufficient reliever sample)")
    if leverage_unavailable > 0:
        down_names = ", ".join(
            r.name for r in relievers
            if r.is_leverage and r.is_unavailable(pitch_threshold, appearance_threshold)
        )
        notes.append(f"down: {down_names}")

    return BullpenStatus(
        team=team_name,
        available=leverage_total > 0,
        relievers=relievers,
        leverage_total=leverage_total,
        leverage_unavailable=leverage_unavailable,
        notes=notes,
    )
