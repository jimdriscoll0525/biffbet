"""Today's lineup status: confirmed vs. projected, and missing key bats.

MLB lineups typically post ~2 hours before first pitch. The engine runs every
~30 minutes through the closing window, so on any given run a game might be:

  * Too early to query           (> N hours from first pitch -- N tunable;
                                  default 3h, configurable in lineup.hours_before)
  * Posted by both teams         -> "confirmed"
  * Only one side / neither side -> "projected"

When projected, we don't penalize the model directly, but we do subtract a
configurable amount from the data-confidence score. That feeds the dynamic
market blend (a less-confident game leans more toward the market). It also
shows up as a "Projected lineup" chip on the UI card.

When confirmed, we cross-check today's batting order against each team's
"key bats" -- the top N hitters by season OPS, with a PA floor so a
small-sample hot stretch can't elevate a fringe player. A missing key bat
(injury, day off) is a real signal in MLB and produces a small win-prob tilt
toward the opposing team. Multiple key bats out compound, clamped tight.

Like every other component, every failure path degrades to "data unavailable"
rather than crashing the slate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from mlb_value_bot.data.mlb_client import MLBClient
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("data.lineup_status")


# Status enum-as-string (kept simple; no Enum overhead in the JSON sync path).
STATUS_CONFIRMED = "confirmed"
STATUS_PROJECTED = "projected"
STATUS_UNAVAILABLE = "unavailable"  # API failure or no data at all


@dataclass
class LineupStatus:
    team: str
    status: str                               # one of STATUS_* above
    batting_order_ids: list[int] = field(default_factory=list)
    key_bats_total: int = 0                   # how many "top N" bats we identified
    key_bats_present: int = 0                 # how many of those are in today's order
    missing_key_bats: list[str] = field(default_factory=list)  # names of missing
    notes: list[str] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        return self.status == STATUS_CONFIRMED

    @property
    def missing_count(self) -> int:
        """Count of key bats NOT in today's lineup. Used for the model tilt."""
        # Only meaningful when confirmed -- otherwise we don't know who's in.
        if not self.is_confirmed or self.key_bats_total == 0:
            return 0
        return self.key_bats_total - self.key_bats_present

    def short_label(self) -> str:
        if self.status == STATUS_UNAVAILABLE:
            return "data unavailable"
        if self.status == STATUS_PROJECTED:
            return "projected lineup"
        if self.key_bats_total == 0:
            return "confirmed"
        return f"confirmed, {self.key_bats_present}/{self.key_bats_total} key bats in"


def get_lineup_status(
    team_name: str,
    game_pk: int,
    side: str,                       # "home" | "away"
    first_pitch_iso: str | None,
    mlb: MLBClient,
    per_player_hitting: dict[str, list[dict]],
    config: dict | None = None,
    now_utc: datetime | None = None,
    _lineup_cache: dict[int, dict[str, list[int]]] | None = None,
) -> LineupStatus:
    """Build a LineupStatus for one team's slot in one game.

    `_lineup_cache` is an optional dict keyed by game_pk shared across home/away
    calls for the same game -- the lineup feed returns both sides in one
    response, so we only need to hit the API once per game per slate.
    """
    config = config or load_config()
    cfg = config.get("lineup", {})
    hours_before = float(cfg.get("hours_before_first_pitch_to_check", 3.0))
    top_n = int(cfg.get("key_bats_top_n", 3))
    min_pa = int(cfg.get("min_pa_for_key_bats", 100))

    # Resolve the team's key bats from season hitting (top N by OPS w/ PA floor).
    team_hitters = per_player_hitting.get(team_name, [])
    candidates = [h for h in team_hitters if h["pa"] >= min_pa]
    key_bats = candidates[:top_n]  # already sorted OPS desc by the API helper
    key_bat_ids = {h["player_id"] for h in key_bats if h["player_id"]}
    key_bats_total = len(key_bat_ids)

    # If it's too early in the day to expect a posted lineup, mark projected
    # WITHOUT consuming an API call. This both saves quota and avoids reading
    # a stale/empty response as "unavailable".
    too_early = False
    if first_pitch_iso and now_utc is not None:
        try:
            fp = datetime.fromisoformat(first_pitch_iso.replace("Z", "+00:00"))
            if fp.tzinfo is None:
                fp = fp.replace(tzinfo=timezone.utc)
            hours_until = (fp - now_utc).total_seconds() / 3600.0
            if hours_until > hours_before:
                too_early = True
        except ValueError:
            pass  # bad iso -> just try to fetch

    if too_early:
        return LineupStatus(
            team=team_name, status=STATUS_PROJECTED,
            key_bats_total=key_bats_total,
            notes=[f"{hours_until:.1f}h before first pitch (threshold {hours_before:.1f}h)"],
        )

    # Fetch (or reuse) the lineup feed. Game-level cache avoids double-fetching
    # for the second side of the same game.
    feed: dict[str, list[int]] | None
    if _lineup_cache is not None and game_pk in _lineup_cache:
        feed = _lineup_cache[game_pk]
    else:
        feed = mlb.get_game_lineup(game_pk)
        if _lineup_cache is not None:
            _lineup_cache[game_pk] = feed

    if not feed:
        return LineupStatus(
            team=team_name, status=STATUS_UNAVAILABLE,
            key_bats_total=key_bats_total,
            notes=["lineup feed unavailable"],
        )

    order = feed.get(side, [])
    # A real confirmed lineup is 9 deep on both sides. Anything short means
    # one or both teams haven't posted yet. We require BOTH to be posted before
    # calling this side "confirmed" -- the head-to-head matters, not just our
    # side, since the model tilt is signed (away-down minus home-down).
    other_side = "away" if side == "home" else "home"
    both_posted = len(order) >= 9 and len(feed.get(other_side, [])) >= 9

    if not both_posted:
        return LineupStatus(
            team=team_name, status=STATUS_PROJECTED,
            batting_order_ids=order,
            key_bats_total=key_bats_total,
            notes=["one or both teams' lineup not yet posted"],
        )

    # Confirmed: figure out which key bats are in / out.
    in_order = set(order)
    present_ids = key_bat_ids & in_order
    missing_ids = key_bat_ids - in_order
    name_by_id = {h["player_id"]: h["name"] for h in team_hitters if h["player_id"]}
    missing_names = sorted(name_by_id.get(pid, "?") for pid in missing_ids)

    notes: list[str] = []
    if missing_names:
        notes.append(f"out: {', '.join(missing_names)}")

    return LineupStatus(
        team=team_name,
        status=STATUS_CONFIRMED,
        batting_order_ids=order,
        key_bats_total=key_bats_total,
        key_bats_present=len(present_ids),
        missing_key_bats=missing_names,
        notes=notes,
    )
