"""Drive-stat prior tests (live_total_v1 inputs) — no network, fixture pbp only.

Locks the math the site's live over/under model depends on:
  * drive points from the posteam score delta (PAT/2pt truth),
  * kneel-only and end-of-half clock-kill drives excluded from PPD,
  * defense PPD-allowed mirrors the same drives grouped by defteam,
  * explosive play rate thresholds (pass 20 / rush 10) from config,
  * pts_per_min_trailing counts only drives that STARTED trailing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mlb_value_bot.football.analysis.drive_stats import nfl_drive_stats

CFG = {"live": {"drive_stats": {
    "explosive_pass_yards": 20, "explosive_rush_yards": 10, "min_games": 1,
}}}


def _play(game_id, posteam, defteam, drive, *, pas=0, rush=0, kneel=0,
          yards=0, pos_score=0, def_score=0, pos_score_post=None,
          top="2:30", gsr=1800):
    return {
        "game_id": game_id, "season": 2025, "week": 1, "season_type": "REG",
        "posteam": posteam, "defteam": defteam,
        "fixed_drive": drive, "fixed_drive_result": "",
        "drive_time_of_possession": top, "drive_play_count": 1,
        "game_seconds_remaining": gsr,
        "posteam_score": pos_score, "defteam_score": def_score,
        "posteam_score_post": pos_score if pos_score_post is None else pos_score_post,
        "qb_kneel": kneel, "pass": pas, "rush": rush, "yards_gained": yards,
    }


@pytest.fixture()
def fixture_pbp() -> pd.DataFrame:
    """One game, KC vs DEN. KC: a 7-pt TD drive (3 plays), then a kneel-out
    drive. DEN: a 3-pt FG drive (2 plays) started trailing, then a 1-play
    clock-kill. Drive numbering is game-wide (nflverse fixed_drive)."""
    rows = [
        # KC drive 1 -> TD+PAT (0 -> 7), 3 plays, TOP 3:00
        _play("g1", "KC", "DEN", 1, pas=1, yards=25, top="3:00", gsr=3600),
        _play("g1", "KC", "DEN", 1, rush=1, yards=5, top="3:00", gsr=3560),
        _play("g1", "KC", "DEN", 1, pas=1, yards=12, pos_score_post=7, top="3:00", gsr=3520),
        # DEN drive 2 -> FG (0 -> 3), trailing 0-7 at the start, 2 plays, TOP 2:00
        _play("g1", "DEN", "KC", 2, rush=1, yards=11, pos_score=0, def_score=7,
              top="2:00", gsr=3400),
        _play("g1", "DEN", "KC", 2, pas=1, yards=8, pos_score=0, def_score=7,
              pos_score_post=3, top="2:00", gsr=3340),
        # KC drive 3 -> kneel-only (excluded)
        _play("g1", "KC", "DEN", 3, kneel=1, top="0:40", gsr=40),
        # DEN drive 4 -> 1-play clock-kill, 8 seconds (excluded)
        _play("g1", "DEN", "KC", 4, pas=1, yards=4, top="0:08", gsr=1808),
    ]
    return pd.DataFrame(rows)


def test_ppd_from_score_delta(fixture_pbp):
    out = nfl_drive_stats(fixture_pbp, CFG)
    assert out.loc["KC", "ppd_off"] == pytest.approx(7.0)     # 1 kept drive, 7 pts
    assert out.loc["DEN", "ppd_off"] == pytest.approx(3.0)
    # Defense-allowed mirrors the same kept drives.
    assert out.loc["KC", "ppd_def_allowed"] == pytest.approx(3.0)
    assert out.loc["DEN", "ppd_def_allowed"] == pytest.approx(7.0)


def test_kneel_and_clock_kill_drives_excluded(fixture_pbp):
    out = nfl_drive_stats(fixture_pbp, CFG)
    # KC: only the TD drive survives (kneel-out excluded) -> 1 drive/game.
    assert out.loc["KC", "drives_pg"] == pytest.approx(1.0)
    assert out.loc["DEN", "drives_pg"] == pytest.approx(1.0)
    # Pace comes from kept drives only: KC 180s / 3 plays.
    assert out.loc["KC", "sec_per_play"] == pytest.approx(60.0)
    assert out.loc["KC", "plays_per_drive"] == pytest.approx(3.0)
    assert out.loc["KC", "drive_sec_avg"] == pytest.approx(180.0)


def test_explosive_play_rate_thresholds(fixture_pbp):
    out = nfl_drive_stats(fixture_pbp, CFG)
    # KC plays: 25yd pass (explosive), 5yd rush, 12yd pass -> 1/3.
    assert out.loc["KC", "explosive_play_rate"] == pytest.approx(1 / 3)
    # DEN plays: 11yd rush (explosive), 8yd pass, 4yd pass (clock-kill play
    # still counts at PLAY level) -> 1/3.
    assert out.loc["DEN", "explosive_play_rate"] == pytest.approx(1 / 3)


def test_pts_per_min_trailing_only_counts_trailing_starts(fixture_pbp):
    out = nfl_drive_stats(fixture_pbp, CFG)
    # DEN's FG drive started 0-7 down: 3 pts over 2:00 -> 1.5 pts/min.
    assert out.loc["DEN", "pts_per_min_trailing"] == pytest.approx(1.5)
    # KC never trailed -> NaN, not 0 (absence of evidence, not zero rate).
    assert pd.isna(out.loc["KC", "pts_per_min_trailing"])


def test_empty_and_min_games():
    assert nfl_drive_stats(pd.DataFrame(), CFG).empty
    cfg_strict = {"live": {"drive_stats": {"min_games": 2}}}
    df = pd.DataFrame([_play("g1", "KC", "DEN", 1, pas=1, yards=30,
                             pos_score_post=7)])
    assert nfl_drive_stats(df, cfg_strict).empty
