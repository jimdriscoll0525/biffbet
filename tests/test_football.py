"""Football model tests — no network, no nflreadpy (fixture frames only).

Locks the invariants Jim called out explicitly:
  * records filtered by model_tag x league x market (the GriffBet record bug),
  * push grading (exactly on the number = push, stake-neutral),
  * percentile / edge-score math (league pools never mixed),
plus the football twins of the MLB suite's guardrails (weather signed/degrade,
tilt symmetry, single-homed sharp-fade, opening-line freeze).

Tests are added milestone by milestone; every section is headed by the
milestone that introduced it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mlb_value_bot.football import season_for_date


# =============================================================================
# M1 — data layer
# =============================================================================

def test_season_for_date_spans_january():
    assert season_for_date("2026-09-10") == 2026
    assert season_for_date("2027-01-11") == 2026   # playoffs belong to the prior season
    assert season_for_date("2026-07-04") == 2025


class TestNflNames:
    def test_full_names_map_to_nflverse_abbr(self):
        from mlb_value_bot.football.data.teams import normalize_nfl

        assert normalize_nfl("Kansas City Chiefs") == "KC"
        assert normalize_nfl("Los Angeles Rams") == "LA"
        assert normalize_nfl("Washington Commanders") == "WAS"

    def test_mlb_nickname_trap_does_not_apply(self):
        """The reason football does NOT reuse constants.normalize_team: its
        nickname fallback maps NY Giants -> SF Giants (MLB). Football's map
        must resolve these correctly."""
        from mlb_value_bot.football.data.teams import normalize_nfl

        assert normalize_nfl("New York Giants") == "NYG"
        assert normalize_nfl("Arizona Cardinals") == "ARI"

    def test_historic_abbr_aliases(self):
        from mlb_value_bot.football.data.teams import normalize_nfl

        assert normalize_nfl("OAK") == "LV"
        assert normalize_nfl("LAR") == "LA"

    def test_unknown_returns_none(self):
        from mlb_value_bot.football.data.teams import normalize_nfl

        assert normalize_nfl("Springfield Atoms") is None
        assert normalize_nfl(None) is None


class TestCfbMatcher:
    FIXTURE = [
        ("Alabama", "Crimson Tide"),
        ("Ohio State", "Buckeyes"),
        ("Ohio", "Bobcats"),
        ("Miami", "Hurricanes"),
        ("Miami (OH)", "RedHawks"),
        ("App State", "Mountaineers"),
    ]

    def test_exact_school_plus_mascot(self):
        from mlb_value_bot.football.data.teams import build_cfb_matcher

        norm = build_cfb_matcher(self.FIXTURE)
        assert norm("Alabama Crimson Tide") == "Alabama"
        assert norm("Ohio State Buckeyes") == "Ohio State"

    def test_prefix_prefers_longest_school(self):
        """'Ohio State Buckeyes' must never resolve to 'Ohio'."""
        from mlb_value_bot.football.data.teams import build_cfb_matcher

        norm = build_cfb_matcher([(s, None) for s, _ in self.FIXTURE])
        assert norm("Ohio State Buckeyes") == "Ohio State"
        assert norm("Ohio Bobcats") == "Ohio"

    def test_manual_overrides_win(self):
        from mlb_value_bot.football.data.teams import build_cfb_matcher

        norm = build_cfb_matcher(self.FIXTURE)
        assert norm("Miami Hurricanes") == "Miami"
        assert norm("Miami (OH) RedHawks") == "Miami (OH)"
        assert norm("Appalachian State Mountaineers") == "App State"

    def test_unmatched_returns_none(self):
        from mlb_value_bot.football.data.teams import build_cfb_matcher

        norm = build_cfb_matcher(self.FIXTURE)
        assert norm("North Dakota State Bison") is None   # FCS opponent -> skipped


class TestFootballWeather:
    CFG = {"max_tilt": 0.06, "wind_mph_start": 12.0, "wind_coef_per_mph": 0.006,
           "cold_f_start": 25.0, "cold_coef_per_f": 0.004}

    def test_weather_is_suppress_only_and_signed_down(self):
        """Football weather can move a total DOWN, never up."""
        from mlb_value_bot.football.data.football_weather import suppression_multiplier

        calm = suppression_multiplier(70.0, 5.0, self.CFG)
        windy = suppression_multiplier(70.0, 22.0, self.CFG)
        frigid = suppression_multiplier(5.0, 5.0, self.CFG)
        assert calm == 1.0
        assert windy < 1.0
        assert frigid < 1.0
        assert suppression_multiplier(95.0, 0.0, self.CFG) == 1.0   # heat never boosts

    def test_multiplier_clamped_at_max_tilt(self):
        from mlb_value_bot.football.data.football_weather import suppression_multiplier

        hurricane = suppression_multiplier(-20.0, 60.0, self.CFG)
        assert hurricane == pytest.approx(1.0 - self.CFG["max_tilt"])

    def test_indoor_is_available_noop(self):
        from mlb_value_bot.football.data.football_weather import game_weather

        env = game_weather(44.9738, -93.2577, "2026-11-15T18:00:00Z", indoor=True,
                           config={"weather": self.CFG})
        assert env.multiplier == 1.0
        assert env.available is True      # a KNOWN no-effect state, not a gap
        assert env.indoor is True

    def test_missing_coords_degrades_unavailable(self):
        from mlb_value_bot.football.data.football_weather import game_weather

        env = game_weather(None, None, "2026-11-15T18:00:00Z", indoor=False,
                           config={"weather": self.CFG})
        assert env.multiplier == 1.0
        assert env.available is False     # -> pipeline holds outdoor totals


class TestStadiums:
    def test_roof_value_wins_over_fallback(self):
        from mlb_value_bot.football.data.stadiums import is_indoor

        # ARI is in the fallback dome set, but an 'open' roof game is outdoor.
        assert is_indoor("open", "ARI") is False
        assert is_indoor("closed", "GB") is True
        assert is_indoor("dome", None) is True
        assert is_indoor(None, "MIN") is True
        assert is_indoor(None, "GB") is False

    def test_all_32_teams_have_coords(self):
        from mlb_value_bot.football.data.stadiums import NFL_STADIUM_COORDS
        from mlb_value_bot.football.data.teams import NFL_ABBR_TO_NAME

        assert set(NFL_ABBR_TO_NAME) <= set(NFL_STADIUM_COORDS)

    def test_cfb_venue_lookup_degrades(self):
        from mlb_value_bot.football.data.stadiums import cfb_venue

        venues = pd.DataFrame([
            {"id": 1, "location.latitude": 33.2, "location.longitude": -87.5, "dome": False},
        ])
        hit = cfb_venue(venues, 1)
        assert hit == {"lat": 33.2, "lon": -87.5, "dome": False}
        assert cfb_venue(venues, 999) is None
        assert cfb_venue(pd.DataFrame(), 1) is None
        assert cfb_venue(venues, None) is None


# =============================================================================
# M2 — matchup scoring
# =============================================================================

def _units(**kw):
    base = {"pass_off_pct": 50.0, "pass_def_pct": 50.0, "rush_off_pct": 50.0,
            "rush_def_pct": 50.0, "takeaway_pct": 50.0, "ball_security_pct": 50.0}
    base.update(kw)
    return base


M2_CFG = {
    "percentiles": {"strong_threshold": 75, "weak_threshold": 25,
                    "prior_weight_week1": 0.8, "prior_out_week": 7},
    "matchup": {"phase_weight_pass": 0.60, "phase_weight_rush": 0.40,
                "script_shift_max": 0.15, "neutral_band": 10.0,
                "dual_edge_bonus": 0.15, "turnover_flag_threshold": 75},
    "ol_layer": {"pressure_elite": 0.15, "pressure_poor": 0.28,
                 "ypc_good_nfl": 4.5, "ypc_bad_nfl": 4.0,
                 "ypc_good_cfb": 5.0, "ypc_bad_cfb": 4.3,
                 "continuity_dampener_max": 0.30, "edge_points_scale": 10.0},
}


class TestPercentiles:
    def test_percentile_math_league_pools(self):
        """Exact 0-100 percentiles on a small pool; direction honored; and the
        pool is whatever frame is passed — NFL and FBS are percentiled by
        SEPARATE calls, never concatenated (locked by construction here)."""
        from mlb_value_bot.football.analysis.percentiles import unit_percentiles

        nfl = pd.DataFrame({"rush_ypg": [80.0, 120.0, 160.0, 200.0],
                            "ypc": [3.8, 4.2, 4.6, 5.0],
                            "rush_epa": [-0.1, 0.0, 0.05, 0.1]},
                           index=["A", "B", "C", "D"])
        pcts = unit_percentiles(nfl)
        assert pcts.loc["D", "rush_off_pct"] == 100.0
        assert pcts.loc["A", "rush_off_pct"] == 25.0

        # Same team stats in a DIFFERENT pool -> different percentile: proof
        # that percentiles are pool-relative, so pools must never be mixed.
        fbs = pd.DataFrame({"rush_ypg": [80.0, 90.0], "ypc": [3.8, 3.9],
                            "rush_epa": [-0.1, -0.05]}, index=["A", "X"])
        assert unit_percentiles(fbs).loc["A", "rush_off_pct"] != \
            pcts.loc["A", "rush_off_pct"]

    def test_lower_is_better_stats_invert(self):
        from mlb_value_bot.football.analysis.percentiles import unit_percentiles

        df = pd.DataFrame({"giveaway_pg": [0.5, 1.0, 2.0]}, index=["A", "B", "C"])
        pcts = unit_percentiles(df)
        assert pcts.loc["A", "ball_security_pct"] == 100.0   # fewest giveaways
        assert pcts.loc["C", "ball_security_pct"] < 50.0

    def test_missing_stats_average_available_and_empty_unit_is_nan(self):
        from mlb_value_bot.football.analysis.percentiles import unit_percentiles

        df = pd.DataFrame({"rush_epa": [0.1, -0.1]}, index=["A", "B"])  # CFB-ish
        pcts = unit_percentiles(df)
        assert pcts.loc["A", "rush_off_pct"] == 100.0        # 1-of-3 stats present
        assert pd.isna(pcts.loc["A", "pass_off_pct"])        # 0 stats -> NaN

    def test_prior_blend_decay(self):
        from mlb_value_bot.football.analysis.percentiles import blend_with_prior, prior_weight

        assert prior_weight(1, M2_CFG) == pytest.approx(0.8)
        assert prior_weight(7, M2_CFG) == 0.0
        assert 0.0 < prior_weight(4, M2_CFG) < 0.8

        cur = pd.DataFrame({"ypc": [4.0], "games": [2]}, index=["A"])
        pri = pd.DataFrame({"ypc": [5.0], "games": [17]}, index=["A"])
        wk1 = blend_with_prior(cur, pri, 1, M2_CFG)
        assert wk1.loc["A", "ypc"] == pytest.approx(0.2 * 4.0 + 0.8 * 5.0)
        assert wk1.loc["A", "games"] == 2                    # games stays current
        wk9 = blend_with_prior(cur, pri, 9, M2_CFG)
        assert wk9.loc["A", "ypc"] == 4.0
        # Team missing from prior (promoted program) keeps current values.
        pri_other = pd.DataFrame({"ypc": [5.0]}, index=["Z"])
        assert blend_with_prior(cur, pri_other, 1, M2_CFG).loc["A", "ypc"] == 4.0


class TestEdgeScoreAndArchetypes:
    def test_edge_formula_matches_jims_convention(self):
        """edge = O_pct - D_pct (defense oriented higher=better) is identical
        to Jim's 'O - (100 - D_badness)'."""
        from mlb_value_bot.football.analysis.matchup import classify_phase

        e = classify_phase(78.0, 8.0, M2_CFG, "home", "rush")
        d_badness = 100.0 - 8.0
        assert e.edge == pytest.approx(78.0 - (100.0 - d_badness))
        assert e.edge == pytest.approx(70.0)
        assert e.archetype == "strong_o_vs_weak_d"

    def test_strong_vs_strong_and_weak_vs_weak_are_neutral(self):
        from mlb_value_bot.football.analysis.matchup import classify_phase

        assert classify_phase(85.0, 88.0, M2_CFG, "home", "pass").archetype == "neutral"
        assert classify_phase(15.0, 12.0, M2_CFG, "home", "pass").archetype == "neutral"

    def test_weak_o_vs_strong_d_is_defense_edge(self):
        from mlb_value_bot.football.analysis.matchup import classify_phase

        e = classify_phase(15.0, 90.0, M2_CFG, "away", "pass")
        assert e.edge == pytest.approx(-75.0)
        assert e.archetype == "weak_o_vs_strong_d"

    def test_unevaluable_unit_yields_no_edge(self):
        from mlb_value_bot.football.analysis.matchup import classify_phase

        assert classify_phase(None, 50.0, M2_CFG, "home", "pass").edge is None
        assert classify_phase(float("nan"), 50.0, M2_CFG, "home", "pass").edge is None

    def test_dual_edge_detection_and_bonus(self):
        from mlb_value_bot.football.analysis.matchup import game_matchup

        home = _units(pass_off_pct=85.0, rush_off_pct=82.0)
        away = _units(pass_def_pct=20.0, rush_def_pct=18.0)
        m = game_matchup(home, away, M2_CFG)
        assert m.dual_edge_side == "home"
        assert m.archetype == "dual_edge"
        # Bonus amplifies: net edge exceeds the unboosted weighted sum.
        unboosted = 0.6 * (85 - 20) + 0.4 * (82 - 18)
        assert m.home_edge > unboosted

    def test_dual_edge_bonus_never_flips_an_opposing_edge(self):
        from mlb_value_bot.football.analysis.matchup import game_matchup

        # Away holds both phase edges -> home_edge is negative and the bonus
        # must push it MORE negative, never toward home.
        home = _units(pass_def_pct=20.0, rush_def_pct=18.0)
        away = _units(pass_off_pct=85.0, rush_off_pct=82.0)
        m = game_matchup(home, away, M2_CFG)
        assert m.dual_edge_side == "away"
        assert m.home_edge < 0

    def test_turnover_pairing_flag(self):
        from mlb_value_bot.football.analysis.matchup import game_matchup

        home = _units(takeaway_pct=90.0)
        away = _units(ball_security_pct=10.0)
        m = game_matchup(home, away, M2_CFG)
        assert m.turnover_flag_side == "home"
        assert game_matchup(_units(), _units(), M2_CFG).turnover_flag_side is None

    def test_script_shift_bounded_and_normalized(self):
        from mlb_value_bot.football.analysis.matchup import phase_weights

        wp0, wr0 = phase_weights(M2_CFG, 0.0)
        assert (wp0, wr0) == (pytest.approx(0.6), pytest.approx(0.4))
        wp, wr = phase_weights(M2_CFG, 5.0)      # lean clamped to 1
        assert wp == pytest.approx(0.45) and wr == pytest.approx(0.55)
        assert wp + wr == pytest.approx(1.0)


class TestOLLayer:
    def test_ol_grades_from_proxies(self):
        from mlb_value_bot.football.analysis.ol_layer import ol_grade

        elite = ol_grade({"pressure_proxy_rate": 0.12, "ypc": 4.8}, "nfl", M2_CFG)
        poor = ol_grade({"pressure_proxy_rate": 0.32, "ypc": 3.7}, "nfl", M2_CFG)
        assert elite.grade == 1.0 and elite.points == 10.0
        assert poor.grade == -1.0
        assert any("proxy" in n for n in elite.notes)   # labeled a proxy

    def test_ol_continuity_dampener_bounded_positive_only(self):
        from mlb_value_bot.football.analysis.ol_layer import ol_grade

        strong = ol_grade({"pressure_proxy_rate": 0.12, "ypc": 4.8}, "nfl", M2_CFG,
                          continuity=0.0)
        assert strong.grade == pytest.approx(1.0 * (1 - 0.30))   # max dampening
        # Negative grades are never "improved" by bad continuity.
        bad = ol_grade({"pressure_proxy_rate": 0.32, "ypc": 3.7}, "nfl", M2_CFG,
                       continuity=0.0)
        assert bad.grade == -1.0

    def test_cfb_falls_back_to_sack_rate_and_own_ypc_scale(self):
        from mlb_value_bot.football.analysis.ol_layer import ol_grade

        g = ol_grade({"sack_rate_allowed": 0.12, "ypc": 5.2}, "cfb", M2_CFG)
        assert g.grade == 1.0
        # 4.8 ypc is elite for NFL (>=4.5) but not yet for CFB's 4.3-5.0 band.
        nfl_g = ol_grade({"ypc": 4.8}, "nfl", M2_CFG)
        cfb_g = ol_grade({"ypc": 4.8}, "cfb", M2_CFG)
        assert nfl_g.grade == 1.0 and 0.0 < cfb_g.grade < 1.0

    def test_nfl_continuity_from_snap_overlap(self):
        from mlb_value_bot.football.analysis.ol_layer import nfl_ol_continuity

        rows = []
        for wk, players in ((3, list("ABCDE")), (4, list("ABCXY"))):
            for p in players:
                rows.append({"team": "GB", "week": wk, "position": "G",
                             "player": p, "offense_snaps": 60})
            rows.append({"team": "GB", "week": wk, "position": "QB",
                         "player": "QB1", "offense_snaps": 60})
        df = pd.DataFrame(rows)
        assert nfl_ol_continuity(df, "GB", 5) == pytest.approx(3 / 5)
        assert nfl_ol_continuity(df, "GB", 4) is None       # only one prior week
        assert nfl_ol_continuity(pd.DataFrame(), "GB", 5) is None


class TestNflUnitStats:
    def _pbp(self):
        # Two teams, one game: A passes well vs B; B runs poorly vs A.
        rows = [
            # posteam A dropbacks (3 attempts, 1 sack, 1 TD, 1 INT)
            dict(game_id="g1", posteam="A", defteam="B", pass_=1, rush=0, qb_dropback=1,
                 pass_attempt=1, rush_attempt=0, sack=0, qb_hit=0, interception=0,
                 fumble_lost=0, pass_touchdown=1, rush_touchdown=0, touchdown=1,
                 yards_gained=25.0, epa=2.0, qtr=1, yardline_100=30),
            dict(game_id="g1", posteam="A", defteam="B", pass_=1, rush=0, qb_dropback=1,
                 pass_attempt=1, rush_attempt=0, sack=0, qb_hit=1, interception=1,
                 fumble_lost=0, pass_touchdown=0, rush_touchdown=0, touchdown=0,
                 yards_gained=0.0, epa=-2.5, qtr=2, yardline_100=60),
            dict(game_id="g1", posteam="A", defteam="B", pass_=1, rush=0, qb_dropback=1,
                 pass_attempt=0, rush_attempt=0, sack=1, qb_hit=1, interception=0,
                 fumble_lost=0, pass_touchdown=0, rush_touchdown=0, touchdown=0,
                 yards_gained=-7.0, epa=-1.5, qtr=4, yardline_100=50),
            # posteam B rushes (2 carries)
            dict(game_id="g1", posteam="B", defteam="A", pass_=0, rush=1, qb_dropback=0,
                 pass_attempt=0, rush_attempt=1, sack=0, qb_hit=0, interception=0,
                 fumble_lost=1, pass_touchdown=0, rush_touchdown=0, touchdown=0,
                 yards_gained=3.0, epa=-0.2, qtr=1, yardline_100=70),
            dict(game_id="g1", posteam="B", defteam="A", pass_=0, rush=1, qb_dropback=0,
                 pass_attempt=0, rush_attempt=1, sack=0, qb_hit=0, interception=0,
                 fumble_lost=0, pass_touchdown=0, rush_touchdown=0, touchdown=0,
                 yards_gained=5.0, epa=0.1, qtr=4, yardline_100=45),
        ]
        df = pd.DataFrame(rows).rename(columns={"pass_": "pass"})
        return df

    def test_offense_and_defense_mirror(self):
        from mlb_value_bot.football.analysis.unit_stats import nfl_unit_stats

        s = nfl_unit_stats(self._pbp())
        assert s.loc["A", "pass_ypg"] == pytest.approx(25.0)   # sack yards excluded
        assert s.loc["A", "ypa"] == pytest.approx(12.5)
        assert s.loc["A", "sack_rate_allowed"] == pytest.approx(1 / 3)
        assert s.loc["A", "pressure_proxy_rate"] == pytest.approx(2 / 3)
        assert s.loc["A", "int_rate"] == pytest.approx(1 / 3)
        assert s.loc["A", "giveaway_pg"] == pytest.approx(1.0)
        # B's defense sees exactly A's offense:
        assert s.loc["B", "pass_ypg_allowed"] == pytest.approx(25.0)
        assert s.loc["B", "sack_rate_made"] == pytest.approx(1 / 3)
        assert s.loc["B", "takeaway_pg"] == pytest.approx(1.0)
        # And A's defense sees B's rushing + fumble:
        assert s.loc["A", "ypc_allowed"] == pytest.approx(4.0)
        assert s.loc["A", "takeaway_pg"] == pytest.approx(1.0)

    def test_empty_pbp_degrades(self):
        from mlb_value_bot.football.analysis.unit_stats import nfl_unit_stats

        assert nfl_unit_stats(pd.DataFrame()).empty


class TestCfbUnitStats:
    def test_long_stats_and_ppa_join(self):
        from mlb_value_bot.football.analysis.unit_stats import cfb_unit_stats

        stats = pd.DataFrame([
            {"team": "Alabama", "statName": "games", "statValue": 12},
            {"team": "Alabama", "statName": "netPassingYards", "statValue": 3600},
            {"team": "Alabama", "statName": "passAttempts", "statValue": 400},
            {"team": "Alabama", "statName": "rushingYards", "statValue": 2400},
            {"team": "Alabama", "statName": "rushingAttempts", "statValue": 480},
            {"team": "Alabama", "statName": "turnovers", "statValue": 12},
            {"team": "Alabama", "statName": "interceptions", "statValue": 14},
            {"team": "Alabama", "statName": "fumblesRecovered", "statValue": 6},
            {"team": "Alabama", "statName": "sacks", "statValue": 36},
        ])
        ppa = pd.DataFrame([{
            "team": "Alabama", "offense.passing": 0.35, "offense.rushing": 0.15,
            "defense.passing": -0.10, "defense.rushing": -0.05,
        }])
        s = cfb_unit_stats(stats, ppa)
        assert s.loc["Alabama", "pass_ypg"] == pytest.approx(300.0)
        assert s.loc["Alabama", "ypa"] == pytest.approx(9.0)
        assert s.loc["Alabama", "ypc"] == pytest.approx(5.0)
        assert s.loc["Alabama", "giveaway_pg"] == pytest.approx(1.0)
        assert s.loc["Alabama", "takeaway_pg"] == pytest.approx(20 / 12)
        assert s.loc["Alabama", "epa_dropback"] == pytest.approx(0.35)
        assert s.loc["Alabama", "epa_dropback_allowed"] == pytest.approx(-0.10)

    def test_ppa_only_still_produces_frame(self):
        from mlb_value_bot.football.analysis.unit_stats import cfb_unit_stats

        ppa = pd.DataFrame([{"team": "Ohio State", "offense.passing": 0.4,
                             "defense.rushing": -0.2}])
        s = cfb_unit_stats(pd.DataFrame(), ppa)
        assert s.loc["Ohio State", "epa_dropback"] == pytest.approx(0.4)


# =============================================================================
# M3 — projections, EV, adjusted EV, confidence
# =============================================================================

M3_CFG = {
    **M2_CFG,
    "projections": {"market_blend": 0.35, "base_pts_nfl": 22.5, "base_pts_cfb": 28.5,
                    "hfa_pts_nfl": 1.5, "hfa_pts_cfb": 2.5, "rz_coef_pts": 8.0,
                    "nfl_margin_sigma": 13.2, "nfl_total_sigma": 10.0,
                    "cfb_margin_sigma": 16.0, "cfb_total_sigma": 12.5,
                    "max_total_divergence_pts": 6.0, "max_spread_divergence_pts": 4.5,
                    "script_lean_full_spread": 21.0},
    "ev": {"devig_method": "power", "threshold": 0.03, "max_picks_per_game": 3},
    "adjusted_ev": {"sharp_support_pp": 2.0, "sharp_support_boost": 0.010,
                    "sharp_fade_reduction": 0.015, "fragile_reduction": 0.010},
    "betting": {"paper_only": True, "flat_stake_pct": 0.01, "flat_stake_pct_strong": 0.02},
    "college": {"league_confidence_mult": 0.90, "g5_confidence_mult": 0.85,
                "g5_stake_mult": 0.75, "sp_blend": 0.30},
    "variance": {"explosive_epa_pctile": 80, "explosive_confidence_mult": 0.90},
    "weather": {"enabled": True, "require_for_bet": True, "max_tilt": 0.06,
                "wind_mph_start": 12.0, "wind_coef_per_mph": 0.006,
                "cold_f_start": 25.0, "cold_coef_per_f": 0.004},
    "odds_api": {"bet_bookmaker": "draftkings",
                 "sharp_bookmakers": ["pinnacle"], "square_bookmakers": ["draftkings"]},
    "model_tag": "matchup_v1",
}


class TestProjectionMath:
    def test_projection_tilt_symmetry_no_over_bias(self):
        """The football twin of test_totals' tilt-symmetry lock: a model total
        X points ABOVE the line must produce the same over-probability as one
        X points BELOW produces for the under."""
        from mlb_value_bot.football.analysis.projections import total_probabilities

        line, sigma, x = 44.0, 10.0, 3.0
        p_over_hi, push_hi, _ = total_probabilities(line + x, line, sigma)
        _, push_lo, p_under_lo = total_probabilities(line - x, line, sigma)
        assert p_over_hi == pytest.approx(p_under_lo, abs=1e-12)
        assert push_hi == pytest.approx(push_lo, abs=1e-12)

    def test_push_mass_only_on_integer_lines(self):
        from mlb_value_bot.football.analysis.projections import (
            cover_probabilities, total_probabilities)

        _, push_int, _ = total_probabilities(44.0, 44.0, 10.0)
        _, push_half, _ = total_probabilities(44.0, 44.5, 10.0)
        assert push_int > 0.0 and push_half == 0.0
        ph, pp, pa = cover_probabilities(3.0, -3.0, 13.2)   # home -3, projects -3
        assert pp > 0.0 and ph + pp + pa == pytest.approx(1.0)

    def test_anchor_mean_recovers_skewed_juice(self):
        """Symmetric juice -> anchor == the line. Over-juiced -> anchor ABOVE
        the line (the market's mean is higher than its median). This is the
        'measure tilt vs the mean, not the line' fix."""
        from mlb_value_bot.football.analysis.projections import market_anchor_mean

        assert market_anchor_mean(44.0, 0.5, 10.0, "total") == pytest.approx(44.0)
        assert market_anchor_mean(44.0, 0.55, 10.0, "total") > 44.0
        # Spread: home -3 with symmetric juice -> mean margin +3 (home by 3).
        assert market_anchor_mean(-3.0, 0.5, 13.2, "spread") == pytest.approx(3.0)

    def test_weather_only_lowers_totals(self):
        from mlb_value_bot.football.analysis.projections import project_game

        units_h = {"epa_dropback": 0.10, "rush_epa": 0.00, "plays_pg": 63.0,
                   "epa_dropback_allowed": 0.0, "rush_epa_allowed": 0.0}
        units_a = dict(units_h)
        clear = project_game(units_h, units_a, "nfl", M3_CFG, 0.6, 0.4, weather_mult=1.0)
        windy = project_game(units_h, units_a, "nfl", M3_CFG, 0.6, 0.4, weather_mult=0.94)
        boosted = project_game(units_h, units_a, "nfl", M3_CFG, 0.6, 0.4, weather_mult=1.10)
        assert windy.total < clear.total
        assert boosted.total == clear.total          # >1 multipliers are clamped
        assert windy.margin == clear.margin          # weather never touches the spread

    def test_hfa_lands_on_margin(self):
        from mlb_value_bot.football.analysis.projections import project_game

        units = {"epa_dropback": 0.0, "rush_epa": 0.0, "plays_pg": 63.0,
                 "epa_dropback_allowed": 0.0, "rush_epa_allowed": 0.0}
        p = project_game(units, dict(units), "nfl", M3_CFG, 0.6, 0.4)
        assert p.margin == pytest.approx(1.5)        # equal teams -> HFA only
        assert p.total_raw == pytest.approx(45.0)    # 2 x base 22.5


class TestEvAndBlend:
    def test_blend_is_35_65(self):
        from mlb_value_bot.football.analysis.football_ev import blend_probability

        assert blend_probability(0.60, 0.50, M3_CFG) == pytest.approx(0.35 * 0.60 + 0.65 * 0.50)

    def test_ev_with_push(self):
        from mlb_value_bot.football.analysis.football_ev import ev_with_push

        # -110 both ways at fair 50/50, no push: EV = .5*(0.909) - .5
        assert ev_with_push(0.5, 0.0, -110) == pytest.approx(0.5 * (100 / 110) - 0.5, abs=1e-4)
        # Push mass shrinks the loss side, not the win side.
        assert ev_with_push(0.5, 0.05, -110) > ev_with_push(0.5, 0.0, -110)

    def test_market_view_consensus_and_devig(self):
        from mlb_value_bot.football.analysis.football_ev import market_view
        from mlb_value_bot.football.data.football_odds import (
            FootballGameOdds, SpreadQuote, TotalQuote)

        game = FootballGameOdds(
            event_id="e", commence_time="2026-09-13T17:00:00Z",
            home_name_raw="H", away_name_raw="A",
            spreads={"draftkings": SpreadQuote(-3.0, -110, -110),
                     "pinnacle": SpreadQuote(-3.5, -105, -115)},
            totals={"draftkings": TotalQuote(44.5, -110, -110)})
        v = market_view(game, "spread", M3_CFG, sigma=13.2)
        assert v.line == -3.0 and v.sharp_line == -3.5 and v.n_sharp_books == 1
        assert v.devig_p_a == pytest.approx(0.5, abs=1e-6)
        vt = market_view(game, "total", M3_CFG, sigma=10.0)
        assert vt.sharp_line is None and vt.anchor_mean == pytest.approx(44.5)


class TestAdjustedEvSingleHome:
    def test_sharp_support_fade_and_fragile(self):
        from mlb_value_bot.football.pipeline_football import _compute_adjusted_ev

        base = 0.05
        support, notes_s = _compute_adjusted_ev(base, +3.0, False, M3_CFG)
        fade, notes_f = _compute_adjusted_ev(base, -3.0, False, M3_CFG)
        neutral, _ = _compute_adjusted_ev(base, +1.0, False, M3_CFG)
        fragile, notes_fr = _compute_adjusted_ev(base, None, True, M3_CFG)
        assert support == pytest.approx(base + 0.010)
        assert fade == pytest.approx(base - 0.015)
        assert neutral == pytest.approx(base)         # inside the pp threshold
        assert fragile == pytest.approx(base - 0.010)
        assert any("sharp support" in n for n in notes_s)
        assert any("sharp fade" in n for n in notes_f)
        assert any("fragile" in n for n in notes_fr)

    def test_sharp_fade_single_homed_stability_never_adjusts(self):
        """stability.assess measures the gap and writes NOTES; it exposes no
        EV delta anywhere. The only application is _compute_adjusted_ev."""
        import inspect

        from mlb_value_bot.football.analysis import football_stability as stab

        s = stab.assess(None, games_min=10.0, epa_available=True,
                        weather_available=True, outdoor_total=False,
                        ol_proxy_only=False, config=M3_CFG)
        assert not hasattr(s, "ev_adjustment") and not hasattr(s, "adjusted_ev")
        # And the module never imports the adjusted-EV function.
        src = inspect.getsource(stab)
        assert "_compute_adjusted_ev" not in src


class TestConfidence:
    def test_g5_variance_haircut(self):
        from mlb_value_bot.football.analysis.football_confidence import confidence_for_pick

        kw = dict(edge_abs=50.0, completeness=1.0, market="spread",
                  explosive_involved=False, stability_label="stable", config=M3_CFG)
        nfl = confidence_for_pick(league="nfl", g5_involved=False, **kw)
        cfb_p4 = confidence_for_pick(league="cfb", g5_involved=False, **kw)
        cfb_g5 = confidence_for_pick(league="cfb", g5_involved=True, **kw)
        assert nfl.value > cfb_p4.value > cfb_g5.value   # same edge, less trust
        assert nfl.stake_mult == 1.0 and cfb_g5.stake_mult == 0.75

    def test_explosiveness_cuts_totals_confidence_only(self):
        from mlb_value_bot.football.analysis.football_confidence import confidence_for_pick

        kw = dict(edge_abs=50.0, completeness=1.0, league="nfl", g5_involved=False,
                  explosive_involved=True, stability_label="stable", config=M3_CFG)
        total = confidence_for_pick(market="total", **kw)
        spread = confidence_for_pick(market="spread", **kw)
        assert total.value < spread.value
        assert "explosive_variance_mult" in total.components
        assert "explosive_variance_mult" not in spread.components


class TestEvaluateMarket:
    """Constructed end-to-end for one market: fixture ScoredGame + MarketView
    + projection + weather -> a FootballPick with the gates applied."""

    def _fixture(self, *, total_line=40.0, model_epa=0.12, weather_available=True,
                 indoor=False, mult=1.0):
        from mlb_value_bot.football.analysis.football_ev import MarketView
        from mlb_value_bot.football.analysis.matchup import game_matchup
        from mlb_value_bot.football.analysis.ol_layer import OLGrade
        from mlb_value_bot.football.analysis.projections import project_game
        from mlb_value_bot.football.data.football_weather import FootballWeather
        from mlb_value_bot.football.pipeline_football import LeagueContext, ScoredGame

        home_u = _units(pass_off_pct=85.0, rush_off_pct=80.0)
        away_u = _units(pass_def_pct=15.0, rush_def_pct=20.0)
        matchup = game_matchup(home_u, away_u, M3_CFG)
        ol = OLGrade(0.0, 0.0, None, None, None, [])
        scored = ScoredGame("H", "A", matchup, ol, ol, home_u, away_u)
        units = {"epa_dropback": model_epa, "rush_epa": 0.05, "plays_pg": 63.0,
                 "epa_dropback_allowed": model_epa, "rush_epa_allowed": 0.05}
        projection = project_game(units, dict(units), "nfl", M3_CFG, 0.6, 0.4,
                                  weather_mult=mult)
        weather = FootballWeather(mult, weather_available, indoor, 60.0, 5.0, "test")
        view = MarketView("total", "draftkings", total_line, -110, -110, 0.5,
                          total_line, 0.5, total_line, 1)
        ctx = LeagueContext("nfl", 2026, 10, M3_CFG, pd.DataFrame(), pd.DataFrame(),
                            pd.DataFrame())
        return ctx, scored, view, projection, weather

    def test_value_pick_surfaces_over(self):
        from mlb_value_bot.football.pipeline_football import _evaluate_market

        # Strong offenses, generous line -> model total well above 40.
        ctx, scored, view, projection, weather = self._fixture(total_line=42.0)
        assert projection.total > 44.0
        pick = _evaluate_market(ctx, scored, view, projection, weather, False, False, 10.0)
        assert pick.side == "over"
        assert pick.market == "total"
        # But the divergence guard may hold it if too far from the anchor:
        if pick.hold_reason:
            assert "divergence" in pick.hold_reason
        else:
            assert pick.is_value is (pick.adjusted_ev > 0.03)

    def test_divergence_guard_holds_runaway_projection(self):
        from mlb_value_bot.football.pipeline_football import _evaluate_market

        ctx, scored, view, projection, weather = self._fixture(
            total_line=34.0, model_epa=0.20)
        assert projection.total - 34.0 > 6.0
        pick = _evaluate_market(ctx, scored, view, projection, weather, False, False, 10.0)
        assert pick.is_value is False
        assert "divergence guard" in pick.hold_reason

    def test_outdoor_total_without_weather_is_held(self):
        from mlb_value_bot.football.pipeline_football import _evaluate_market

        ctx, scored, view, projection, weather = self._fixture(weather_available=False)
        pick = _evaluate_market(ctx, scored, view, projection, weather, False, False, 10.0)
        assert pick.is_value is False
        assert pick.hold_reason is not None
        assert "weather" in pick.hold_reason or "divergence" in pick.hold_reason

    def test_reasoning_carries_the_full_transparent_trail(self):
        from mlb_value_bot.football.pipeline_football import _evaluate_market

        ctx, scored, view, projection, weather = self._fixture(total_line=44.0)
        pick = _evaluate_market(ctx, scored, view, projection, weather, False, False, 10.0)
        r = pick.reasoning
        assert r["model_tag"] == "matchup_v1"
        assert r["matchup"]["archetype"] in ("dual_edge", "strong_o_vs_weak_d")
        assert r["blend"]["market_blend"] == 0.35
        assert "adjusted_ev" in r and "stability" in r and "weather" in r


# =============================================================================
# M4 — grading, records, store
# =============================================================================

class TestPushGrading:
    def test_push_grading_spread_and_total(self):
        """Exactly on the number = push, every direction, both markets."""
        from mlb_value_bot.football.tracking.football_results import grade_pick

        # Home -3 pick, home wins by exactly 3 -> push.
        assert grade_pick("home", -3.0, 27, 24) == "push"
        assert grade_pick("home", -3.0, 28, 24) == "win"
        assert grade_pick("home", -3.0, 26, 24) == "loss"
        # Away +3 pick mirrors.
        assert grade_pick("away", 3.0, 27, 24) == "push"
        assert grade_pick("away", 3.0, 26, 24) == "win"
        assert grade_pick("away", 3.0, 28, 24) == "loss"
        # Totals: 44 exactly = push both ways.
        assert grade_pick("over", 44.0, 24, 20) == "push"
        assert grade_pick("under", 44.0, 24, 20) == "push"
        assert grade_pick("over", 44.0, 27, 20) == "win"
        assert grade_pick("under", 44.0, 21, 20) == "win"
        # Half lines can't push.
        assert grade_pick("home", -2.5, 27, 24) == "win"
        assert grade_pick("over", 44.5, 24, 20) == "loss"

    def test_profit_units(self):
        from mlb_value_bot.football.tracking.football_results import profit_for

        assert profit_for("win", 0.01, 1.909) == pytest.approx(0.00909, abs=1e-5)
        assert profit_for("loss", 0.01, 1.909) == pytest.approx(-0.01)
        assert profit_for("push", 0.01, 1.909) == 0.0
        assert profit_for("void", 0.02, 2.0) == 0.0


def _perf_df():
    """Mixed store: two model tags, two leagues, two markets, bets + analyses."""
    rows = []

    def add(league, market, result, *, tag="matchup_v1", is_value=1, side="home",
            clv=None, stake=0.01, pl=None):
        if pl is None:
            pl = {"win": stake * 0.909, "loss": -stake}.get(result, 0.0)
        rows.append(dict(league=league, market=market, result=result,
                         model_tag=tag, is_value=is_value, pick_side=side,
                         clv_pp=clv, flat_stake=stake, profit_loss=pl,
                         model_prob=0.55, created_at=f"2026-09-{len(rows)+1:02d}"))
    add("nfl", "spread", "win", clv=1.5)
    add("nfl", "spread", "loss", clv=-0.5)
    add("nfl", "spread", "push")
    add("nfl", "total", "win", side="over", clv=2.0)
    add("cfb", "spread", "win", clv=0.5)
    add("cfb", "total", "loss", side="under")
    add("nfl", "spread", "win", is_value=0)              # analysis: never counted
    add("nfl", "spread", "win", tag="other_model")       # other tag: never counted
    add("nfl", "spread", "void")
    add("nfl", "spread", "pending")
    return pd.DataFrame(rows)


class TestRecordFiltering:
    def test_record_filtering_by_tag_league_market(self):
        """THE GriffBet-record-bug regression: aggregates count ONLY is_value
        rows of the requested model_tag x league x market — analyses, other
        tags, other leagues, other markets never leak in."""
        from mlb_value_bot.football.tracking.football_performance import record

        df = _perf_df()
        nfl_spread = record(df, "matchup_v1", "nfl", "spread")
        assert (nfl_spread["wins"], nfl_spread["losses"], nfl_spread["pushes"]) == (1, 1, 1)
        assert nfl_spread["graded"] == 3            # voids excluded
        assert nfl_spread["voids"] == 1
        assert nfl_spread["pending"] == 1

        nfl_total = record(df, "matchup_v1", "nfl", "total")
        assert (nfl_total["wins"], nfl_total["losses"]) == (1, 0)
        cfb_all = record(df, "matchup_v1", "cfb")
        assert (cfb_all["wins"], cfb_all["losses"]) == (1, 1)
        # The all-bucket is still tag- and is_value-filtered.
        overall = record(df, "matchup_v1")
        assert overall["wins"] == 3                  # not 5: analysis + other tag out
        assert record(df, "other_model")["wins"] == 1

    def test_pushes_are_stake_neutral(self):
        from mlb_value_bot.football.tracking.football_performance import record

        df = _perf_df()
        r = record(df, "matchup_v1", "nfl", "spread")
        # P/L = one win (+0.00909) + one loss (-0.01); push contributes 0.
        assert r["flat_pl_units"] == pytest.approx(0.01 * 0.909 - 0.01, abs=1e-4)

    def test_clv_summary(self):
        from mlb_value_bot.football.tracking.football_performance import record

        r = record(_perf_df(), "matchup_v1", "nfl")
        assert r["clv_tracked"] == 3
        assert r["clv_positive"] == 2
        assert r["avg_clv_pp"] == pytest.approx((1.5 - 0.5 + 2.0) / 3, abs=1e-3)


class TestDistributionMonitor:
    def _df(self, overs: int, unders: int):
        rows = []
        for i in range(overs + unders):
            rows.append(dict(league="nfl", market="total", result="pending",
                             model_tag="matchup_v1", is_value=1,
                             pick_side="over" if i < overs else "under",
                             clv_pp=None, flat_stake=0.01, profit_loss=None,
                             model_prob=0.55, created_at=f"2026-09-01T{i:02d}:00"))
        return pd.DataFrame(rows)

    CFG = {"distribution_monitor": {"window": 50, "alert_share": 0.60, "min_picks": 25}}

    def test_alert_fires_past_60_40_either_way(self):
        from mlb_value_bot.football.tracking.football_performance import pick_distribution

        hot = pick_distribution(self._df(35, 15), "matchup_v1", self.CFG, "nfl")
        assert hot["over_share"] == 0.70 and hot["alert"] is True
        cold = pick_distribution(self._df(15, 35), "matchup_v1", self.CFG, "nfl")
        assert cold["alert"] is True                 # symmetric: under-bias too
        ok = pick_distribution(self._df(27, 23), "matchup_v1", self.CFG, "nfl")
        assert ok["alert"] is False

    def test_no_alert_below_min_sample(self):
        from mlb_value_bot.football.tracking.football_performance import pick_distribution

        small = pick_distribution(self._df(10, 2), "matchup_v1", self.CFG, "nfl")
        assert small["alert"] is False and small["n"] == 12

    def test_window_is_rolling(self):
        from mlb_value_bot.football.tracking.football_performance import pick_distribution

        # 60 picks: first 10 over, then 20 over + 30 under. Window=50 sees
        # only the last 50 (20 over / 30 under).
        df = pd.concat([self._df(10, 0), self._df(20, 30)], ignore_index=True)
        df["created_at"] = [f"2026-09-01T{i:04d}" for i in range(len(df))]
        d = pick_distribution(df, "matchup_v1", self.CFG, "nfl")
        assert d["n"] == 50 and d["overs"] == 20


class TestStoreUpsert:
    def _pick(self, market="spread", side="home", line=-3.0, is_value=True,
              odds=-110, sharp_p=0.52):
        from mlb_value_bot.football.pipeline_football import FootballPick

        return FootballPick(
            market=market, side=side, line=line, american_odds=odds,
            model_prob=0.55, market_prob=0.5, p_push=0.02, raw_ev=0.04,
            adjusted_ev=0.04, adjustments=[], confidence=70.0,
            tier="standard" if is_value else "pass",
            stake_pct=0.01 if is_value else 0.0, stability_label="stable",
            is_value=is_value, hold_reason=None,
            reasoning={"market": {"devig_p_a": 0.50, "sharp_devig_p_a": sharp_p,
                                  "sharp_line": line},
                       "matchup": {"home_edge": 20.0, "archetype": "neutral"},
                       "projection": {"margin": 4.0, "total": 44.0}})

    def _analysis(self, picks):
        from mlb_value_bot.football.pipeline_football import FootballGameAnalysis

        return FootballGameAnalysis(league="nfl", date="2026-09-13", week=1,
                                    game_id="2026_01_A_H", home="H", away="A",
                                    commence_time="2026-09-13T17:00:00Z", picks=picks)

    def test_opening_frozen_on_upsert(self, tmp_path, monkeypatch):
        """Second save of a committed bet keeps opening_*, refreshes the close
        + CLV — the CLV-by-re-pricing contract."""
        from mlb_value_bot.football.tracking import football_store as store

        monkeypatch.setattr(store, "FOOTBALL_DB_PATH", tmp_path / "fb.db")
        cfg = {"model_tag": "matchup_v1", "betting": {"paper_only": True}}

        a1 = self._analysis([self._pick(sharp_p=0.52)])
        store.save_slate([a1], cfg)
        # Later run: line moved to -3.5, sharps now 55% on home.
        a2 = self._analysis([self._pick(line=-3.5, odds=-115, sharp_p=0.55)])
        store.save_slate([a2], cfg)

        df = store.to_dataframe()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["opening_line"] == -3.0            # frozen
        assert row["opening_price"] == -110
        assert row["closing_line"] == -3.5            # refreshed
        assert row["closing_price"] == -115
        # CLV: sharp close 55% vs opening de-vig 50% = +5pp (we beat the close).
        assert row["clv_pp"] == pytest.approx(5.0)

    def test_analysis_rows_refresh_and_promote(self, tmp_path, monkeypatch):
        from mlb_value_bot.football.tracking import football_store as store

        monkeypatch.setattr(store, "FOOTBALL_DB_PATH", tmp_path / "fb.db")
        cfg = {"model_tag": "matchup_v1", "betting": {"paper_only": True}}

        store.save_slate([self._analysis([self._pick(is_value=False)])], cfg)
        df = store.to_dataframe()
        assert df.iloc[0]["is_value"] == 0
        assert pd.isna(df.iloc[0]["opening_line"])    # analyses set no opening

        # Promotion: THIS run's price becomes the opening reference.
        store.save_slate([self._analysis([self._pick(line=-2.5, is_value=True)])], cfg)
        row = store.to_dataframe().iloc[0]
        assert row["is_value"] == 1
        assert row["opening_line"] == -2.5

    def test_away_and_under_lines_stored_from_picked_side(self, tmp_path, monkeypatch):
        from mlb_value_bot.football.tracking import football_store as store

        monkeypatch.setattr(store, "FOOTBALL_DB_PATH", tmp_path / "fb.db")
        cfg = {"model_tag": "matchup_v1", "betting": {"paper_only": True}}
        away = self._pick(side="away", line=-3.0)     # home -3 -> away +3
        under = self._pick(market="total", side="under", line=44.0)
        store.save_slate([self._analysis([away, under])], cfg)
        df = store.to_dataframe().set_index("market")
        assert df.loc["spread", "line"] == 3.0
        assert df.loc["total", "line"] == 44.0


class TestOddsParsing:
    EVENT = {
        "id": "ev1", "commence_time": "2026-09-13T17:00:00Z",
        "home_team": "New York Giants", "away_team": "Dallas Cowboys",
        "bookmakers": [{
            "key": "draftkings",
            "markets": [
                {"key": "spreads", "outcomes": [
                    {"name": "New York Giants", "price": -108, "point": 3.5},
                    {"name": "Dallas Cowboys", "price": -112, "point": -3.5},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "price": -110, "point": 44.5},
                    {"name": "Under", "price": -110, "point": 44.5},
                ]},
            ],
        }],
    }

    def test_parse_event_spreads_and_totals(self):
        from mlb_value_bot.football.data.football_odds import _parse_event

        game = _parse_event(self.EVENT)
        assert game.home_name_raw == "New York Giants"   # raw names preserved
        dk_spread = game.spreads["draftkings"]
        assert dk_spread.home_line == 3.5
        assert dk_spread.home_price == -108
        assert dk_spread.away_price == -112
        dk_total = game.totals["draftkings"]
        assert dk_total.line == 44.5
        assert dk_total.over_price == -110

    def test_partial_markets_kept_and_missing_teams_dropped(self):
        from mlb_value_bot.football.data.football_odds import _parse_event

        no_totals = {**self.EVENT,
                     "bookmakers": [{"key": "dk", "markets": self.EVENT["bookmakers"][0]["markets"][:1]}]}
        game = _parse_event(no_totals)
        assert game.totals == {}
        assert _parse_event({"id": "x", "home_team": None, "away_team": "Y"}) is None
