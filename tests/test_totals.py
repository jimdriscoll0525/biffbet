"""Totals model tests -- no network, no pybaseball. Run via pytest or
`python -m mlb_value_bot.tests.test_totals`.

Covers the core math (distribution, market de-vig, weather) AND the integration
layer (EV/tier sizing, confidence, stability, the pipeline guards, and the
DB/grading/CLV path) -- all with synthetic inputs so it stays offline.
"""
from __future__ import annotations

import os
import tempfile

from mlb_value_bot.analysis.pitcher_metrics import PitcherProfile
from mlb_value_bot.analysis.team_metrics import TeamProfile
from mlb_value_bot.analysis import run_distribution as RD
from mlb_value_bot.analysis import totals_ev as TEV
from mlb_value_bot.analysis import totals_confidence as TC
from mlb_value_bot.analysis import totals_market as TM
from mlb_value_bot.analysis import totals_stability as TS
from mlb_value_bot.data import weather as WX
from mlb_value_bot.utils import load_config


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def _team(wrc=100.0, bp=4.0, pf=100.0):
    return TeamProfile(team="T", raw_winpct=0.5, games=60, offense_wrc_plus=wrc,
                       bullpen_fip=bp, park_factor=pf)


def _pitcher(xfip=4.0, ip=80.0, gs=14):
    return PitcherProfile(player_id=1, name="P", ip=ip, games_started=gs, xfip=xfip)


def _cfg():
    c = load_config()
    return c


class _LU:
    """Minimal LineupStatus stand-in (only `.status` is read)."""
    def __init__(self, status):
        self.status = status


class _Profiles:
    def __init__(self, home_pp, away_pp, home_tp, away_tp, home_lu=None, away_lu=None):
        self.home_pp, self.away_pp = home_pp, away_pp
        self.home_tp, self.away_tp = home_tp, away_tp
        self.home_lu, self.away_lu = home_lu, away_lu
        self.home_bp = self.away_bp = None


# --- expected runs / explicit bullpen innings --------------------------------
def test_explicit_bullpen_innings():
    cfg = _cfg()["totals"]
    si = cfg["starter_innings"]
    # Ace starter (3.0), bad bullpen (5.0). A SHORT outing must pull the staff
    # rate toward the bullpen (more relief innings).
    long_outing, _ = RD._staff_rate(3.0, 5.0, 7.0, 4.0, si)   # 7ip SP
    short_outing, _ = RD._staff_rate(3.0, 5.0, 3.0, 4.0, si)  # 3ip SP
    assert long_outing < short_outing                          # short outing -> worse (bullpen-heavy)
    # 7ip: 7/9*3 + 2/9*5 = 3.44 ; 3ip: 3/9*3 + 6/9*5 = 4.33
    assert approx(long_outing, 7 / 9 * 3 + 2 / 9 * 5, tol=1e-9)
    assert approx(short_outing, 3 / 9 * 3 + 6 / 9 * 5, tol=1e-9)
    # Missing SP rate -> league fallback, not a crash.
    fallback, note = RD._staff_rate(None, 5.0, None, 4.0, si)
    assert "league" in note and fallback is not None


# --- market recenter + NB probabilities --------------------------------------
def test_market_anchor_calibration():
    """The mean is solved so P(over) at the line equals the market's de-vigged
    P(over) -- the count distribution's right-skew creates no spurious edge."""
    tcfg = _cfg()["totals"]
    mu0 = RD._solve_mean_for_p_over(0.52, 8.5, tcfg)
    po, pu, pp = RD._p_over_for_mean(mu0, 8.5, tcfg)
    assert approx(po, 0.52, tol=1e-3) and approx(po + pu, 1.0, tol=1e-9) and pp == 0.0
    assert RD._solve_mean_for_p_over(0.60, 8.5, tcfg) > mu0     # higher target -> higher mean
    # Integer line carries an explicit push mass.
    _, _, push = RD._p_over_for_mean(9.0, 9.0, tcfg)
    assert push > 0.0


def test_run_tilt_moves_prob_and_clamps():
    cfg = _cfg()
    tcfg = cfg["totals"]
    wx = WX.WeatherEnv(1.0, True, 20, 0, 0, "open", "neutral")
    rd = RD.run_distribution(_team(), _team(), _pitcher(), _pitcher(), 8.5, 0.50, wx, cfg)
    # The tilt is measured against the ANCHOR mean (market-implied mean), never
    # the posted line: raw is a mean, the line is ~a median of a right-skewed
    # count distribution (tilting off the line was the systematic-OVER bug).
    anchor = RD._solve_mean_for_p_over(0.50, 8.5, tcfg)
    assert approx(rd.anchor_mean, anchor, tol=0.01)
    want_tilt = max(-1.5, min(1.5, rd.raw_model_total - anchor))
    assert approx(rd.expected_total, anchor + want_tilt, tol=0.01)
    assert (rd.p_over > 0.50) == (want_tilt > 0)               # prob follows the tilt sign
    # A wild raw build is clamped to anchor + max_tilt (1.5).
    hot = RD.run_distribution(_team(wrc=180, pf=130), _team(wrc=180, pf=130),
                              _pitcher(xfip=6.0), _pitcher(xfip=6.0), 8.0, 0.50, wx, cfg)
    assert hot.raw_model_total > 11.0
    assert approx(hot.expected_total - hot.anchor_mean, 1.5, tol=0.01)
    assert hot.p_over > rd.p_over                              # bigger over-tilt -> higher P(over)


def test_tilt_symmetry_no_over_bias():
    """Regression for the systematic-OVER bug: a raw build equal to the
    market-implied mean must reproduce the market's P(over) (zero tilt = agree
    with the market), and equal-and-opposite raw tilts must move P(over) by
    ~equal amounts in both directions. The old code tilted off the LINE, which
    re-imported the ~+0.8-run mean-vs-median skew offset and made 97% of value
    picks OVERs (172/27 lean on 199 production rows)."""
    tcfg = _cfg()["totals"]
    anchor = RD._solve_mean_for_p_over(0.50, 8.5, tcfg)
    assert anchor > 8.5                                        # implied mean sits ABOVE the line
    po0, pu0, _ = RD._p_over_for_mean(anchor, 8.5, tcfg)
    assert approx(po0, 0.50, tol=1e-3)                         # zero tilt = market
    po_up, _, _ = RD._p_over_for_mean(anchor + 1.0, 8.5, tcfg)
    po_dn, _, _ = RD._p_over_for_mean(anchor - 1.0, 8.5, tcfg)
    assert po_up > 0.5 > po_dn
    assert abs((po_up - 0.5) - (0.5 - po_dn)) < 0.02           # ~symmetric moves


def test_nb_is_overdispersed():
    # variance must exceed the mean (Poisson would set them equal).
    n, p = RD._nb_params(9.0, 22.0)
    mean = n * (1 - p) / p
    var = n * (1 - p) / (p * p)
    assert approx(mean, 9.0, tol=1e-6) and var > mean


# --- weather -----------------------------------------------------------------
def test_weather_wind_projection_and_degrade():
    # Wind FROM south (180) blows toward north (0); a park whose CF is north (0)
    # -> blowing straight OUT (+ component).
    out = WX._wind_out_component(20.0, 180.0, 0.0)
    assert out > 19                                            # ~full out
    inn = WX._wind_out_component(20.0, 0.0, 0.0)               # wind from N -> toward S, CF N -> IN
    assert inn < -19
    cross = WX._wind_out_component(20.0, 90.0, 0.0)
    assert abs(cross) < 1                                      # crosswind ~ 0
    # Fixed roof -> neutral but AVAILABLE (we correctly model no weather effect).
    cfg = _cfg()
    fr = WX.weather_env("Tampa Bay Rays", "2026-06-16", cfg)
    assert fr.multiplier == 1.0 and fr.available and fr.roof == "fixed_closed"
    # Unknown park -> neutral + NOT available (bet blind -> flag).
    unk = WX.weather_env("Nowhere FC", "2026-06-16", cfg)
    assert unk.multiplier == 1.0 and not unk.available


# --- totals market de-vig + consensus ----------------------------------------
def _book(key, line, over, under):
    return {"key": key, "markets": [{"key": "totals", "outcomes": [
        {"name": "Over", "price": over, "point": line},
        {"name": "Under", "price": under, "point": line}]}]}


def test_totals_devig_and_sharp_consensus():
    books = [
        _book("draftkings", 8.5, -110, -110),
        _book("pinnacle", 8.5, -108, -102),     # sharp leans over
        _book("fanduel", 8.5, -115, -105),
    ]
    intel = TM.compute_totals_market(books, "draftkings", ["pinnacle"], ["draftkings", "fanduel"], "power")
    assert intel.available and approx(intel.bet_line, 8.5)
    assert approx(intel.bet_devig_over, 0.5, tol=1e-3)        # -110/-110 ~ 50%
    assert intel.sharp_available and intel.sharp_devig_over > 0.5   # pinnacle over-lean
    assert intel.n_sharp == 1 and intel.n_square == 2
    # Sharp close prefers pinnacle.
    close = TM.sharp_totals_close(books, ["pinnacle", "betonlineag"], "power")
    assert close.book == "pinnacle" and approx(close.line, 8.5)


def test_totals_market_skips_books_without_totals():
    books = [{"key": "x", "markets": [{"key": "h2h", "outcomes": []}]}, _book("pinnacle", 9.0, -105, -115)]
    intel = TM.compute_totals_market(books, None, ["pinnacle"], [], "power")
    assert intel.n_total == 1 and approx(intel.sharp_line, 9.0)


# --- EV + tier sizing --------------------------------------------------------
def test_ou_sides_label_and_ev():
    evals = TEV.evaluate_ou_sides(0.55, -110, -110, "power")
    assert set(evals) == {"over", "under"}
    assert evals["over"].side == "over" and evals["under"].side == "under"
    # P(over)=0.55 at -110 (decimal 1.909) -> EV = 0.55*1.909-1 ~ +5.0%
    assert approx(evals["over"].ev_pct, 0.55 * (1 + 100 / 110) - 1, tol=1e-9)
    assert evals["over"].ev_pct > 0 and evals["under"].ev_pct < 0


def test_totals_tier_bands_and_guards():
    cfg = _cfg()
    # Bands off EV.
    assert TEV.classify_totals_tier(0.06, 80, "stable", cfg)[0] == "standard"
    assert TEV.classify_totals_tier(0.10, 80, "stable", cfg)[0] == "strong"
    # Never Strong on fragile -> downgraded to standard.
    assert TEV.classify_totals_tier(0.10, 80, "fragile", cfg)[0] == "standard"
    # Low confidence downgrades one step.
    assert TEV.classify_totals_tier(0.06, 50, "stable", cfg)[0] == "small"
    # Selection decoupled from sizing: a raw pick whose adj EV is tiny still
    # floors at `small`, never pass.
    assert TEV.classify_totals_tier(0.01, 90, "stable", cfg, is_raw_pick=True)[0] == "small"
    assert TEV.classify_totals_tier(0.01, 90, "stable", cfg, is_raw_pick=False)[0] == "pass"


# --- confidence --------------------------------------------------------------
def test_totals_confidence_rewards_complete_data():
    cfg = _cfg()
    wx_ok = WX.WeatherEnv(1.0, True, 20, 0, 0, "open", "ok")
    wx_no = WX.WeatherEnv(1.0, False, None, None, None, "open", "unavailable")
    rd = RD.run_distribution(_team(), _team(), _pitcher(), _pitcher(), 8.5, 0.5, wx_ok, cfg)
    full = _Profiles(_pitcher(), _pitcher(), _team(), _team(), _LU("confirmed"), _LU("confirmed"))
    bare = _Profiles(PitcherProfile(None, None), PitcherProfile(None, None),
                     TeamProfile(team="T", offense_wrc_plus=None, bullpen_fip=None),
                     TeamProfile(team="T", offense_wrc_plus=None, bullpen_fip=None))
    hi = TC.compute_totals_confidence(full, wx_ok, rd, full.home_lu, full.away_lu, cfg)
    lo = TC.compute_totals_confidence(bare, wx_no, rd, None, None, cfg)
    assert hi > lo and hi > 70 and lo < 40


# --- stability ---------------------------------------------------------------
def test_totals_stability_archetypes():
    cfg = _cfg()
    wx_ok = WX.WeatherEnv(1.0, True, 20, 0, 0, "open", "ok")
    wx_no = WX.WeatherEnv(1.0, False, None, None, None, "open", "unavailable")
    # Neutral build at a matching line -> ~0 raw-vs-market gap, so the only thing
    # deciding the label is the data quality (rated starters / weather / lineups).
    rd = RD.run_distribution(_team(), _team(), _pitcher(), _pitcher(), 9.0, 0.5, wx_ok, cfg)
    # Two rated starters, weather, BOTH lineups confirmed, no gap -> STABLE.
    stable_prof = _Profiles(_pitcher(), _pitcher(), _team(), _team(),
                            _LU("confirmed"), _LU("confirmed"))
    s = TS.classify_totals_stability(stable_prof, wx_ok, rd, stable_prof.home_lu, stable_prof.away_lu, 0.0, cfg)
    assert s.label == "stable" and not s.hard_fragile_signals
    # Missing weather -> FRAGILE (hard signal), even with everything else solid.
    f1 = TS.classify_totals_stability(stable_prof, wx_no, rd, stable_prof.home_lu, stable_prof.away_lu, 0.0, cfg)
    assert f1.label == "fragile" and any("weather" in x for x in f1.hard_fragile_signals)
    # TBD starter -> FRAGILE.
    tbd = _Profiles(PitcherProfile(None, None), _pitcher(), _team(), _team(), _LU("confirmed"), _LU("confirmed"))
    f2 = TS.classify_totals_stability(tbd, wx_ok, rd, tbd.home_lu, tbd.away_lu, 0.0, cfg)
    assert f2.label == "fragile"
    # Projected (not missing) lineup is timing, not fragility -> MODERATE.
    proj = _Profiles(_pitcher(), _pitcher(), _team(), _team(), _LU("projected"), _LU("projected"))
    m = TS.classify_totals_stability(proj, wx_ok, rd, proj.home_lu, proj.away_lu, 0.0, cfg)
    assert m.label == "moderate"


# --- pipeline guards (synthetic, no network) ---------------------------------
def _sched():
    from mlb_value_bot.data.mlb_client import ProbablePitcher, ScheduledGame
    return ScheduledGame(
        game_id=901, game_date="2026-06-16", status="Scheduled",
        home_team="Boston Red Sox", away_team="New York Yankees",
        home_pitcher=ProbablePitcher(1, "A"), away_pitcher=ProbablePitcher(2, "B"),
        venue="Fenway", game_datetime="2026-06-16T23:00:00Z", home_team_id=111, away_team_id=147)


def _odds(line, over, under, sharp_over=None, sharp_under=None):
    from mlb_value_bot.data.odds_client import GameOdds

    def bk(key, ov, un):
        return {"key": key, "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "price": ov, "point": line}, {"name": "Under", "price": un, "point": line}]}]}
    books = [bk("draftkings", over, under), bk("fanduel", over, under)]
    if sharp_over is not None:
        books.append(bk("pinnacle", sharp_over, sharp_under))
    return GameOdds(event_id="e", commence_time="x", home_team="Boston Red Sox",
                    away_team="New York Yankees", all_books=books)


def _wx(date="2026-06-16", available=True):
    WX._WEATHER_CACHE[(date, "Boston Red Sox")] = WX.WeatherEnv(
        1.0, available, 20, 0, 0, "open", "test").__dict__
    return WX.weather_env("Boston Red Sox", date, _cfg())


def test_pipeline_divergence_skip():
    from mlb_value_bot.pipeline_totals import evaluate_totals_game
    cfg = _cfg()
    # Two huge offenses + bad pens vs a LOW market line -> raw total far above
    # the line -> divergence guard skips.
    prof = _Profiles(_pitcher(xfip=6.0), _pitcher(xfip=6.0),
                     _team(wrc=160, bp=5.5, pf=120), _team(wrc=160, bp=5.5, pf=120))
    a = evaluate_totals_game(_sched(), _odds(7.0, -110, -110), prof, _wx(), cfg)
    assert a.skipped_reason and "diverge" in a.skipped_reason


def test_pipeline_market_bounds_and_no_market():
    from mlb_value_bot.pipeline_totals import evaluate_totals_game
    cfg = _cfg()
    prof = _Profiles(_pitcher(), _pitcher(), _team(), _team())
    # Implausible posted total.
    a = evaluate_totals_game(_sched(), _odds(2.0, -110, -110), prof, _wx(), cfg)
    assert a.skipped_reason and "implausible market total" in a.skipped_reason
    # No totals market at all (h2h-only book).
    from mlb_value_bot.data.odds_client import GameOdds
    h2h = GameOdds(event_id="e", commence_time="x", home_team="Boston Red Sox",
                   away_team="New York Yankees",
                   all_books=[{"key": "draftkings", "markets": [{"key": "h2h", "outcomes": []}]}])
    a2 = evaluate_totals_game(_sched(), h2h, prof, _wx(), cfg)
    assert a2.skipped_reason and "no totals market" in a2.skipped_reason


def test_pipeline_weather_hold():
    from mlb_value_bot.pipeline_totals import evaluate_totals_game
    cfg = _cfg()
    prof = _Profiles(_pitcher(), _pitcher(), _team(), _team())
    # Weather unavailable -> pick is computed but HELD to analysis-only + flagged.
    a = evaluate_totals_game(_sched(), _odds(8.5, -110, -110), prof, _wx(available=False), cfg)
    if a.skipped_reason is None:        # only assert the hold when it wasn't otherwise skipped
        assert a.weather_held and not a.is_value(0.0)
        assert any("weather" in f for f in a.flags)


# --- DB / grading / CLV ------------------------------------------------------
def _totals_db():
    """Point the totals tracking table at a fresh temp DB; return the module."""
    from mlb_value_bot.tracking import totals_recommendations as T
    path = os.path.join(tempfile.gettempdir(), f"totals_test_{os.getpid()}_{id(object())}.db")
    if os.path.exists(path):
        os.remove(path)
    T.DB_PATH = path
    return T, path


def _rec(T, **kw):
    base = dict(date="2026-06-16", game_id=601, home_team="Boston Red Sox", away_team="New York Yankees",
                pick_side="under", bet_odds=105, decimal_odds=2.05, model_prob=0.55, market_prob_devigged=0.49,
                ev_pct=0.06, kelly_stake=0.008, confidence=78.0, market_total=8.5, over_odds=-115, under_odds=105,
                model_p_over=0.45, market_devig_over=0.51, blended_p_over=0.47, tier="standard", stability="stable",
                raw_model_total=8.2, expected_total=8.4, paper=True, opening_devig_p_side=0.49,
                best_close_line=8.5, best_close_price=110, reasoning={"market_type": "totals"}, is_value=True)
    base.update(kw)
    return T.TotalsRecommendationRecord(**base)


def test_db_clv_freeze_and_refresh():
    T, path = _totals_db()
    sc1 = TM.SharpTotalsLine("pinnacle", 8.5, -108, -102, 0.49)   # P(over) 0.49 -> under 0.51
    T.upsert_totals_recommendation(_rec(T, sharp_close=sc1))
    r = T.to_dataframe().iloc[0]
    # under: opening market P(under)=1-0.51=0.49; close P(under)=1-0.49=0.51 -> +2.0pp
    assert approx(float(r["clv_pp"]), 2.0, tol=0.05) and approx(float(r["opening_devig_p_side"]), 0.49)
    # Committed bet: re-price with a moved sharp close + different bet price; the
    # opening reference + price must stay FROZEN, only the close + CLV move.
    sc2 = TM.SharpTotalsLine("pinnacle", 8.5, -102, -108, 0.45)   # under 0.55
    T.upsert_totals_recommendation(_rec(T, bet_odds=120, opening_devig_p_side=0.40, sharp_close=sc2))
    r = T.to_dataframe().iloc[0]
    assert int(r["bet_odds"]) == 105 and approx(float(r["opening_devig_p_side"]), 0.49)
    assert approx(float(r["clv_pp"]), 6.0, tol=0.05)
    try:
        os.remove(path)
    except OSError:
        pass


def test_grade_totals_over_under_push():
    from mlb_value_bot.data.mlb_client import GameResult
    from mlb_value_bot.tracking import results as R
    T, path = _totals_db()

    T.upsert_totals_recommendation(_rec(T, game_id=611, pick_side="under", market_total=8.5,
                                        sharp_close=TM.SharpTotalsLine("pinnacle", 8.5, -110, -110, 0.5)))
    T.upsert_totals_recommendation(_rec(T, game_id=612, pick_side="over", market_total=9.0,
                                        bet_odds=-110, decimal_odds=1.909,
                                        sharp_close=TM.SharpTotalsLine("pinnacle", 9.0, -110, -110, 0.5)))

    class FakeMLB:
        def get_results(self, d):
            return [
                GameResult(611, "Final", "Boston Red Sox", "New York Yankees", 3, 4),   # total 7 < 8.5 -> under WIN
                GameResult(612, "Final", "Boston Red Sox", "New York Yankees", 5, 4),   # total 9 == 9.0 -> PUSH
            ]
    summary = R.grade_totals_date("2026-06-16", mlb_client=FakeMLB())
    df = T.to_dataframe().set_index("game_id")
    assert df.loc[611, "result"] == "win" and df.loc[611, "profit_loss"] > 0
    assert df.loc[612, "result"] == "push" and approx(float(df.loc[612, "profit_loss"]), 0.0)
    assert summary.wins == 1 and summary.pushes == 1 and summary.voids == 0
    try:
        os.remove(path)
    except OSError:
        pass


def test_totals_performance_record_counts_pushes_separately():
    """Regression: the totals W-L-P record breaks pushes out — a push is never a
    loss, never a win, and is excluded from hit-rate/flat-ROI denominators."""
    T, path = _totals_db()
    sc = TM.SharpTotalsLine("pinnacle", 8.5, -110, -110, 0.5)
    outcomes = {621: ("win", 0.005), 622: ("loss", -0.008), 623: ("loss", -0.008),
                624: ("push", 0.0), 625: (None, None)}       # 625 stays pending
    for gid in outcomes:
        T.upsert_totals_recommendation(_rec(T, game_id=gid, sharp_close=sc))
    ids = T.to_dataframe().set_index("game_id")["id"]
    for gid, (res, pl) in outcomes.items():
        if res is not None:
            T.update_result(int(ids.loc[gid]), res, pl)

    from mlb_value_bot.tracking import totals_performance as TP
    rep = TP.compute_totals_performance()
    o = rep.overall
    assert o["bets"] == 5
    assert o["wins"] == 1 and o["losses"] == 2 and o["pushes"] == 1
    assert o["settled"] == 3                      # W+L only; push is stake-neutral
    assert approx(o["hit_rate"], 1 / 3)           # push not in the denominator
    try:
        os.remove(path)
    except OSError:
        pass


# --- production-card fixture: mean provenance + weather-applied-once ---------
def test_card_fixture_mean_provenance_and_weather_applied_once():
    """Regression fixture from the PHI card audited 2026-07-16 (line 9.5,
    market P(over) 50.5%). Pins the pipeline order of operations so the site's
    run breakdown always reconciles:

      * park + weather are applied INSIDE the per-team runs (exactly once) --
        the breakdown's multiplier rows are display-only;
      * the distribution mean is anchor_mean + clamped tilt, NOT
        sum(runs) x park x weather (the '8.57 x 1.02 = 8.7' reading of this
        card was a numerical coincidence);
      * variance scales off the FINAL mean.
    """
    cfg = _cfg()
    tcfg = cfg["totals"]

    home_tp = _team(wrc=94.0, bp=4.52, pf=102.0)
    away_tp = _team(wrc=90.0, bp=4.08, pf=100.0)
    home_pp = _pitcher(xfip=4.08, ip=106.0, gs=20)   # 5.3 ip/start exactly
    away_pp = _pitcher(xfip=4.26, ip=106.0, gs=20)

    class _Weather:
        multiplier, note, available = 0.961, "28C, wind 10km/h in 9", True

    er = RD.expected_runs(home_tp, away_tp, home_pp, away_pp, _Weather(), cfg)
    # Per-team runs already include park (x1.020) AND weather (x0.961).
    assert approx(er["home_rs"], 4.34, tol=0.01)
    assert approx(er["away_rs"], 4.23, tol=0.01)
    assert approx(er["raw_total"], 8.57, tol=0.02)
    # Weather applied exactly once: removing it rescales raw by the multiplier.
    er_nw = RD.expected_runs(home_tp, away_tp, home_pp, away_pp, None, cfg)
    assert approx(er["raw_total"] / er_nw["raw_total"], 0.961, tol=0.002)

    rd = RD.run_distribution(home_tp, away_tp, home_pp, away_pp, 9.5, 0.505,
                             _Weather(), cfg)
    # Mean provenance: anchor 10.23 (market-implied MEAN, above the line by the
    # NB right-skew), raw-anchor = -1.66 clamped to -1.50, mean 8.73.
    assert approx(rd.anchor_mean, 10.23, tol=0.02)
    assert approx(rd.tilt, -float(tcfg["max_tilt_runs"]), tol=1e-9)  # clamp engaged
    assert approx(rd.expected_total, 8.73, tol=0.02)
    assert approx(rd.expected_total, rd.anchor_mean + rd.tilt, tol=0.011)
    # Variance is scaled off the FINAL mean, consistently with the config ratio.
    ratio = tcfg["league"]["total_variance"] / tcfg["league"]["avg_total"]
    assert approx(rd.variance, rd.expected_total * ratio, tol=0.02)
    assert approx(rd.variance, 21.81, tol=0.05)
    assert approx(rd.p_over, 0.380, tol=0.005)

    # The reasoning JSON must carry the provenance fields the site renders.
    from mlb_value_bot.pipeline_totals import TotalsAnalysis
    a = TotalsAnalysis(game_id=1, game_date="2026-07-15", home_team="H",
                       away_team="A", status="S", home_pitcher=None,
                       away_pitcher=None, rd=rd)
    r = a.reasoning()["run_distribution"]
    assert approx(r["anchor_mean"], rd.anchor_mean, tol=1e-9)
    assert approx(r["tilt"], rd.tilt, tol=1e-9)
    assert approx(r["expected_total"], rd.expected_total, tol=1e-9)


# --- weather fetch: transient failures retry instead of holding the pick -----
def test_weather_fetch_retries_transient_failures():
    """2026-07-19: six of sixteen parks failed the single un-retried
    Open-Meteo fetch in one run, holding a qualified totals pick to
    analysis-only. Two connection flakes must now recover on retry."""
    import requests

    calls = {"n": 0}

    class _Resp:
        status_code = 200

        def json(self):
            return {"current": {"temperature_2m": 25.0, "wind_speed_10m": 5.0,
                                "wind_direction_10m": 100.0}}

    def flaky_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("transient")
        return _Resp()

    orig = requests.get
    requests.get = flaky_get
    try:
        cur = WX._fetch_open_meteo(39.9, -75.2, 2.0)
    finally:
        requests.get = orig
    assert calls["n"] == 3
    assert cur is not None and cur["temperature_2m"] == 25.0

    # A hard failure (all attempts exhausted) still degrades to None, never raises.
    def dead_get(url, params=None, timeout=None):
        raise requests.ConnectionError("down")

    requests.get = dead_get
    try:
        assert WX._fetch_open_meteo(39.9, -75.2, 2.0) is None
    finally:
        requests.get = orig


# --- results CLI: totals must grade even when the ML ledger is settled -------
def test_results_grades_totals_even_with_no_open_ml_dates():
    """2026-07-17 production bug: `results` early-returned when the MONEYLINE
    sweep found no open dates, so the TOTALS sweep below it never ran and a
    pending totals pick (PHI U9.5 on 7/16) sat ungraded through five
    successful pipeline runs while the ML ledger happened to be fully settled.
    """
    from click.testing import CliRunner
    from mlb_value_bot import cli as cli_mod
    from mlb_value_bot.tracking import results as results_mod

    called = {"totals": False}
    orig_ml = results_mod.grade_all_open
    orig_totals = results_mod.grade_all_open_totals
    results_mod.grade_all_open = lambda before, **kw: []          # ML: nothing open

    def _fake_totals(before, **kw):
        called["totals"] = True
        return []

    results_mod.grade_all_open_totals = _fake_totals
    try:
        result = CliRunner().invoke(cli_mod.cli, ["results"])
        assert result.exit_code == 0, result.output
        assert called["totals"], (
            "totals sweep must run even when no moneyline bets are open"
        )
    finally:
        results_mod.grade_all_open = orig_ml
        results_mod.grade_all_open_totals = orig_totals


def _run_all():
    import inspect
    import sys
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in inspect.getmembers(mod, inspect.isfunction) if n.startswith("test_")]
    for n, f in fns:
        f()
        print(f"  PASS  {n}")
    print(f"\n{len(fns)}/{len(fns)} totals tests passed.")


if __name__ == "__main__":
    _run_all()
