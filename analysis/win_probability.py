"""Transparent, tunable win-probability model.

Philosophy (per spec): NOT a black box. We start from a base rate and apply a
series of additive probability deltas, each with a weight you control in
config.yaml. Every component is reported back so you can see *why* the model
favors a side.

    home_win_prob = base_wp
                  + w_starter     * starter_delta
                  + w_bullpen     * bullpen_delta
                  + w_park        * park_delta
                  + w_home_field  * home_field_delta
                  + w_form        * form_delta
    (clamped to [prob_floor, prob_ceiling])

Where:
  base_wp        log5 of the two teams' regressed season win% (neutral site)
  starter_delta  log5 of the two SPs' win-equivalent ratings, minus 0.5
  bullpen_delta  relief-FIP differential converted to win%
  park_delta     ballpark run environment x offense gap (small, totals-leaning)
  home_field     constant home boost
  form_delta     recent pitcher form (last ~5 starts), via Statcast

Each delta is a signed number where POSITIVE favors the HOME team.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from mlb_value_bot.analysis.pitcher_metrics import PitcherProfile
from mlb_value_bot.analysis.team_metrics import TeamProfile
from mlb_value_bot.utils import get_logger, load_config

log = get_logger("analysis.win_probability")


# --- Small math helpers ------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def log5(p_a: float, p_b: float) -> float:
    """Probability that A beats B given each team's win% vs an average team.

    log5(A, B) = (A - A*B) / (A + B - 2*A*B). Returns 0.5 when both are equal.
    """
    p_a = clamp(p_a, 1e-6, 1 - 1e-6)
    p_b = clamp(p_b, 1e-6, 1 - 1e-6)
    denom = p_a + p_b - 2 * p_a * p_b
    if denom == 0:
        return 0.5
    return (p_a - p_a * p_b) / denom


def regress_winpct(winpct: float, games: float, k: float) -> float:
    """Regress a team's win% toward .500 using k 'phantom' .500 games."""
    if games <= 0:
        return 0.5
    return (winpct * games + 0.5 * k) / (games + k)


def pitcher_win_equiv(rate: float | None, lg_avg: float, run_to_wp: float) -> float | None:
    """Convert a run-prevention rate (xFIP/SIERA) to a win% vs league average.

    Lower (better) rate -> higher win%. Centered at .500 for a league-average
    pitcher. Clamped to a believable [0.30, 0.70] single-pitcher range.
    """
    if rate is None:
        return None
    wp = 0.5 + (lg_avg - rate) * run_to_wp
    return clamp(wp, 0.30, 0.70)


# --- Result containers -------------------------------------------------------
@dataclass
class Component:
    """One model factor and its contribution to the home win probability."""

    name: str
    raw_delta: float       # signed delta before weighting (+ favors home)
    weight: float
    weighted_delta: float  # raw_delta * weight (what's actually added)
    note: str = ""
    available: bool = True
    # Signal-quality flag. True when this component's contribution is
    # disproportionately driven by noisy short-window inputs (e.g. 14d
    # xwOBAcon is ~50 batted balls, mostly variance). The pick-level
    # stability classifier (Step 3) treats a fragile-flagged dominant
    # component as evidence the whole edge is fragile.
    fragile: bool = False


@dataclass
class WinProbabilityResult:
    home_team: str
    away_team: str
    base_prob: float
    home_win_prob: float
    away_win_prob: float
    components: list[Component] = field(default_factory=list)
    confidence: float = 0.0  # filled in by compute_confidence()

    @property
    def favored_side(self) -> str:
        return "home" if self.home_win_prob >= 0.5 else "away"

    def reasoning(self) -> dict:
        """JSON-able breakdown stored as `reasoning_json` in the DB."""
        return {
            "base_prob": round(self.base_prob, 4),
            "home_win_prob": round(self.home_win_prob, 4),
            "away_win_prob": round(self.away_win_prob, 4),
            "favored_side": self.favored_side,
            "components": [asdict(c) for c in self.components],
        }


# --- The model ---------------------------------------------------------------
def compute_win_probability(
    home_team: TeamProfile,
    away_team: TeamProfile,
    home_pitcher: PitcherProfile,
    away_pitcher: PitcherProfile,
    config: dict | None = None,
    home_bullpen_status: "BullpenStatus | None" = None,
    away_bullpen_status: "BullpenStatus | None" = None,
    home_lineup_status: "LineupStatus | None" = None,
    away_lineup_status: "LineupStatus | None" = None,
) -> WinProbabilityResult:
    """Run the model and return the home win probability with full breakdown.

    `home_bullpen_status` / `away_bullpen_status` are the optional fatigue
    snapshots from data.bullpen_status. When provided, an additional
    `bullpen_fatigue` component is added on top of the existing season-FIP
    bullpen component. When missing, the component contributes 0 with the
    note "data unavailable" -- existing behavior is preserved exactly.
    """
    config = config or load_config()
    m = config["model"]
    lg = config["league"]
    weights = m["weights"]
    prefer = m.get("pitcher_stat", "xfip")
    lg_pitch = float(lg.get("avg_xfip", 4.0) if prefer == "xfip" else lg.get("avg_siera", 4.0))

    # 1) BASE RATE — log5 of regressed team win%.
    k = float(m.get("team_regression_games", 30))
    home_wp = regress_winpct(home_team.raw_winpct, home_team.games, k)
    away_wp = regress_winpct(away_team.raw_winpct, away_team.games, k)
    base = log5(home_wp, away_wp)

    components: list[Component] = []

    # 2) STARTER — log5 of pitcher win-equivalents.
    h_rate = home_pitcher.primary_rate(prefer)
    a_rate = away_pitcher.primary_rate(prefer)
    h_pwp = pitcher_win_equiv(h_rate, lg_pitch, float(m["pitcher_run_to_winpct"]))
    a_pwp = pitcher_win_equiv(a_rate, lg_pitch, float(m["pitcher_run_to_winpct"]))
    if h_pwp is not None and a_pwp is not None:
        starter_cap = float(m.get("starter_clamp", 0.15))
        starter_delta = clamp(log5(h_pwp, a_pwp) - 0.5, -starter_cap, starter_cap)
        h_src = home_pitcher.primary_rate_source(prefer) or "?"
        a_src = away_pitcher.primary_rate_source(prefer) or "?"
        note = f"rate H={h_rate:.2f}({h_src}) A={a_rate:.2f}({a_src})"
        starter_avail = True
    else:
        starter_delta = 0.0
        note = "missing pitcher rate stat(s)"
        starter_avail = False
    components.append(_mk("starter", starter_delta, weights["starter"], note, starter_avail))

    # 3) BULLPEN — relief FIP differential -> win%.
    if home_team.bullpen_fip is not None and away_team.bullpen_fip is not None:
        diff = away_team.bullpen_fip - home_team.bullpen_fip  # + favors home
        bullpen_delta = clamp(diff * float(m["bullpen_run_to_winpct"]), -0.10, 0.10)
        note = f"relief FIP H={home_team.bullpen_fip:.2f} A={away_team.bullpen_fip:.2f}"
        bp_avail = True
    else:
        bullpen_delta = 0.0
        note = "missing bullpen FIP"
        bp_avail = False
    components.append(_mk("bullpen", bullpen_delta, weights["bullpen"], note, bp_avail))

    # 3b) BULLPEN FATIGUE — additive tilt on top of (3) using today's
    # availability of each team's leverage arms (see data/bullpen_status.py).
    # Each "down" leverage arm contributes a configurable per-arm tilt; the
    # net is (away_down - home_down) * scale, clamped tight (default +/-0.03).
    # If either side's status is unavailable we contribute 0 and label it --
    # never crash the slate on a transient API hiccup.
    bp_scale = float(m.get("bullpen_fatigue_scale", 0.012))
    bp_clamp = float(m.get("bullpen_fatigue_clamp", 0.03))
    bp_weight = float(weights.get("bullpen_fatigue", 1.0))
    if (
        home_bullpen_status is not None and home_bullpen_status.available
        and away_bullpen_status is not None and away_bullpen_status.available
    ):
        h_down = home_bullpen_status.leverage_unavailable
        a_down = away_bullpen_status.leverage_unavailable
        # + favors home (away more tired).
        bp_fatigue_delta = clamp((a_down - h_down) * bp_scale, -bp_clamp, bp_clamp)
        bp_fatigue_note = (
            f"H {home_bullpen_status.short_label()}; "
            f"A {away_bullpen_status.short_label()}"
        )
        bp_fatigue_avail = True
    else:
        bp_fatigue_delta = 0.0
        bp_fatigue_note = "bullpen status unavailable"
        bp_fatigue_avail = False
    components.append(_mk("bullpen_fatigue", bp_fatigue_delta, bp_weight, bp_fatigue_note, bp_fatigue_avail))

    # 3c) LINEUP — confirmed-vs-projected lineup status and key-bats out.
    # Tilts toward whichever team has FEWER key bats missing from today's
    # confirmed lineup. Returns 0 with note "lineup projected" when either
    # side isn't confirmed yet; the projected-state confidence penalty is
    # applied separately in compute_data_confidence / compute_confidence.
    lu_scale = float(m.get("lineup_per_missing_bat_scale", 0.005))
    lu_clamp = float(m.get("lineup_clamp", 0.02))
    lu_weight = float(weights.get("lineup", 1.0))
    if (
        home_lineup_status is not None and home_lineup_status.is_confirmed
        and away_lineup_status is not None and away_lineup_status.is_confirmed
    ):
        h_missing = home_lineup_status.missing_count
        a_missing = away_lineup_status.missing_count
        lu_delta = clamp((a_missing - h_missing) * lu_scale, -lu_clamp, lu_clamp)
        lu_note = (
            f"H {home_lineup_status.short_label()}; "
            f"A {away_lineup_status.short_label()}"
        )
        lu_avail = True
    else:
        lu_delta = 0.0
        if home_lineup_status is not None and away_lineup_status is not None:
            lu_note = f"H {home_lineup_status.short_label()}; A {away_lineup_status.short_label()}"
        else:
            lu_note = "lineup status unavailable"
        lu_avail = False
    components.append(_mk("lineup", lu_delta, lu_weight, lu_note, lu_avail))

    # 4) PARK — run environment x offense gap. Small, mostly a totals factor.
    off_home = home_team.offense_wrc_plus
    off_away = away_team.offense_wrc_plus
    if off_home is not None and off_away is not None:
        off_diff = (off_home - off_away) / 100.0       # +0.10 => home 10% better
        park_dev = (home_team.park_factor - 100.0) / 100.0
        park_delta = clamp(off_diff * park_dev * 0.5, -0.05, 0.05)
        note = f"PF={home_team.park_factor:.0f} wRC+ H={off_home:.0f} A={off_away:.0f}"
        park_avail = True
    else:
        park_delta = 0.0
        note = "missing team offense (wRC+)"
        park_avail = False
    components.append(_mk("park", park_delta, weights["park"], note, park_avail))

    # 5) HOME FIELD — park-specific boost to the home side (Coors > the rest).
    default_hfa = float(m.get("default_park_hfa", m.get("home_field_advantage", 0.035)))
    hfa = float(m.get("park_hfa", {}).get(home_team.team, default_hfa))
    components.append(_mk("home_field", hfa, weights["home_field"], f"+{hfa:.3f} home", True))

    # 6) RECENT FORM — pitcher last ~5 starts via Statcast (xwOBA-on-contact,
    # fallback CSW%). NOTE: rolling team-offense form (last 14d) is not yet
    # pulled; this component currently reflects recent PITCHING form only.
    # The fragility flag flows to Step 3's pick-level stability classifier.
    form_delta, form_note, form_avail, form_fragile = _form_delta(
        home_pitcher, away_pitcher,
        float(m.get("form_scale", 0.5)), float(m.get("form_recent_weight", 0.6)),
        config=config,
    )
    components.append(_mk("form", form_delta, weights["form"], form_note, form_avail,
                          fragile=form_fragile))

    # Assemble.
    home_prob = base + sum(c.weighted_delta for c in components)
    home_prob = clamp(home_prob, float(m["prob_floor"]), float(m["prob_ceiling"]))

    result = WinProbabilityResult(
        home_team=home_team.team,
        away_team=away_team.team,
        base_prob=base,
        home_win_prob=home_prob,
        away_win_prob=1.0 - home_prob,
        components=components,
    )
    log.debug("WP %s vs %s -> home %.3f (base %.3f)", home_team.team, away_team.team, home_prob, base)
    return result


def _mk(name: str, raw: float, weight: float, note: str, available: bool,
        fragile: bool = False) -> Component:
    weight = float(weight)
    return Component(name=name, raw_delta=raw, weight=weight, weighted_delta=raw * weight,
                     note=note, available=available, fragile=fragile)


def _regress_recent(recent: float | None, season: float | None, w_recent: float) -> float | None:
    """Regress a recent rate toward its season value to damp streak noise:
    w_recent*recent + (1-w_recent)*season. Falls back to recent if season is absent."""
    if recent is None:
        return None
    if season is None:
        return recent
    return w_recent * recent + (1.0 - w_recent) * season


@dataclass
class _FormBlend:
    """Output of `_blended_form_xwoba` -- weighted-window xwOBAcon plus the
    raw per-window values + the share of total weight that came from 14d.
    Returned in one shape so the form-delta logic can compare windows
    directly (for the directional-agreement / fragility checks)."""
    blended: float
    note: str
    w14_share: float          # 0..1 -- fraction of total weight from the 14d window
    v14: float | None
    v30: float | None
    vseason: float | None


def _blended_form_xwoba(p: PitcherProfile, config: dict) -> _FormBlend | None:
    """Weighted 14d / 30d / season blend of xwOBA-on-contact for one pitcher.

    Each window only contributes if it has enough sample (BIP floor for the
    two recent windows; season is always trusted if present). Missing windows
    have their weight redistributed to the available ones, so the blend stays
    well-formed even when a pitcher is fresh off the IL or skipped a turn.

    Returns None when no window has usable data -- caller falls back to the
    legacy single-window logic.
    """
    m = config["model"]
    fw = m.get("form_windows", {})
    min_bip = m.get("form_min_bip", {})

    candidates: list[tuple[str, float, float]] = []  # (label, value, weight)
    w14 = float(fw.get("d14", 0.5))
    w30 = float(fw.get("d30", 0.3))
    wseason = float(fw.get("season", 0.2))
    bip14_min = int(min_bip.get("d14", 25))
    bip30_min = int(min_bip.get("d30", 60))

    v14 = v30 = vseason = None
    if p.recent_xwoba_con_14d is not None and p.recent_bip_14d >= bip14_min:
        v14 = p.recent_xwoba_con_14d
        candidates.append(("14d", v14, w14))
    if p.recent_xwoba_con_30d is not None and p.recent_bip_30d >= bip30_min:
        v30 = p.recent_xwoba_con_30d
        candidates.append(("30d", v30, w30))
    if p.xwoba_con is not None:
        vseason = p.xwoba_con
        candidates.append(("season", vseason, wseason))

    if not candidates:
        return None
    total_w = sum(w for _, _, w in candidates)
    if total_w <= 0:
        return None
    blended = sum(v * w for _, v, w in candidates) / total_w
    # Compact note: which windows fed the blend, with their values (so the UI
    # can still show the raw 14d/30d/season numbers for transparency even
    # though the contribution itself is capped tight).
    label = "/".join(f"{name}={val:.3f}" for name, val, _ in candidates)
    w14_share = next((w / total_w for n, _, w in candidates if n == "14d"), 0.0)
    return _FormBlend(blended=blended, note=label, w14_share=w14_share,
                      v14=v14, v30=v30, vseason=vseason)


def _form_delta(home: PitcherProfile, away: PitcherProfile, scale: float,
                recent_weight: float = 0.6,
                config: dict | None = None) -> tuple[float, str, bool, bool]:
    """Recent-form delta. Returns (delta, note, available, fragile).

    Prefers the multi-window xwOBA-on-contact blend (14d / 30d / season,
    weights in model.form_windows); falls back to the legacy
    `recent_xwoba_con` + season blend when the new per-window fields aren't
    populated. + favors home (away's xwOBA is higher = away worse).

    CAP RATIONALE: short-window hitting stats are noise. 14d xwOBAcon is
    ~50 BIP, well within sampling variance; a "hot streak" that's actually
    a coin flip can swing a game several win-prob points if the cap is loose.
    We tighten to:

        normal:  +/- 1.5%
        extreme: +/- 2.5%, ONLY if 14d AND 30d agree directionally with
                 the blended diff AND sample sizes are meaningful

    The form component is also tagged `fragile=True` when its blend is
    >=60% driven by the 14d window -- i.e. the older windows underweighted
    OR missing -- so Step 3's stability classifier can mark the whole edge
    fragile if form is what's carrying it.
    """
    config = config or load_config()

    # Tight caps (config.model.form_normal_cap / form_extreme_cap, defaulted
    # in code so legacy configs still work). Both are FAR tighter than the
    # +/- 0.05 we had pre-2026-05-30 -- intentional, see docstring.
    m = config["model"]
    normal_cap = float(m.get("form_normal_cap", 0.015))
    extreme_cap = float(m.get("form_extreme_cap", 0.025))
    bip14_extreme_min = int(m.get("form_extreme_min_bip_14d", 30))
    bip30_extreme_min = int(m.get("form_extreme_min_bip_30d", 70))
    fragile_w14_threshold = float(m.get("form_fragile_w14_share", 0.60))

    # Multi-window path -- preferred. Only taken when BOTH sides have at
    # least one new per-window field populated; otherwise fall through to
    # the legacy single-window path so tests that hand-build PitcherProfile
    # with only the legacy fields behave exactly as they used to.
    h_has_window = home.recent_xwoba_con_14d is not None or home.recent_xwoba_con_30d is not None
    a_has_window = away.recent_xwoba_con_14d is not None or away.recent_xwoba_con_30d is not None
    if h_has_window and a_has_window:
        hb = _blended_form_xwoba(home, config)
        ab = _blended_form_xwoba(away, config)
        if hb is not None and ab is not None:
            diff = ab.blended - hb.blended  # + favors home (away xwOBA higher = away worse)
            raw_delta = diff * scale

            # Directional agreement: do 14d-only and 30d-only diffs match
            # the sign of the blended diff? Both must agree (and both must
            # exist on BOTH pitchers) to license the extreme cap.
            diff_14 = (ab.v14 - hb.v14) if (hb.v14 is not None and ab.v14 is not None) else None
            diff_30 = (ab.v30 - hb.v30) if (hb.v30 is not None and ab.v30 is not None) else None
            both_agree = (
                diff_14 is not None and diff_30 is not None
                and diff_14 * diff >= 0 and diff_30 * diff >= 0
            )
            # Sample-size gate: only extend the cap when ALL FOUR
            # pitcher-windows clear the meaningful-sample floor. A single
            # under-sampled side disqualifies the extreme cap.
            sample_meaningful = (
                home.recent_bip_14d >= bip14_extreme_min
                and away.recent_bip_14d >= bip14_extreme_min
                and home.recent_bip_30d >= bip30_extreme_min
                and away.recent_bip_30d >= bip30_extreme_min
            )
            cap = extreme_cap if (both_agree and sample_meaningful) else normal_cap
            delta = clamp(raw_delta, -cap, cap)

            # Fragile flag: form is "carrying noise" when the 14d window is
            # disproportionately driving the blend on EITHER pitcher.
            # `w14_share` is per-pitcher; we take the max to be conservative.
            w14_max = max(hb.w14_share, ab.w14_share)
            fragile = w14_max >= fragile_w14_threshold

            cap_note = "extreme cap" if cap == extreme_cap else "normal cap"
            note = f"xwOBAcon H {hb.note} | A {ab.note} ({cap_note} +/-{cap * 100:.1f}%)"
            return delta, note, True, fragile

    # Legacy path: single-window recent regressed toward season. No per-window
    # validation possible here -- always normal cap, always fragile (the
    # input data isn't rich enough to confirm the signal is stable).
    h_xw = _regress_recent(home.recent_xwoba_con, home.xwoba_con, recent_weight)
    a_xw = _regress_recent(away.recent_xwoba_con, away.xwoba_con, recent_weight)
    if h_xw is not None and a_xw is not None:
        diff = a_xw - h_xw
        delta = clamp(diff * scale, -normal_cap, normal_cap)
        return delta, f"recent xwOBAcon(reg) H={h_xw:.3f} A={a_xw:.3f} (normal cap +/-{normal_cap * 100:.1f}%)", True, True
    h_csw = _regress_recent(home.recent_csw_pct, home.csw_pct, recent_weight)
    a_csw = _regress_recent(away.recent_csw_pct, away.csw_pct, recent_weight)
    if h_csw is not None and a_csw is not None:
        diff = h_csw - a_csw  # higher CSW = better
        delta = clamp(diff * scale, -normal_cap, normal_cap)
        return delta, f"recent CSW%(reg) H={h_csw:.3f} A={a_csw:.3f} (normal cap +/-{normal_cap * 100:.1f}%)", True, True
    return 0.0, "missing recent Statcast form", False, False


# --- Confidence score (0-100) ------------------------------------------------
def compute_confidence(
    result: WinProbabilityResult,
    home_pitcher: PitcherProfile,
    away_pitcher: PitcherProfile,
    home_team: TeamProfile,
    away_team: TeamProfile,
    recommended_ev: float,
    config: dict | None = None,
    lineup_penalty: float = 0.0,
) -> float:
    """Composite 0-100 confidence in the recommendation.

    A weighted average of four normalized [0,1] sub-scores (weights in
    config.confidence.weights), scaled to 100:

      (a) data_completeness  — did we actually get both pitchers + team data?
          0.6 * mean(pitcher completeness) + 0.4 * team-data completeness.
      (b) sample_size        — min(home_IP, away_IP) / ip_full_confidence,
          capped at 1. The weakest-link pitcher governs trust.
      (c) edge_magnitude     — |recommended EV%| / edge_full_confidence, capped.
          Bigger measured edges are (weakly) more trustworthy.
      (d) component_agreement— do the skill components (starter/bullpen/park/form)
          point the same way? 1.0 = unanimous, ~0.5 = the edge hinges on a single
          factor while others disagree (a coin-flip we should distrust).

    The score is intentionally conservative: missing data or internal
    disagreement pulls it down even when the raw EV looks juicy.
    """
    config = config or load_config()
    cw = config["confidence"]
    weights = cw["weights"]

    # (a) data completeness
    pitcher_dc = (home_pitcher.data_completeness + away_pitcher.data_completeness) / 2.0
    team_flags = [
        home_team.has_record, away_team.has_record,
        home_team.offense_wrc_plus is not None, away_team.offense_wrc_plus is not None,
        home_team.bullpen_fip is not None, away_team.bullpen_fip is not None,
    ]
    team_dc = sum(team_flags) / len(team_flags)
    data_completeness = 0.6 * pitcher_dc + 0.4 * team_dc

    # (b) sample size — weakest-link pitcher IP
    ips = [ip for ip in (home_pitcher.ip, away_pitcher.ip) if ip is not None]
    if ips:
        sample_size = clamp(min(ips) / float(cw.get("ip_full_confidence", 60.0)), 0.0, 1.0)
    else:
        sample_size = 0.0

    # (c) edge magnitude
    edge_magnitude = clamp(abs(recommended_ev) / float(cw.get("edge_full_confidence", 0.10)), 0.0, 1.0)

    # (d) component agreement among skill factors
    skill = [c for c in result.components if c.name in {"starter", "bullpen", "park", "form"} and c.weighted_delta != 0.0]
    if skill:
        net = sum(c.weighted_delta for c in skill)
        total_abs = sum(abs(c.weighted_delta) for c in skill)
        if total_abs > 0 and net != 0:
            sign = 1.0 if net > 0 else -1.0
            agreeing = sum(abs(c.weighted_delta) for c in skill if (c.weighted_delta > 0) == (sign > 0))
            component_agreement = agreeing / total_abs
        else:
            component_agreement = 0.5
    else:
        component_agreement = 0.5

    score = (
        weights["data_completeness"] * data_completeness
        + weights["sample_size"] * sample_size
        + weights["edge_magnitude"] * edge_magnitude
        + weights["component_agreement"] * component_agreement
    )
    total_weight = sum(weights.values())
    confidence = 100.0 * score / total_weight if total_weight else 0.0
    # Same lineup-penalty mechanic as compute_data_confidence: a projected
    # lineup is a measurable confidence gap regardless of EV magnitude. Floor
    # at 0 so we never report a negative score.
    confidence = max(0.0, confidence - lineup_penalty)
    result.confidence = round(confidence, 1)
    return result.confidence


# --- Data confidence (no EV dependency) --------------------------------------
def compute_data_confidence(
    result: WinProbabilityResult,
    home_pitcher: PitcherProfile,
    away_pitcher: PitcherProfile,
    home_team: TeamProfile,
    away_team: TeamProfile,
    config: dict | None = None,
    lineup_penalty: float = 0.0,
) -> float:
    """0-100 confidence in the DATA INPUTS, with no EV dependency.

    The dynamic market blend needs to be picked BEFORE the EV is computed, so
    we can't use `compute_confidence` (which folds in edge_magnitude). This
    helper returns the same composite minus the edge term, renormalized over
    the remaining three weights. The full `compute_confidence` is still the
    public score shown to the user; this exists only to choose the blend.

    Why no edge term: letting EV drive the blend would create a fragile
    feedback loop (a tiny edge -> tiny blend boost -> slightly bigger edge
    -> larger blend...). Anchoring the blend on data quality instead means
    the model only earns market-overruling weight when its inputs are good.
    """
    config = config or load_config()
    cw = config["confidence"]
    weights = cw["weights"]

    pitcher_dc = (home_pitcher.data_completeness + away_pitcher.data_completeness) / 2.0
    team_flags = [
        home_team.has_record, away_team.has_record,
        home_team.offense_wrc_plus is not None, away_team.offense_wrc_plus is not None,
        home_team.bullpen_fip is not None, away_team.bullpen_fip is not None,
    ]
    team_dc = sum(team_flags) / len(team_flags)
    data_completeness = 0.6 * pitcher_dc + 0.4 * team_dc

    ips = [ip for ip in (home_pitcher.ip, away_pitcher.ip) if ip is not None]
    sample_size = clamp(min(ips) / float(cw.get("ip_full_confidence", 60.0)), 0.0, 1.0) if ips else 0.0

    skill = [c for c in result.components if c.name in {"starter", "bullpen", "park", "form"} and c.weighted_delta != 0.0]
    if skill:
        net = sum(c.weighted_delta for c in skill)
        total_abs = sum(abs(c.weighted_delta) for c in skill)
        if total_abs > 0 and net != 0:
            sign = 1.0 if net > 0 else -1.0
            agreeing = sum(abs(c.weighted_delta) for c in skill if (c.weighted_delta > 0) == (sign > 0))
            component_agreement = agreeing / total_abs
        else:
            component_agreement = 0.5
    else:
        component_agreement = 0.5

    # Renormalize over the three non-EV weights so this stays comparable in
    # magnitude to the full confidence score (same 0-100 scale, same units).
    w_data = weights["data_completeness"]
    w_samp = weights["sample_size"]
    w_agree = weights["component_agreement"]
    denom = w_data + w_samp + w_agree
    if denom <= 0:
        return 0.0
    score = (w_data * data_completeness + w_samp * sample_size + w_agree * component_agreement) / denom
    # `lineup_penalty` subtracts confidence points when today's lineups
    # haven't been confirmed yet (we're effectively betting on a projected
    # roster). Floored at 0 so a heavy penalty + low base doesn't go negative.
    return round(max(0.0, 100.0 * score - lineup_penalty), 1)


def resolve_market_blend(data_confidence: float, model_config: dict) -> tuple[float, str]:
    """Pick the market-blend weight from a confidence tier table.

    `model_config["market_blend"]` is either:
      * a scalar  -> legacy fixed blend (used for every game) -> tier="fixed"
      * a dict    -> tiered: {low_conf, mid_conf, high_conf,
                              mid_threshold, high_threshold}

    Returns (blend_weight, tier_label). `blend_weight` is the MODEL side of
    `blended = blend*model + (1-blend)*market`, so higher = more model.

    Conservative default: even the high-confidence tier keeps the market as
    the primary anchor (blend <= 0.5). The model only earns more weight than
    the market once CLV proves out at that confidence level. Re-tune upward
    in config once that's true.
    """
    mb = model_config.get("market_blend", 0.35)
    if not isinstance(mb, dict):
        # Legacy scalar config: same blend for every game.
        return float(mb), "fixed"

    high_t = float(mb.get("high_threshold", 85.0))
    mid_t = float(mb.get("mid_threshold", 70.0))
    if data_confidence >= high_t:
        return float(mb.get("high_conf", 0.45)), "high"
    if data_confidence >= mid_t:
        return float(mb.get("mid_conf", 0.35)), "mid"
    return float(mb.get("low_conf", 0.25)), "low"
