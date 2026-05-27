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
) -> WinProbabilityResult:
    """Run the model and return the home win probability with full breakdown."""
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
        starter_delta = log5(h_pwp, a_pwp) - 0.5
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
    form_delta, form_note, form_avail = _form_delta(
        home_pitcher, away_pitcher,
        float(m.get("form_scale", 0.5)), float(m.get("form_recent_weight", 0.6)),
    )
    components.append(_mk("form", form_delta, weights["form"], form_note, form_avail))

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


def _mk(name: str, raw: float, weight: float, note: str, available: bool) -> Component:
    weight = float(weight)
    return Component(name=name, raw_delta=raw, weight=weight, weighted_delta=raw * weight, note=note, available=available)


def _regress_recent(recent: float | None, season: float | None, w_recent: float) -> float | None:
    """Regress a recent rate toward its season value to damp streak noise:
    w_recent*recent + (1-w_recent)*season. Falls back to recent if season is absent."""
    if recent is None:
        return None
    if season is None:
        return recent
    return w_recent * recent + (1.0 - w_recent) * season


def _form_delta(home: PitcherProfile, away: PitcherProfile, scale: float,
                recent_weight: float = 0.6) -> tuple[float, str, bool]:
    """Recent-form delta from pitchers' last-5-start Statcast, each regressed
    toward the pitcher's season rate (+ favors home)."""
    # Prefer xwOBA-on-contact (lower = better); home advantage when home's
    # (season-regressed) recent xwOBA is lower than away's.
    h_xw = _regress_recent(home.recent_xwoba_con, home.xwoba_con, recent_weight)
    a_xw = _regress_recent(away.recent_xwoba_con, away.xwoba_con, recent_weight)
    if h_xw is not None and a_xw is not None:
        diff = a_xw - h_xw  # + favors home
        delta = clamp(diff * scale, -0.05, 0.05)
        return delta, f"recent xwOBAcon(reg) H={h_xw:.3f} A={a_xw:.3f}", True
    h_csw = _regress_recent(home.recent_csw_pct, home.csw_pct, recent_weight)
    a_csw = _regress_recent(away.recent_csw_pct, away.csw_pct, recent_weight)
    if h_csw is not None and a_csw is not None:
        diff = h_csw - a_csw  # higher CSW = better
        delta = clamp(diff * scale, -0.05, 0.05)
        return delta, f"recent CSW%(reg) H={h_csw:.3f} A={a_csw:.3f}", True
    return 0.0, "missing recent Statcast form", False


# --- Confidence score (0-100) ------------------------------------------------
def compute_confidence(
    result: WinProbabilityResult,
    home_pitcher: PitcherProfile,
    away_pitcher: PitcherProfile,
    home_team: TeamProfile,
    away_team: TeamProfile,
    recommended_ev: float,
    config: dict | None = None,
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
    result.confidence = round(confidence, 1)
    return result.confidence
