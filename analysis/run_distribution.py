"""Purpose-built run estimator + total-runs DISTRIBUTION for the totals model.

The bet is P(total > line) / P(total < line), which needs a DISTRIBUTION, not a
point estimate. We:

  1. Build absolute expected runs per side from transparent run-environment
     factors -- offense (wRC+), opposing staff rate with EXPLICIT bullpen innings,
     park, weather.
  2. Recenter onto the market total, keeping only a BOUNDED TILT (so starter
     quality already priced by the market nets ~0 -- no double-count).
  3. Model total runs as a NEGATIVE BINOMIAL (over-dispersed: var > mean) and read
     P(over)/P(under) at the posted line off its CDF.

Every factor degrades to neutral on missing data (never fabricates a tilt); the
caller drops confidence / flags fragility. NOT used for the moneyline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from scipy.stats import nbinom

from mlb_value_bot.utils import get_logger

log = get_logger("analysis.run_distribution")


@dataclass
class RunDistribution:
    market_total: float
    raw_model_total: float | None   # absolute build (for the divergence guard)
    expected_total: float | None    # market-recentered E[T] (distribution mean)
    anchor_mean: float | None = None  # market-implied MEAN (zero-tilt reference)
    tilt: float | None = None       # clamped raw-vs-anchor shift actually applied
    variance: float | None = None
    p_over: float | None = None
    p_under: float | None = None
    p_push: float = 0.0
    home_runs: float | None = None
    away_runs: float | None = None
    components: list = field(default_factory=list)  # transparent breakdown
    available: bool = False
    notes: list = field(default_factory=list)


def _ip_per_start(pp) -> float | None:
    """Season innings per start (ip / games_started), or None."""
    ip = getattr(pp, "ip", None)
    gs = getattr(pp, "games_started", None)
    if ip and gs and gs > 0:
        return float(ip) / float(gs)
    return None


def _staff_rate(starter_rate, bullpen_fip, ip_per_start, league_rate, cfg) -> tuple[float, str]:
    """Innings-weighted runs/9 of the staff: starter over SP_IP, bullpen over the
    rest. EXPLICIT bullpen innings -- the thing naive totals models get wrong."""
    lo = float(cfg.get("min", 3.0))
    hi = float(cfg.get("max", 7.0))
    sp_ip = max(lo, min(hi, ip_per_start if ip_per_start else float(cfg.get("default_ip_per_start", 5.3))))
    sp_share = sp_ip / 9.0
    sr = starter_rate if starter_rate is not None else league_rate
    br = bullpen_fip if bullpen_fip is not None else league_rate
    rate = sp_share * sr + (1.0 - sp_share) * br
    note = f"SP {sr:.2f}x{sp_ip:.1f}ip + RP {br:.2f}x{9 - sp_ip:.1f}ip = {rate:.2f}"
    if starter_rate is None:
        note += " (SP rate missing->league)"
    if bullpen_fip is None:
        note += " (RP FIP missing->league)"
    return rate, note


def expected_runs(home_tp, away_tp, home_pp, away_pp, weather, config) -> dict:
    """Absolute expected runs per side + a transparent component list. Degrades
    to neutral inputs (never raises); records what was missing."""
    lg = config.get("league", {})
    tcfg = config.get("totals", {})
    league_rpg = float(lg.get("runs_per_game", 4.5))
    league_rate = float(lg.get("avg_xfip", 4.0))
    si = tcfg.get("starter_innings", {})

    home_off = home_tp.offense_wrc_plus
    away_off = away_tp.offense_wrc_plus
    notes = []
    if home_off is None:
        home_off, notes = 100.0, notes + ["home wRC+ missing->neutral"]
    if away_off is None:
        away_off, notes = 100.0, notes + ["away wRC+ missing->neutral"]

    home_staff, hs_note = _staff_rate(home_pp.primary_rate(), home_tp.bullpen_fip,
                                      _ip_per_start(home_pp), league_rate, si)
    away_staff, as_note = _staff_rate(away_pp.primary_rate(), away_tp.bullpen_fip,
                                      _ip_per_start(away_pp), league_rate, si)

    park_mult = float(home_tp.park_factor or 100.0) / 100.0
    wx_mult = weather.multiplier if weather else 1.0

    # Home bats vs away staff (and vice versa); park + weather hit both teams.
    home_rs = league_rpg * (home_off / 100.0) * (away_staff / league_rate) * park_mult * wx_mult
    away_rs = league_rpg * (away_off / 100.0) * (home_staff / league_rate) * park_mult * wx_mult
    home_rs = max(0.5, min(14.0, home_rs))
    away_rs = max(0.5, min(14.0, away_rs))

    components = [
        {"name": "home_offense", "value": f"wRC+ {home_off:.0f} vs away staff {away_staff:.2f}", "runs": round(home_rs, 2)},
        {"name": "away_offense", "value": f"wRC+ {away_off:.0f} vs home staff {home_staff:.2f}", "runs": round(away_rs, 2)},
        {"name": "park", "value": f"PF {home_tp.park_factor:.0f}", "mult": round(park_mult, 3)},
        {"name": "weather", "value": (weather.note if weather else "n/a"),
         "mult": round(wx_mult, 3), "available": bool(weather and weather.available)},
        {"name": "home_staff", "value": hs_note},
        {"name": "away_staff", "value": as_note},
    ]
    return {"home_rs": round(home_rs, 2), "away_rs": round(away_rs, 2),
            "raw_total": round(home_rs + away_rs, 2), "components": components, "notes": notes}


def _nb_params(mean: float, variance: float) -> tuple[float, float]:
    """scipy nbinom (n, p) from mean+variance (variance must exceed mean)."""
    variance = max(variance, mean * 1.0001)   # enforce over-dispersion
    p = mean / variance
    n = mean * mean / (variance - mean)
    return n, p


def _variance_for(mean: float, tcfg: dict) -> float:
    lg = tcfg.get("league", {})
    avg_total = float(lg.get("avg_total", 8.8))
    var_at_avg = float(lg.get("total_variance", 22.0))
    # Dispersion proportional to the mean (over-dispersed: var/mean = ratio > 1).
    return mean * (var_at_avg / avg_total)


def _p_over_for_mean(mean: float, line: float, tcfg: dict) -> tuple[float, float, float]:
    """(p_over, p_under, p_push) at `line` for a NB with this mean."""
    n, p = _nb_params(mean, _variance_for(mean, tcfg))
    if abs(line - round(line)) < 1e-9:                       # integer line -> push possible
        k = int(round(line))
        return float(nbinom.sf(k, n, p)), float(nbinom.cdf(k - 1, n, p)), float(nbinom.pmf(k, n, p))
    k_over = int(line) + 1                                   # T >= k_over -> over
    po = float(nbinom.sf(k_over - 1, n, p))
    return po, 1.0 - po, 0.0


def _solve_mean_for_p_over(target_p_over: float, line: float, tcfg: dict) -> float:
    """Mean that makes P(over) at `line` equal target -- anchors the distribution
    to the MARKET's de-vigged P(over) so zero tilt = agree with market (no
    spurious edge from the count distribution's right-skew)."""
    lo, hi = max(2.0, line - 6.0), line + 6.0
    for _ in range(45):
        mid = (lo + hi) / 2.0
        if _p_over_for_mean(mid, line, tcfg)[0] < target_p_over:
            lo = mid                                         # need a higher mean
        else:
            hi = mid
    return (lo + hi) / 2.0


def run_distribution(home_tp, away_tp, home_pp, away_pp, market_total,
                     market_devig_over, weather, config) -> RunDistribution:
    """Full distribution + P(over)/P(under). Anchored to the market: the mean is
    calibrated so zero run-tilt reproduces the market's de-vigged P(over), then
    our bounded run-tilt shifts it."""
    tcfg = config.get("totals", {})
    if market_total is None:
        return RunDistribution(market_total=0.0, raw_model_total=None, expected_total=None,
                               variance=None, p_over=None, p_under=None,
                               available=False, notes=["no market total"])

    er = expected_runs(home_tp, away_tp, home_pp, away_pp, weather, config)
    raw = er["raw_total"]
    line = float(market_total)
    max_tilt = float(tcfg.get("max_tilt_runs", 1.5))

    # Calibrate the anchor mean to the market's P(over) (fall back to the line
    # itself when the market prob is unknown).
    anchor_mean = (_solve_mean_for_p_over(market_devig_over, line, tcfg)
                   if market_devig_over is not None else line)

    # Tilt is MEAN-vs-MEAN: `raw` is a distribution mean, so it is compared to
    # the market-implied mean (`anchor_mean`), NOT the posted line. The line
    # sits near the distribution's MEDIAN, which for a right-skewed count
    # distribution is ~0.8 runs BELOW the implied mean. Tilting off the line
    # re-imports exactly the skew offset the anchor solve removes: measured on
    # production data it made the tilt positive on 78% of games (raw > line)
    # even though the raw projection was unbiased vs actual totals, which
    # pushed P(over) > 0.5 on 83% of games and made ~97% of value picks OVERs.
    tilt = max(-max_tilt, min(max_tilt, raw - anchor_mean))
    model_mean = anchor_mean + tilt
    p_over, p_under, p_push = _p_over_for_mean(model_mean, line, tcfg)

    return RunDistribution(
        market_total=line, raw_model_total=raw, expected_total=round(model_mean, 2),
        anchor_mean=round(anchor_mean, 2), tilt=round(tilt, 2),
        variance=round(_variance_for(model_mean, tcfg), 2),
        p_over=round(p_over, 4), p_under=round(p_under, 4), p_push=round(p_push, 4),
        home_runs=er["home_rs"], away_runs=er["away_rs"], components=er["components"],
        available=True,
        notes=er["notes"] + [f"tilt {tilt:+.2f} off anchor mean {anchor_mean:.2f} "
                             f"(raw {raw:.2f}; mkt line {line:.1f})"],
    )
