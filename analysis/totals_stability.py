"""Pick-level edge-stability classification for the TOTALS model.

The moneyline classifier reasons over win-probability Components and a home/away
pick, none of which transfers to totals -- so this is a parallel, totals-specific
classifier built around the inputs that actually make a totals read trustworthy
or fragile.

Per the spec, the two archetypes are a STABILITY FILTER, not the signal: the
over/under lean is whatever the run distribution says; the archetype only governs
how much we trust it.

  STABLE   both starters rated, weather available, BOTH lineups confirmed,
           and no hard fragile signal. ("Two stable starters vs two weak
           confirmed lineups with weather" -> a stable under candidate.)
  FRAGILE  any hard fragile signal: a TBD/unrated starter, an UNAVAILABLE
           lineup feed, missing weather, fading the sharp total, or a raw-vs-
           market gap near the skip limit.
  MODERATE in between (e.g. a normal morning run with projected -- not missing
           -- lineups).

Hard rule the caller enforces: never size a FRAGILE totals edge as "Strong".
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TotalsStability:
    label: str                                      # "stable" | "moderate" | "fragile"
    hard_fragile_signals: list[str] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)  # the solid inputs (for UI)


def classify_totals_stability(
    profiles,
    weather,
    rd,
    home_lu,
    away_lu,
    sharp_fade_pp: float | None,
    config: dict,
) -> TotalsStability:
    """Classify a totals pick STABLE / MODERATE / FRAGILE.

    `sharp_fade_pp` is the pp by which our blended P(pick side) exceeds the sharp
    totals consensus on that side (positive = fading the sharps); None = no sharp
    data. The pipeline's sanity guard already SKIPS games above its fade limit;
    this flags the band below it.
    """
    tcfg = config.get("totals", {})
    scfg = tcfg.get("stability", {})
    sharp_fade_fragile_pp = float(scfg.get("sharp_fade_fragile_pp", 3.0))
    max_div = float(tcfg.get("sanity", {}).get("max_total_divergence_runs", 1.75))

    home_pp, away_pp = profiles.home_pp, profiles.away_pp
    hard: list[str] = []

    # TBD / unrated starter -> the staff-rate input fell back to league average.
    for label, pp in (("home", home_pp), ("away", away_pp)):
        if pp.name is None or pp.primary_rate() is None:
            hard.append(f"{label} starter TBD/unrated")

    # Weather missing -> we'd be betting blind to the dominant totals factor.
    weather_ok = weather is not None and weather.available
    if not weather_ok:
        hard.append("weather unavailable")

    # Lineups: an UNAVAILABLE feed (genuine gap, or feature disabled so lu=None)
    # is a hard signal; a merely PROJECTED lineup is timing, not fragility.
    statuses = [getattr(lu, "status", "unavailable") if lu is not None else "unavailable"
                for lu in (home_lu, away_lu)]
    if any(s == "unavailable" for s in statuses):
        hard.append("lineup feed unavailable")

    # Fading the sharp total beyond the fragility threshold.
    if sharp_fade_pp is not None and sharp_fade_pp * 100.0 >= sharp_fade_fragile_pp:
        hard.append(f"fading sharp total by {sharp_fade_pp * 100:.1f}pp")

    # Raw build sits near the model-vs-market divergence skip limit -- a sign
    # we may be missing what the market has even if it didn't cross the line.
    if rd is not None and rd.raw_model_total is not None and rd.market_total:
        gap = abs(rd.raw_model_total - rd.market_total)
        if gap >= 0.75 * max_div:
            hard.append(f"raw-vs-market gap {gap:.2f}r near skip limit")

    both_confirmed = all(s == "confirmed" for s in statuses)
    both_rated = home_pp.primary_rate() is not None and away_pp.primary_rate() is not None

    if hard:
        label = "fragile"
    elif both_rated and weather_ok and both_confirmed:
        label = "stable"
    else:
        label = "moderate"

    drivers: list[str] = []
    if both_rated:
        drivers.append("both starters rated")
    if weather_ok:
        drivers.append("weather available")
    if both_confirmed:
        drivers.append("lineups confirmed")

    return TotalsStability(label=label, hard_fragile_signals=hard, drivers=drivers)
