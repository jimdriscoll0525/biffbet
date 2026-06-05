"""Pick-level edge-stability classification: STABLE / MODERATE / FRAGILE.

The intent is to separate edges that come from durable, well-sampled signals
(starter quality, season bullpen rating, confirmed lineup vs key bats) from
edges that hinge on noisy or incomplete inputs (a 14-day xwOBAcon hot streak,
a projected lineup, missing data, or fading sharp consensus).

Definitions (from the upgrade spec):

  STABLE   driven mostly by starting pitcher, bullpen, confirmed lineup,
           or projected run differential.
  FRAGILE  driven mostly by recent form, an UNAVAILABLE lineup feed,
           missing data, or sharp fade. (A merely *projected* lineup is
           timing, not fragility -- it's priced via the Adjusted-EV haircut,
           not a hard fragile signal.)
  MODERATE in between.

Hard rule the caller is expected to enforce after reading the label:
"NEVER display Strong sizing when edge is FRAGILE -- downgrade it."

The classifier returns BOTH a label and a breakdown dict (per-driver shares,
hard-signal flags) so we can audit why a pick was tagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mlb_value_bot.analysis.win_probability import Component


# Component-name buckets. A component classified as "stable" contributes to
# stable_drive when it's pushing toward our pick; "fragile" contributes to
# fragile_drive. Components not in either set are neutral (counted in the
# total drive, but not as evidence either way). "lineup" is dynamic -- only
# stable when its `available` flag is True (i.e. both lineups were confirmed
# at win_probability evaluation time).
_STABLE_COMPONENT_NAMES = {"starter", "bullpen"}
# Bullpen-fatigue is real-data when present but the schedule is non-linear
# and the magnitudes are tiny -- treat as MODERATE rather than STABLE so it
# doesn't tip a marginal pick to STABLE by itself.


@dataclass
class EdgeStability:
    """Output of classify_edge_stability(). `label` is what gates the tier
    downgrade and what the UI badges. The other fields are for transparency
    and Step 8 segment analysis."""
    label: str                      # "stable" | "moderate" | "fragile"
    stable_share: float             # 0..1, fraction of pick-side drive from stable components
    fragile_share: float            # 0..1, fraction from fragile components
    hard_fragile_signals: list[str] = field(default_factory=list)
    # Top drivers ordered by absolute pick-aligned contribution (for UI).
    drivers: list[dict] = field(default_factory=list)


def classify_edge_stability(
    components: Iterable[Component],
    best_side: str,
    sharp_fade_pp: float | None,
    home_lineup_status,
    away_lineup_status,
    config: dict | None = None,
) -> EdgeStability:
    """Classify the pick's edge as STABLE / MODERATE / FRAGILE.

    `sharp_fade_pp` is the pp-points we're more bullish on our pick side than
    sharp consensus (positive = fading sharps). None = no sharp data.
    `home/away_lineup_status` are the LineupStatus objects (or None).
    """
    config = config or {}
    cfg = config.get("stability", {})
    stable_threshold = float(cfg.get("stable_share_min", 0.60))
    fragile_threshold = float(cfg.get("fragile_share_min", 0.50))
    sharp_fade_fragile_pp = float(cfg.get("sharp_fade_fragile_pp", 3.0))
    missing_component_fragile_n = int(cfg.get("missing_component_fragile_n", 2))

    sign = 1.0 if best_side == "home" else -1.0

    # Drivers: components whose pick-side contribution is positive (>0 when
    # aligned to the pick). Components that ARGUE AGAINST our pick are
    # ignored here -- they're not "drivers" of the edge.
    drivers: list[tuple[str, float, Component]] = []
    for c in components:
        aligned = c.weighted_delta * sign
        if aligned > 1e-9:
            drivers.append((c.name, aligned, c))

    total_drive = sum(a for _, a, _ in drivers)
    if total_drive <= 0:
        # No positive drive on our side (e.g. pick is purely a market/blend
        # artifact). Default to MODERATE -- not enough signal to call stable.
        return EdgeStability(label="moderate", stable_share=0.0, fragile_share=0.0)

    stable_drive = 0.0
    fragile_drive = 0.0
    for name, aligned, comp in drivers:
        if not comp.available:
            # A driver whose data is missing IS a fragile driver. Should
            # rarely happen with positive aligned weight (missing components
            # have 0 weighted_delta) -- defensive case.
            fragile_drive += aligned
        elif comp.fragile:
            # Step 1's fragile flag on the form component (14d-dominated blend).
            fragile_drive += aligned
        elif name in _STABLE_COMPONENT_NAMES:
            stable_drive += aligned
        elif name == "lineup":
            # Lineup is STABLE when its component is available (the
            # win_probability check requires BOTH sides confirmed). Otherwise
            # treated as fragile via the not-available branch above.
            stable_drive += aligned
        # Other components (park, home_field, bullpen_fatigue) are NEUTRAL.

    # Hard fragility signals -- ANY one of these flips the label to FRAGILE
    # regardless of the share math.
    hard_signals: list[str] = []

    # Sharp fade: we're tilting against the sharpest counterparty on the
    # board. The pipeline's sanity guard already SKIPS games where the
    # fade is >5pp, but anything in (2pp, 5pp] is still a fragility signal.
    if sharp_fade_pp is not None and sharp_fade_pp * 100.0 >= sharp_fade_fragile_pp:
        hard_signals.append(f"fading sharps by {sharp_fade_pp * 100:.1f}pp")

    # Lineup state: an UNAVAILABLE lineup feed (a genuine API/data gap, or the
    # feature disabled so lu is None) on EITHER side means the offense input is
    # missing -- that's a hard fragility signal. A merely PROJECTED lineup (not
    # posted yet -- pure timing, which resolves as first pitch nears since the
    # engine re-runs through the day) is NOT a fragility signal: forcing every
    # pre-lineup (morning) run to FRAGILE was a flat slate-wide tax, not a
    # discriminator. Projected-lineup uncertainty is instead carried by the
    # graduated lineup CONFIDENCE penalty (config.lineup.confidence_penalty).
    for label, lu in (("home", home_lineup_status), ("away", away_lineup_status)):
        status = getattr(lu, "status", None)
        if lu is None or status == "unavailable":
            hard_signals.append(f"{label} lineup {status or 'no data'}")
            break  # one signal is enough; don't double-count both sides

    # Missing-data pile-up: betting on a game where multiple model
    # components are unavailable is fragile by definition.
    missing_count = sum(1 for c in components if not c.available)
    if missing_count >= missing_component_fragile_n:
        hard_signals.append(f"{missing_count} components missing data")

    stable_share = stable_drive / total_drive
    fragile_share = fragile_drive / total_drive

    if hard_signals or fragile_share >= fragile_threshold:
        label = "fragile"
    elif stable_share >= stable_threshold:
        label = "stable"
    else:
        label = "moderate"

    # Top drivers for UI: sort by absolute aligned contribution descending.
    drivers_sorted = sorted(drivers, key=lambda x: -abs(x[1]))
    driver_summaries = [
        {"name": n, "aligned_delta": round(a, 4), "fragile": c.fragile, "available": c.available}
        for n, a, c in drivers_sorted
    ]

    return EdgeStability(
        label=label,
        stable_share=round(stable_share, 3),
        fragile_share=round(fragile_share, 3),
        hard_fragile_signals=hard_signals,
        drivers=driver_summaries,
    )
