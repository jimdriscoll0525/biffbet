# Upgrades roadmap

Design notes for upgrades discussed but intentionally deferred from the
2026-05-28 "safe core" landing (dynamic blend + Kelly tiers + segmentation
+ UI sections). Each note: scope, data sources, integration plan, fallback
when data is missing, validation strategy. Pick one per session; do it
properly; don't bundle.

---

## (2) Weighted multi-window recent form

**Goal:** replace the current "last ~5 starts xwOBA-on-contact" form delta
with a weighted blend of three windows so streak noise is damped and the
season baseline keeps the model honest.

**Weights (subject to backtest):** 14d 50% / 30d 30% / season 20%.

**Integration:**
- `analysis/pitcher_metrics.py::_statcast_rates` already pulls Statcast for
  a `start..end` range. Refactor to pull THREE ranges per pitcher (14d, 30d,
  season-to-date) and return all three on `PitcherProfile`. Cache per
  (pitcher_id, window) — the season pull doesn't change within a day, the
  30d pull changes once a day, the 14d pull changes a few times a day.
- `analysis/win_probability._form_delta`: blend the three windows before
  computing `xwOBA_diff * scale`.
- New tunables in config: `model.form_windows: { d14, d30, season }` (weights),
  `model.form_min_pa: 30` (require at least N batters in the 14d window before
  trusting it -- otherwise fall back to 30d-only).

**Cost:** ~3x the current Statcast pulls per pitcher per slate run. Cache
helps but the morning cold run will be slower (~30s -> ~60s for a 12-game
slate). Acceptable.

**Fallback:** if 14d is empty (rotation skip, IL), shift to 30d 70% / season
30%. If both recent windows are empty, season-only. Mark the component
`available=True` but tag the note with which windows were used.

**Validation:** rerun backtest with form deltas computed both ways across
existing seasons; compare CLV and ROI by window blend. Don't ship unless
the blended version isn't WORSE than current.

---

## (3) Bullpen fatigue

**Goal:** replace the current static "season bullpen FIP" component with a
fatigue-aware version that knows which relievers are available today.

**Data source:** MLB Stats API `/v1/schedule?sportId=1&teamId=N&startDate=
YYYY-MM-DD&endDate=YYYY-MM-DD&hydrate=probablePitcher,linescore`. For each
of a team's last 3 games, hydrate the game with `boxscore` and pull pitcher
appearances (each entry has `pitchesThrown`, `inheritedRunners`, plus the
`gamesPlayed.pitching` map for back-to-back days). No FanGraphs required;
no auth required.

**New module:** `data/bullpen_status.py` with one function:
```
get_bullpen_status(team_name: str, as_of: date, mlb: MLBClient) ->
    BullpenStatus(
        relievers_used_yesterday: list[Reliever],
        b2b_arms: list[Reliever],                   # pitched 2 days in a row
        unavailable: list[Reliever],                # >= 35 pitches yesterday OR
                                                    # >= 3 appearances in last 3 days
        leverage_arms_available: int,               # high-leverage relievers free
        notes: list[str],
    )
```

**Identifying "leverage" arms:** simplest heuristic is "relievers with the
team's top-3 lowest reliever ERAs YTD" from the same FanGraphs pitching CSV
the model already consumes. Don't hardcode names; derive each run.

**Component change:** `win_probability` gains a `bullpen_fatigue_delta`
that ADDS to the existing `bullpen_delta`:
- If fewer leverage arms available than usual -> small negative for that
  team (favors opponent in close-late spots).
- If both teams equally fatigued -> ~0.
- Clamped to +/- 0.03 (smaller than the FIP delta -- this is a tilt, not
  a takeover).

**Fallback:** if the boxscore query fails or the leverage list can't be
derived, return a `BullpenStatus` with `available=False` and the component
contributes 0 with the note "bullpen status unavailable" -- same pattern as
every other component.

**UI:** add a "Bullpen availability" line under "Dynamic adjustments"
showing each team's leverage-arms-available count and a chip when one team
has notable fatigue.

**Validation:** segment performance by "either bullpen fatigued" vs both
fresh; if the fatigue signal isn't predictive of CLV+ on close-game lines,
keep the component but weight it small.

---

## (4) Lineup confirmation

**Goal:** lower confidence on picks where the lineup isn't yet confirmed;
adjust offense when it is; surface missing key bats.

**Data source:** MLB Stats API `/v1.1/game/{gamePk}/feed/live` exposes
`liveData.boxscore.teams.<side>.battingOrder` once lineups are posted
(typically ~2h before first pitch). Also `gameData.players.{id}.status`
for injury status when fully scratched.

**Timing problem:** the engine runs every 30 min during the closing window;
lineups land at varying times per game. Solution: on each run,
- For games > 3h before first pitch: don't query lineups; mark
  `lineup_status = "projected"` in reasoning.
- For games <= 3h before first pitch: query lineups; mark
  `lineup_status = "confirmed"` if both teams' battingOrder is non-empty,
  else "projected".

**Confidence impact:** when `lineup_status == "projected"`, multiply
`compute_data_confidence` by 0.9 (or subtract 5 points). Add a new chip
"Projected lineup" to the UI. When confirmed, no penalty.

**Offense adjustment with confirmed lineups:** compute today's lineup wRC+
as the average of the 9 starters' season wRC+ (from FanGraphs batting CSV),
weighted by typical PA share. Pass this in place of the team's overall
wRC+. Cap delta at +/- 5 wRC+ points relative to the team's full-roster
wRC+ to prevent extreme deviations on one-off rest days.

**Missing key bats:** for each team, identify the top-3 hitters by wRC+
(again, derive don't hardcode). If a top-3 hitter isn't in today's lineup
(injury, day off), add a `notes` line and apply a small offense penalty
(-3 wRC+ proxy per missing top-3 bat, capped).

**Fallback:** every step degrades cleanly. No CSV -> no key-bats logic,
no per-lineup wRC+, just the existing team-level offense component plus
the "projected lineup" confidence penalty. No game feed available -> mark
projected, continue.

**Validation:** segment by `lineup_status` after a few hundred bets.
Confirmed-lineup picks should show better CLV than projected if the
adjustment is real.

---

## (5) Sharp/square market intelligence

**Goal:** "market agreement" indicator that's louder when sharp books and
square books disagree (one of them is wrong).

**Data source:** already in The Odds API response -- the existing
`OddsClient` parses all returned books and currently picks the best price.
We just need to keep the per-book table around instead of collapsing to
the best.

**Definition:** sharp books = Pinnacle, Circa, Bookmaker. Square books =
DraftKings, FanDuel, BetMGM, Caesars. (Make this a config list, don't
hardcode.) The user's current config is DK-only (`config.yaml`
`odds_api.bookmakers`); they'd need to ADD the sharp books to the list for
this signal to exist. The DK price stays the bet price (preserves
"displayed line == price you'll actually bet").

**Two new fields per game:**
- `price_dispersion`: stdev of devigged-home-prob across books, in
  probability points. A wide dispersion (say > 1.5pp) flags an unsettled
  market.
- `sharp_minus_square`: devigged-home-prob (sharp consensus) minus
  devigged-home-prob (square consensus). Positive = sharps like home more
  than squares -- if our pick is on home, this is corroboration; on away,
  caution.

**Use:**
- In confidence: bump `data_completeness` by a small amount when sharp
  agreement with our side; reduce when sharp disagreement.
- In UI: market intelligence section shows the dispersion + sharp-square
  number, with a clear "sharps agree" / "sharps disagree" chip.
- In sizing: reduce bet tier from Strong -> Standard when sharp-square
  > 1pp against our side (we'd be fading sharps).

**Fallback:** if config has no sharp books (user's current state), skip
all of this with `sharp_intel = None` and no UI section. Engine works
fine; this is additive.

**Validation:** does CLV on bets where sharps and our pick agree beat CLV
on bets where they disagree? If not, the signal isn't useful for us.

---

## (6) Run-environment / expected-runs modeling

**Status: deferred indefinitely until there's a dedicated validated effort.**

This isn't an upgrade -- it's a model replacement. The current model is
a transparent weighted-sum of probability deltas. CLAUDE.md states:
*"The model is a transparent weighted sum, not ML... Don't replace this
with a black box -- transparency is a product requirement."*

A runs-distribution model:
1. Projects RS and RA per team per game,
2. Assumes a runs distribution (negative binomial / empirical bootstrap),
3. Derives win probability from the joint distribution.

It would produce a single opaque win probability. The user could no longer
see "Tigers favored because their starter is 0.18 runs better" -- they'd
only see a number. Every existing UI breakdown would have to be reworked
to surface RS/RA components instead of the current win-prob deltas.

If we ever do this, the right approach is:
1. Implement it side-by-side with the current model behind a config flag.
2. Backtest both against the same historical slates.
3. Compare CLV, hit rate, calibration plots (predicted vs realized win%).
4. Only switch over if the new model is calibrationally better AND we can
   produce a transparent breakdown that preserves the product contract.

Time estimate: 2-3 dedicated sessions of build + 1 of backtest validation.

A scoped middle-ground that DOES fit the current paradigm: surface a
projected score and total lean as ADDITIONAL outputs (not replacing win
prob). Estimate RS/RA via:
  team_rs = league_rpg * (team_offense_wrc+ / 100) * (1/opp_starter_quality)
  team_ra = league_rpg * (opp_offense_wrc+ / 100) * (1/team_starter_quality)
Display "Projected score: 4.8 - 3.6" alongside the win prob. This is a
useful display feature without changing the model. Worth doing once the
form/bullpen/lineup upgrades are in.
