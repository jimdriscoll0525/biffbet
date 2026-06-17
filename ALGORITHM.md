# BiffBet — the algorithm

A plain-English + math walkthrough of how BiffBet finds +EV MLB moneyline bets.
It is **starting-pitcher focused** and built around one principle: *measure
whether the model has a real edge* (via disciplined expected-value accounting
and closing-line-value tracking), not maximize raw win rate.

This document is self-contained — you don't need the source code to follow it.
It contains no API keys, credentials, or private data.

---

## 0. The one-paragraph version

For each game we build a **transparent win-probability model** — a base rate
plus a handful of signed, weighted adjustments (starter, bullpen, bullpen
fatigue, lineup, park, home field, recent form). We then **blend that model
toward the de-vigged betting market** (the sharp consensus is the smartest
public probability there is), so the model only ever expresses a *bounded tilt*
off the market rather than replacing it. From the blended probability and the
price we compute **expected value** and a **fractional-Kelly stake**. A series
of **sanity guards** throws out games where the market clearly knows something
we don't, and a set of **context haircuts + sizing tiers** decides how much (if
anything) to actually bet. Everything is logged so we can grade it honestly,
and the real scoreboard is **closing-line value (CLV)**, not the win/loss
record.

---

## 1. Data sources

| Input | Source | Notes |
|---|---|---|
| Moneyline odds | The Odds API (v4 `h2h`, American) | We take the **best price per side** across configured books, and separately track **sharp** books (Pinnacle, BetOnline, LowVig) vs **square** books (DraftKings, FanDuel, BetMGM). |
| Schedule, probable pitchers, final scores | MLB Stats API | `gamePk` is the canonical game id. No key required. |
| Pitcher rate stats (xFIP, SIERA, K-BB%, FIP, bullpen FIP, wRC+) | FanGraphs leaderboards | Sits behind a Cloudflare challenge, so these come from browser-exported CSVs when available. |
| Pitch-level Statcast (Whiff%, CSW%, xwOBA-on-contact) | Baseball Savant | Drives the starter rate when FanGraphs is unavailable, and the recent-form component. |
| Bullpen availability | MLB Stats API boxscores | Who pitched in each team's last few games → leverage-arm fatigue. |
| Confirmed lineups | MLB Stats API game feed | Posted ~2–3h before first pitch; before that, lineups are "projected." |

**Graceful degradation is a hard rule.** A missing pitcher, postponed game,
rate-limited API, or absent stat never crashes the slate — the affected
component contributes 0 and the confidence score drops instead.

---

## 2. The win-probability model

The model produces a **home win probability** as a base rate plus signed
adjustments ("deltas"), each scaled by a configurable weight. **Positive always
favors the home team.** Every component is reported back with its value so you
can see exactly why a side is favored.

```
home_win_prob = base_wp
              + w_starter         · starter_delta
              + w_bullpen         · bullpen_delta
              + w_bullpen_fatigue · bullpen_fatigue_delta
              + w_lineup          · lineup_delta
              + w_park            · park_delta
              + w_home_field      · home_field_delta
              + w_form            · form_delta
        (then clamped to [0.02, 0.98])
```

### 2.1 Base rate — team strength via log5

Each team's season win% is **regressed toward .500** to tame small early-season
samples:

```
regressed_winpct = (wins + k·0.5) / (games + k)          # k = 30 phantom .500 games
```

The two regressed win%s are combined with the **log5** formula (the standard way
to get "probability A beats B" from two teams' strength vs an average team):

```
log5(A, B) = (A − A·B) / (A + B − 2·A·B)
base_wp    = log5(home_regressed, away_regressed)
```

### 2.2 Starter — the headline component

Each starter's run-prevention rate (preferring **xFIP**, then SIERA, then a
Statcast-derived rate, then FIP, then ERA) is converted to a win-equivalent vs a
league-average pitcher:

```
pitcher_wp = 0.5 + (league_avg_rate − pitcher_rate) · 0.065     # clamped to [0.30, 0.70]
```

(Better/lower rate → higher win%. The 0.065 converts ~1 run/9 into ~6.5 win%
points.) The two starters are then run through log5 and centered:

```
starter_delta = clamp( log5(home_pitcher_wp, away_pitcher_wp) − 0.5,  ±0.15 )
```

The ±0.15 clamp matters: without it, an extreme matchup on a thin sample could
dominate the whole model.

> **Statcast fallback for the starter rate.** When FanGraphs xFIP/SIERA isn't
> available, we derive a rate from full xwOBA-against on the same ~4.00 scale:
> `rate = 4.00 + (xwoba_against − 0.315) · 40`. This is regressed toward league
> average by batters faced (until ~250 PA) so a few hot/cold starts don't read
> as ace/replacement level.

### 2.3 Bullpen (season FIP)

```
diff           = away_bullpen_FIP − home_bullpen_FIP        # + favors home
bullpen_delta  = clamp( diff · 0.022,  ±0.10 )
```

Bullpens throw ~1/3 of innings, so the per-run effect is smaller than a
starter's. (Bullpen FIP is derived as the IP-weighted FIP of each team's
relievers.)

### 2.4 Bullpen fatigue (leverage-arm availability today)

For each team we identify its **top-3 leverage arms** (lowest season ERA, with a
minimum-appearances floor) and check whether they're available today — a
reliever who threw 35+ pitches yesterday, or appeared in 3 of the last 3 days, is
"unavailable." The penalty is **non-linear** (one arm down barely matters; a
gassed pen is a real signal):

| Leverage arms unavailable | Per-team penalty |
|---|---|
| 0 of 3 | 0.0% |
| 1 of 3 | −0.5% |
| 2 of 3 | −1.5% |
| 3 of 3 | −3.0% |

```
bullpen_fatigue_delta = clamp( home_penalty − away_penalty,  ±0.03 )
```

If either team's availability feed is missing, this contributes 0 (we don't
fabricate a tilt from partial data) and the confidence score takes a small hit
instead.

### 2.5 Lineup (confirmed key bats in/out)

Once **both** lineups are confirmed, we count each team's missing "key bats"
(top-3 by season OPS, with a plate-appearance floor) and tilt toward the team
with fewer absences:

```
lineup_delta = clamp( (away_missing − home_missing) · 0.005,  ±0.02 )
```

Before lineups post, this is 0 — the uncertainty is carried by a graded
**confidence penalty** (−3 if one side is projected, −5 if both, −6 if a feed is
genuinely unavailable), not by a fake probability move.

### 2.6 Park / offense

A small adjustment for how the ballpark's run environment interacts with the
offensive gap between the teams (mostly a totals factor, kept deliberately low):

```
off_diff   = (home_wRC+ − away_wRC+) / 100
park_dev   = (park_factor − 100) / 100
park_delta = clamp( off_diff · park_dev · 0.5,  ±0.05 )      # then weighted ·0.25
```

### 2.7 Home field

A park-specific constant added to the home side (Coors Field is the largest at
+0.045; the default is +0.025).

### 2.8 Recent form (pitching)

A blend of each starter's recent **xwOBA-on-contact** over three windows —
**14-day (50%) / 30-day (30%) / season (20%)** — with weights renormalized when
a window lacks sample. The home-vs-away difference becomes the delta:

```
form_delta = clamp( (away_blended_xwOBAcon − home_blended_xwOBAcon) · 0.5,  cap )
```

The cap is tight because short-window hitting stats are mostly noise (14 days ≈
50 batted balls): **±1.5% normally**, widening to **±2.5% only** when the 14d and
30d windows agree directionally *and* both pitchers clear a sample floor. If the
14-day window dominates the blend (≥60%), the component is flagged **fragile**,
which feeds the stability classifier below.

---

## 3. Market anchoring (the most important design choice)

A standalone heuristic "finds" an edge on nearly every game (we observed +EV on
13/13 at up to +44% before adding this). The fix: blend the raw model toward the
**de-vigged market**, which is the sharpest public probability estimate.

```
final_prob = blend · model_prob + (1 − blend) · market_devigged_prob
```

- **De-vigging** removes the bookmaker's margin so the two sides sum to 1. We use
  the **power method** (find exponent *k* such that `Σ pᵢᵏ = 1`), which preserves
  the favorite-longshot structure better than simple normalization.
- **`blend` is data-driven, not fixed.** It's chosen from a tier table keyed on
  **data confidence** (data completeness + sample size + component agreement —
  *not* EV, deliberately, to avoid a feedback loop where the model talks itself
  into bigger edges):

  | Data confidence | Model weight (`blend`) |
  |---|---|
  | ≥ 85 (high) | 0.45 |
  | 70–85 (mid) | 0.35 |
  | < 70 (low) | 0.25 |

  Even the high-confidence tier keeps the market as the **primary anchor**
  (≤ 0.5). The model earns more weight only once CLV proves out. *If output ever
  looks too edgy, lower the blend.*

The de-vigged market probability and the raw model are both stored so every pick
shows its full breakdown.

---

## 4. Expected value & Kelly staking

All P/L is tracked as a **fraction of bankroll** (bankroll-independent).

```
decimal_odds = american_to_decimal(price)
EV%          = final_prob · decimal_odds − 1

# Fractional Kelly:
b      = decimal_odds − 1
full_k = (b · final_prob − (1 − final_prob)) / b
stake  = min( full_k · 0.25,  0.02 )      # quarter-Kelly, capped at 2% of bankroll
```

A bet is flagged **+EV** when `EV% ≥ 3%` and the Kelly stake is positive. (3%,
not 5% — a higher bar filters out almost everything early in the season.)

---

## 5. Sanity guards (reject before a fake edge can ship)

A game is **skipped** (never saved, ranked, or shown) when any of these fire —
these bounds sit far outside real MLB markets, so they catch bad/stale data
without rejecting legitimate games:

1. **Implausible odds** — either side beyond ±800.
2. **Model-vs-market divergence** — raw model and de-vigged market disagree by
   **> 15 probability points**. This is the late-scratch catcher: when a starter
   is pulled, the market moves 25–30pp while our probable-pitcher data is still on
   the announced starter, manufacturing a fake edge. (Real production incident:
   Red Sox/Braves flipping −149 → +295 in minutes.)
3. **Sharp fade** — if our pick side disagrees with the **sharp consensus** by
   **> 4 probability points**, skip. The sharps are the smartest counterparty on
   the board; betting hard against them means we're claiming to know more.
4. **Implausible EV** — any surviving EV above 30% is treated as a data error.

---

## 6. Adjusted EV (context the raw number can't see)

"Raw EV" (model vs price) is displayed unchanged. **Adjusted EV** applies a few
small, signed haircuts/boosts and is what the sizing tiers read:

- **Sharp support** (sharps agree with us by ≥3pp): **+1.0pp**
- **Sharp fade** (we're more bullish than sharps by ≥3pp): **−1.0pp**, or
  **−2.0pp** past 5pp (above the skip threshold this is moot).
- **Fragile edge** (see §7): **−1.0pp**

These each count **exactly once** — the sharp signal lives only here, not also in
the confidence score or as a separate tier penalty.

---

## 7. Edge stability classification

Each pick is labeled **STABLE / MODERATE / FRAGILE** based on *which* components
drive the edge:

- **Stable drivers**: starter, bullpen, confirmed lineup.
- **Fragile drivers**: form flagged as 14d-dominated, components with missing
  data.
- **Hard fragile signals**: fading the sharps by ≥3pp, or ≥2 model components
  missing data.

A pick is **STABLE** when ≥60% of the pick-aligned drive comes from stable
drivers and no hard signal fires; **FRAGILE** when a hard signal fires or ≥50% of
the drive is fragile. There is one inviolable rule downstream: **never size a
fragile edge as "Strong."**

---

## 8. Bet sizing tiers

Selection and sizing are **decoupled**: a game is *selected* purely on raw EV
clearing the 3% threshold; the haircuts above then *size* it. Tiers are read off
**Adjusted EV**:

| Adjusted EV | Tier | Kelly cap (fraction of bankroll) |
|---|---|---|
| < 2% | Pass | 0% (no action) |
| 2% – 5% | Small / Lean | 0.5% |
| 5% – 8% | Standard | 1.0% |
| ≥ 8% | Strong | 2.0% (flag for manual review) |

The stake is `min(quarter-Kelly, tier cap)`. The tier is **downgraded one step**
if confidence < 60, or (the hard rule) if the band is Strong but the edge is
fragile. The caps are deliberately tight — the model is unproven on a small
graded sample, so we cap hard and downgrade on any quality flag.

---

## 9. Confidence score (0–100)

A weighted average of four normalized sub-scores, shown on each pick:

| Sub-score | Weight | What it measures |
|---|---|---|
| Data completeness | 0.30 | Did we get both pitchers + team data? |
| Sample size | 0.20 | Weakest-link pitcher IP vs a 60-IP "trustworthy" line |
| Edge magnitude | 0.25 | Bigger measured EV → (weakly) more confidence |
| Component agreement | 0.25 | Do the skill components all lean the same way? |

Missing-data penalties (projected lineup, unavailable bullpen feed) are
subtracted on top. Note the deliberate split: **data confidence** (the first,
second, and fourth terms — *excluding* EV) is what drives the market blend in §3;
the full score including EV is only ever *displayed*, never fed back into the
blend.

---

## 10. Measuring the edge (the actual point)

- **CLV (closing-line value)** is the primary signal. With no historical-odds
  feed, we capture it by re-pricing: record the bet price, then run again near
  first pitch and compare. `CLV% = decimal(opening) / decimal(closing) − 1`
  (positive = we beat the close). At small samples CLV is a far better signal of
  real edge than win rate.
- **Two ROI views** are tracked: Kelly ROI (how the bankroll actually moved) and
  flat 1-unit ROI (comparable across staking schemes).
- **Stratified performance**: every result is sliced by edge stability, sharp
  fade, odds bucket, confidence, EV bucket, favorite/underdog, home/road, and
  CLV sign — so a leak shows up as a *bucket*, not just an overall number.
- **Self-healing bookkeeping**: grading sweeps every past date with open bets (a
  missed grade retries until it settles), and committed picks keep getting their
  closing line refreshed even on games the sanity guards later skip.
- **Honest record-keeping**: a committed pick is never retroactively deleted or
  downgraded when the rules change — it was published at a real price and is
  graded as such. Late starter scratches on committed picks are flagged, not
  hidden.

---

## 11. Reference — current weights & thresholds

Everything below is config-driven and reloaded every run (no magic numbers in
code). Current values:

```
Model weights:          starter 1.0 · bullpen 1.0 · bullpen_fatigue 1.0 ·
                        lineup 1.0 · park 0.25 · home_field 1.0 · form 1.0
Team regression:        30 phantom .500 games
Pitcher run→win%:       0.065 per run/9      (starter clamp ±0.15)
Bullpen run→win%:       0.022 per run/9      (clamp ±0.10)
Bullpen-fatigue clamp:  ±0.03
Lineup:                 0.005 per missing key bat   (clamp ±0.02)
Form scale:             0.5    (caps: ±1.5% normal, ±2.5% extreme)
Form windows:           14d 50% / 30d 30% / season 20%
Market blend:           0.45 high / 0.35 mid / 0.25 low   (conf thresholds 85 / 70)
Statcast regression:    250 PA
Home field:             0.025 default (Coors 0.045, Fenway 0.030, …)
Prob clamp:             [0.02, 0.98]

EV threshold:           3%        De-vig: power method
Kelly:                  quarter-Kelly, cap 2% of bankroll
Sizing tiers:           small ≥2% (cap 0.5%) · standard ≥5% (1.0%) · strong ≥8% (2.0%)
Tier downgrade:         confidence < 60

Sanity skips:           |odds| > 800 · model-market divergence > 15pp ·
                        sharp fade > 4pp · EV > 30%
Adjusted-EV haircuts:   sharp support +1pp · sharp fade −1pp (−2pp past 5pp) ·
                        fragile −1pp
```

---

## 12. What it is *not* (limitations, stated honestly)

- **Not machine learning.** It's a transparent weighted sum on purpose —
  transparency is a product requirement, not a limitation we plan to "fix."
- The recent-form component is **pitching-only** (rolling team-offense form is a
  planned addition).
- xwOBA-on-contact is a proxy, not full plate-appearance-weighted xwOBA-against.
- The backtester uses full-season stats → look-ahead bias on ROI (CLV is fine).
- It's a **personal research tool focused on measuring edge**, not a tout
  service. The honest scoreboard — CLV and disciplined EV accounting — is the
  whole point.
```
