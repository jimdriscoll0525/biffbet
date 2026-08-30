# Ability: min-model-prob

**Status line:** proposed 2026-08-30, **approved by Jim 2026-08-30 ("approve proposal D")**, active from the same day's engine commit.

## Hypothesis
Moneyline picks whose *blended* probability on the picked side is below 50% lose money at every price, and removing them raises the book's ROI without hurting CLV.

## Mechanism
The blend is 35% model / 65% de-vigged market. When that blend still has our side as the likelier loser, the pick exists only because a long underdog price made raw EV positive against a small model tilt - a price artifact, not a directional view. The CLV on these bets is positive (+2.0%): the market agrees with our *direction* while the results don't follow, which is long-price variance a book this small can't carry.

## Evidence (Supabase pull 2026-08-30, 170 settled ML bets)
| cell | n | W-L | hit | b/e hit | flat ROI | t | avg CLV |
|---|---|---|---|---|---|---|---|
| blended p on pick < 50% (removed) | 24 | 6-18 | 25.0% | 46.1% | -46.0% | -2.36 | +2.03% |
| blended p >= 50% (kept) | 112 | 60-52 | 53.6% | - | -1.9% | -0.22 | +0.20% |
| all bets (baseline) | 170 | 76-94 | 44.7% | 50.1% | -10.9% | -1.39 | +1.15% |

Split check: kept book May-Jul +2.1% (n=53), August -5.6% (n=59) vs baseline -0.9% / -17.8%. Kelly bankroll impact -13.5% -> -4.3%.

Caveats: single-run evidence (not the 2-consecutive-weekly-runs bar - Jim approved directly); 127+ cells tested, so p~0.02 is not Bonferroni-significant; CLV on the removed cell is *positive*, so this is results-based, not CLV-based. Alternatives backtested and rejected: dog EV thresholds (>=5% dogs still 8-14), favorites-only (zero-CLV book), dispersion / narrow-price cells (overfit shapes).

## Change
`config.yaml -> filters.min_model_prob: 0.50`, read in `pipeline._selection_filters` (same mechanism as the heavy-favorite filter: tier -> pass, stake 0, `is_value=false`, reason string `"filtered: blend has pick at 47% (below min_model_prob 50%)"` shown on the site and bucketed by the retro as `blend below min_model_prob`). A real-money change that only *removes* bets. Grandfathering: `upsert_recommendation` never downgrades an already-committed bet.

## Kill criteria (pre-registered)
Retire (set the key to `null`) if, after **40 settled post-activation removed games** (counterfactually graded by the weekly retro under `filter_reason = blend below min_model_prob`), **either** their flat ROI is > 0 **or** the kept book's average CLV since activation is < 0.

## Expected effect
Roughly 14% fewer bets (24 of 170 historically), all underdogs, concentrated in +110...+149 and +150+.
