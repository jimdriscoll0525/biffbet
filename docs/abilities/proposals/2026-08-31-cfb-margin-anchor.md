# Ability: cfb-margin-anchor

**Status line:** proposed 2026-08-31 (after the Sep 3-5 board produced 0 spread picks), backtest-validated same day, **approved by Jim 2026-08-31 ("implement it")**, active from the same day's engine commit.

## Hypothesis
Anchoring the CFB margin projection on a points-denominated power rating (pregame Elo), with the EPA matchup signal as a bounded tilt, removes the structural big-spread dog bias and produces spread picks whose CLV vs the sharp close is at worst neutral.

## Mechanism
Per-play strength stats saturate long before blowout territory: the EPA-only margin spans ~±12 pts while CFB spreads reach 38+, so on every big spread the model "disagreed" by 8-33 pts — always on the dog side — manufacturing fake EV (UMass +29.5 showed +22% EV off a 20-pt scale artifact). Elo/SP+ are denominated in points, so a rating gap IS an expected margin and does not saturate. Shape mirrors the engine's proven pattern: anchor + clamped tilt (totals model), market anchor + bounded model tilt (MLB).

## Evidence (2025 full-season backtest, 16 weekly commit snapshots, real closes)
| CFB spread book | picks | W-L-P | ROI* | avg CLV (line-pts) | dog share | 14-28pt band |
|---|---|---|---|---|---|---|
| current (EPA-only) | 74 | 41-32-1 | +7.4% | **-0.23** | 93% | 3-3, -4.8% |
| **Elo-anchored** | 86 | 47-38-1 | +5.4% | **+0.02** | 30% | 12-8, +14.4%, CLV +0.28 |

*ROI look-ahead-inflated (full-season stats); CLV honest. Current model's profit sat entirely in |line|<=3.5 (13-3, +55%, CLV -0.41 = luck); anchored book is monotone in spread size. Totals identical under both. Benchmark vs actual 2025 margins: engine shape corr 0.05 / sd 5.0; Elo+HFA corr 0.61 / sd 12.5 / MAE 4.8 vs close (as-of-kickoff, no look-ahead); actual margin sd 20.4.

## Change
`config_football.yaml -> projections.margin_anchor_cfb: "elo"`, `elo_points_per_margin: 23.0` (fitted 2025), `margin_tilt_max_cfb: 6.0`. Read in `pipeline_football._power_margin_for` (pregame Elo from the matched CFBD schedule row) and applied in `projections.project_game`: `margin = elo_diff/23 + HFA + clamp(EPA margin, ±6)`. Totals untouched; NFL untouched (its EPA margins already match its spread universe: sd 6.0 vs 6.2). Games without Elo (new FBS programs) fall back to EPA-only. Paper-only like all football.

## Kill criteria (pre-registered)
Set `margin_anchor_cfb: null` if, after **40 graded CFB spread paper picks** post-activation, their avg `clv_pp` < 0. Review at 40 regardless. The `fb-spread-holds` retro lens keeps measuring what the divergence guard still holds.

## Expected effect
CFB spread picks exist at all (~5-6/week in backtest); balanced sides; divergence guard fires on genuine disagreements (2025: 490 -> 485 holds).
