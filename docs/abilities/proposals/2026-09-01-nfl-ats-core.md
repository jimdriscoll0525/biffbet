# NFL ATS core (epa_rating anchor) — M6 gate verdict: NOT ACTIVATED

**Status line:** plan approved 2026-09-01, M1-M4 built and shipped same day, M6 gate **FAILED** 2026-09-01 -> `projections.margin_anchor_nfl` stays `null`. The module remains in the codebase (ratings CLI, weekly aggregates, matchup/luck layers) but prices no live board.

## What was built
EPA+success-rate power rating (garbage-time re-weighted, recency-decayable, prior-blended, walk-forward by construction) + conditional top-10-vs-bottom-10 matchup adjustments (rush/pass/OL/QB-pressure) + luck/regression fade-buy flags, wired as anchor+clamped-tilt behind a config gate.

## Tuning (2023-24, walk-forward, graded vs real closing spreads — $0)
Best family: epa/sr 0.7/0.3, gt_weight 0-0.2, prior_out_week 8, **decay 1.0** (recency weighting added nothing in-sample). In-sample ATS at |edge|>=2: 52.2-52.9% (n~400). Matchup/luck layers: within noise (kept, bounded, for explainability).

## Holdout (2025, untouched by tuning)
| test | result |
|---|---|
| pure margin, 272 games | corr 0.815 with close, MAE 5.17 |
| ATS at edge >= 1 / 2 / 3 / 4 pts | 50.4% / 50.5% / **49.1% / 46.2%** (n=232/196/169/143) |
| real pipeline, cached weekly snapshots | 24 picks, 14-8-2, +22.1% ROI, **avg CLV +0.12 pts, CLV+ 29%** |

**Verdict.** The monotone DEGRADATION with edge size is the tell: the model's biggest disagreements with the market are its noisiest, i.e. no exploitable signal beyond the close. The 14-8-2 pipeline record is variance on a 50/50 margin distribution (n=24, CLV+ 29%). Gate required holdout CLV >= 0 AND top-tier ATS >= ~52%; the spirit of the gate fails decisively on the 196-game sample. The in-sample 52% was grid selection.

## What survives
- The rating is a strong market model (corr 0.82) — useful as a sanity/diagnostic board (`football ratings` CLI) and for future lenses.
- The divergence guard's behavior is vindicated AGAIN: the |edge|>=4 tail it holds is exactly where the model is worst (46%).
- Luck flags + QB pressure sensitivity exist for the retro to study against live data.

## Re-proposal bar
Only on live evidence: if the weekly retro's fb-spread-holds / fb-bets NFL cells show the anchored-margin disagreement band beating the sharp close with positive clv_pp on a real sample. No re-tuning on the same 2023-25 data.
