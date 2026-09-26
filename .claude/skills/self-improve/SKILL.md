---
name: self-improve
description: >
  Run or inspect the BiffBet RSI (recursive self-improvement) loop: the
  weekly review that mines every engine's pick AND pass history for
  segment trends, writes Proposals to Supabase for Jim's approval, tracks
  approved changes as shadow challengers, and flags underperforming
  promoted versions. Use when asked to run the review, look at proposals,
  explain a finding, or when invoked as /self-improve. Decisions happen in
  the Proposals tab on biffbet.com, never in this skill.
---

# BiffBet self-improvement (RSI)

Since 2026-09-26 the learning loop is code, not a checklist: the `rsi/`
package (see CLAUDE.md "RSI") runs every Monday in GitHub Actions
(`scope=rsi`) and the decisions live in the Proposals tab at
https://biffbet.com/proposals. This skill is for running or reading the
loop by hand. The ethos is unchanged: *measure edge honestly, stay
transparent, never trade on noise*. "No candidates this week" is a
successful run.

## What the loop does (so you can explain it)

1. **Review** (`rsi/review.py`): reads the cross-engine view
   `rsi_scored_games` (every scored game, picks and passes, all four engines),
   drops the 20% holdout, prepares the segment buckets (`rsi/segments.py`,
   the single source of bucket definitions — `tracking/performance.py`
   delegates to it), and computes per-cell stats (`rsi/stats.py`). A cell is
   a candidate when it has at least `min_sample[sport]` settled rows (75 MLB,
   40 football, in `rsi/config_rsi.yaml`) and either a |t| >= 2 on flat P/L
   or a strong CLV sign. The pre-registered small-favorites test runs first
   and is exempt from Bonferroni. CLV first, win/loss second, always in the
   engine's own unit (`clv_metric`: clv_pct for MLB ML, clv_pp for totals and
   football, clv_blended_vs_sharp for GriffBet — never mixed).
2. **Reconcile** (`rsi/reconcile.py`): new finding -> `watch`; two consecutive
   runs with settled N growing AND (Bonferroni-significant OR CLV agrees) ->
   `pending`; two misses -> `dropped`; calendar-only dims (`month`) never
   leave `watch`; rejected/dropped/promoted findings that reappear only gain
   evidence. `rsi/suggest.py` attaches a plain-English description and, when a
   rule exists, a config **overlay** (dotted keys, e.g.
   `{"filters.min_model_prob": 0.5}`); otherwise the proposal is an `insight`
   Jim can edit into an overlay in the tab.
3. **Shadow** (`rsi/shadow.py`, run inside every `today`): approved proposals
   are re-evaluated on the SAME fetched slate with the overlay applied and
   written to `rsi_shadow_picks` (never the public tables).
   `rsi/shadow_stats.py` compares champion vs challenger and computes the
   promotion gate (challenger settled >= min_sample, challenger CLV > champion
   CLV, holdout agrees, window complete: 4 weeks MLB, 3 football).
4. **Versions** (`rsi/config.py`, `rsi/state.py`, `rsi/versions.py`): the live
   config is base YAML + the ACTIVE version's cumulative overlay from
   `rsi_model_versions`; every row is stamped with the version tag
   (biff_vN / totals_vN / matchup_vN). Rolling-window underperformance sets
   `rollback_flagged` and the tab offers one-click rollback. Nothing here
   changes status on its own.
5. **Email** (`rsi/report.py`, `rsi/email.py`): one plain-text Monday summary
   via Resend. No actions in the email.

## Running it by hand

From `C:\Users\jim` (the package imports from the parent dir):

```powershell
mlb_value_bot\.venv\Scripts\python -m mlb_value_bot.rsi review --sport all --dry-run --no-email
```

`--dry-run` computes everything, prints the per-sport summary and the
would-be proposal transitions, and writes nothing. Drop `--dry-run` to write
proposals/events/run rows (this is what the Monday job does). Other commands:
`state --engine mlb|mlb_totals|football` (effective tag + overlay + shadows),
`shadow-stats`, `versions`, `grade --engine mlb|football`, `email`,
`import-ledger PATH` (one-off migration of the old ledger, already done).

## Hard rules (unchanged)

- Never activate, tune, promote, roll back or retire anything from this
  skill or a chat. Approve/reject/snooze/promote/rollback are buttons in the
  Proposals tab and are recorded as permanent `rsi_proposal_events`.
- Never promote on one run's evidence. The reconcile lifecycle enforces
  persistence; do not hand-edit `rsi_proposals` to shortcut it.
- Pass-pool findings are counterfactual (graded, zero P/L, frozen opening but
  refreshed snapshot) and never justify a real-money change on their own.
- The holdout slice (`is_holdout` in the view, `rsi/holdout.py`) is for
  promotion validation only. Never use it for discovery.
- Prefer fewer, better proposals. Overlapping cells (e.g. `side_type=underdog`
  and `fav_size=dog +120..+149`) are ONE finding; reconcile folds them.

## History

The pre-RSI loop (`scripts/retro_analysis.py`, `docs/abilities/ledger.json`,
the "BiffBet weekly retro" Windows task, SMTP notifier) was retired on
2026-09-26; its 14 ledger entries were imported into `rsi_proposals` with
their evidence, and `docs/abilities/` is kept as frozen history.
