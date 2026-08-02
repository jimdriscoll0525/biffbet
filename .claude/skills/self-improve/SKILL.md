---
name: self-improve
description: >
  Weekly self-improvement retro for the BiffBet engine. Mines the full
  recommendation history — settled bets AND the low-EV games it passed on —
  for durable trends, tracks them through a watch -> proposed -> active
  lifecycle in docs/abilities/, and turns proven trends into ability
  proposals for Jim's approval. Use when asked to run the retro, review
  trends, propose/approve/retire an ability, or when invoked as /self-improve.
---

# BiffBet self-improvement retro

You are running the engine's learning loop. The engine's whole ethos
(CLAUDE.md) is *measure edge honestly, stay transparent, never trade on
noise*. This skill exists to find real, durable trends in the pick history
and convert them into **abilities** — small, config-driven, reversible
changes to how the engine bets. It must never become a machine for
overfitting last week's variance. When in doubt, do nothing: "no candidates
this week" is a successful run.

## Definitions

- **Trend**: a segment of history (e.g. `ml-bets|stability=fragile`) whose
  results deviate from breakeven strongly enough to clear the evidence bar.
- **Ability**: a concrete engine change derived from a trend — always
  expressed as `config.yaml` keys (per CLAUDE.md design decision #2), never
  hardcoded logic, always with kill criteria. Examples: a new `adjusted_ev`
  haircut/boost, a `bet_sizing` gate, a `filters` rule, a paper-mode
  threshold loosening.
- **Ledger**: `docs/abilities/ledger.json` — the persistent memory of every
  trend ever watched, proposed, activated, rejected, or retired. Read it
  before doing anything; it is what makes this loop cumulative instead of
  amnesiac.

## Workflow

### 1. Collect (deterministic)

From the repo root, run the analysis script with the venv Python:

```powershell
.venv\Scripts\python .claude\skills\self-improve\scripts\retro_analysis.py
```

It writes `storage/retro/retro_report_<date>.json` + `.md`. It is read-only
against the engine DB and grades passed games counterfactually via the MLB
Stats API (cached in `storage/retro/pass_grades.json`). If the network is
unavailable, rerun with `--skip-pass-grading` and note in your summary that
pass counterfactuals are stale. Match `--ev-threshold` to `config.yaml ->
ev.threshold` if Jim has changed it from 0.03.

### 2. Reconcile against the ledger

Read the newest JSON report and `docs/abilities/ledger.json`. For every
candidate in the report, match on `key`:

- **Not in ledger** → add it with `status: "watch"`, `consecutive_hits: 1`,
  and this run's evidence appended to `evidence[]`.
- **In ledger as `watch`** and it cleared the bar again with a *larger*
  settled N → increment `consecutive_hits`, append evidence. If it now meets
  the promotion bar (below), promote to `proposed` and write a proposal doc.
- **In ledger as `watch`** but absent from this run's candidates → increment
  `consecutive_misses` (reset `consecutive_hits`). At 2 consecutive misses,
  set `status: "dropped"` (keep the entry — dropped trends that come back
  are themselves information).
- **In ledger as `rejected`/`dropped`** and it reappears → append evidence
  and note it in your summary, but do NOT re-propose unless Jim asks or the
  evidence is now Bonferroni-significant with materially more data.

Also update every `active` ability (step 5).

### 3. Promotion bar — ALL of these, no exceptions

A watch item becomes a proposal only when every condition holds:

1. **Persistence**: cleared the evidence bar in >= 2 *consecutive* weekly
   runs, with settled N growing between them.
2. **Significance**: `bonferroni_significant: true` in at least one run
   (the report tells you how many cells were tested).
3. **CLV discipline**: for bet-pool trends, `clv_agrees: true`. Pass-pool
   trends have no CLV, so they may only ever propose **paper-mode**
   abilities (e.g. a papered threshold loosening tracked like the totals
   model) — real-money changes from pass data alone are forbidden.
4. **Mechanism**: you can articulate a plausible baseball or market-
   structure reason the trend exists (e.g. "small dogs near the threshold
   are systematically underpriced by square books"). Write it down in the
   proposal. Calendar-only trends (`month=...`) NEVER promote — they are
   context for other trends, not abilities.
5. **Expressible in config**: the ability can be implemented as config keys
   plus a small, transparent read of them — consistent with the "no black
   box" rule. If it can't, it's a research note, not an ability.

### 4. Write the proposal (never implement in the same breath)

Create `docs/abilities/proposals/<YYYY-MM-DD>-<slug>.md` containing:

- **Hypothesis** — one sentence, falsifiable.
- **Mechanism** — why this should exist in the world, not just in the data.
- **Evidence** — the per-run stats table from `evidence[]` (record, flat
  ROI, t, p, CLV agreement, N growth), plus the cells-tested caveat.
- **Proposed change** — exact `config.yaml` keys and values, which module
  reads them, expected effect size, and whether it starts in paper mode.
- **Kill criteria** — the pre-registered auto-retire rule, e.g. "retire if
  after 40 settled post-activation games the segment's flat ROI is negative
  or avg CLV < 0". Every ability MUST have one before activation.
- **Status line** — `proposed <date>, awaiting approval`.

Update the ledger entry: `status: "proposed"`, `proposal_doc: <path>`.

Then STOP. Present the proposal(s) to Jim in your summary. Do not edit
`config.yaml`, `pipeline.py`, or any model code in the same run — approval
is a human step, always.

### 5. On approval (a later session, when Jim says "approve <ability>")

Implement it the engine's way: add the config keys, read them in the
narrowest sensible place (`_compute_adjusted_ev`, sizing, or filters), keep
the transparent-components ethos, add/extend tests per CLAUDE.md ("add to
tests when touching ev_calculator or win_probability"), and run `pytest`.
Record in the ledger: `status: "active"`, `activated: <date>`,
`config_keys: [...]`, `baseline` (the segment's stats at activation), and
the kill criteria verbatim. Note the change in the proposal doc.

### 6. Evaluate active abilities (every run)

For each `active` ledger entry, compute the segment's performance on games
settled AFTER activation (the report's tables give you this once the
activation date is in the ledger — filter `evidence[]` runs or rerun the
script with `--since <activation-date>`). If the kill criteria are met,
set `status: "retire-recommended"` and tell Jim; retire (revert config to
neutral values) only with his approval. An ability that survives its first
40 post-activation games gets a note saying so — that is the real win
condition of this whole loop.

### 7. Summarize

End with a short plain-language summary: rows analyzed (bets/passes, both
markets), candidates found, ledger transitions (new watches, promotions,
drops), proposals awaiting approval, active-ability health, and the single
most interesting thing in the data this week even if it cleared no bar.
Keep it honest — small N is small N.

## Hard rules

- Never mutate the engine DB. The script doesn't; neither do you.
- Never activate, tune, or retire an ability without Jim's explicit
  approval in the conversation.
- Never promote a trend on one run's evidence, however significant.
- Never let the pass-pool counterfactuals (no CLV, refreshed-snapshot
  caveat) justify a real-money change on their own.
- Prefer fewer, better abilities. If two candidates overlap (e.g.
  `side_type=underdog` and `odds_bucket=small dog`), treat them as ONE
  trend and pick the sharper framing.
