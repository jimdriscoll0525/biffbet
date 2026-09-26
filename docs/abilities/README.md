# Abilities — FROZEN history of the pre-RSI learning loop (to 2026-09-26)

This directory was the persistent memory of BiffBet's first self-improvement
loop (a Monday retro script + this JSON ledger + markdown proposal docs).
On 2026-09-26 that loop was replaced by the RSI system (`rsi/` package,
Supabase `rsi_*` tables, the Proposals tab on biffbet.com) — see CLAUDE.md
"RSI (Recursive Self-Improvement)" and `.claude/skills/self-improve/SKILL.md`.

- `ledger.json` — every trend the old retro tracked, with its evidence
  history and lifecycle status. All 14 entries were imported into
  `rsi_proposals` (`python -m mlb_value_bot.rsi import-ledger`): `active` ->
  `promoted` (attached to the seed versions biff_v1 / matchup_v1), `watch` ->
  `watch` with hits preserved, `rejected` / `dropped` as-is. Kept for the
  record; not updated any more. The source of truth is `rsi_proposals` +
  `rsi_proposal_events` in Supabase.
- `proposals/` — the original markdown proposals (hypothesis, mechanism,
  evidence, config change, kill criteria). Their text became the imported
  proposals' descriptions.

The rules that made the old loop safe carry over unchanged into RSI: every
change is a small, config-driven, reversible overlay; nothing goes live
without Jim's approval; CLV before win/loss; pass-pool counterfactuals never
justify a real-money change on their own.
