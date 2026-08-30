# Abilities — the engine's learning ledger

This directory is the persistent memory of BiffBet's self-improvement loop
(`/self-improve`, defined in `.claude/skills/self-improve/SKILL.md`).

- `ledger.json` — every trend the weekly retro has ever tracked, with its
  full evidence history and lifecycle status:

  `watch` -> `proposed` -> (Jim approves) -> `active` -> `retired`
  with `dropped` / `rejected` as off-ramps.

- `proposals/` — one markdown doc per proposed ability: hypothesis,
  mechanism, evidence, exact config change, and pre-registered kill
  criteria. Nothing goes live without a proposal doc and Jim's approval.

An **ability** is always a small, config-driven, reversible change
(`config.yaml` keys + a transparent read of them), never a black-box model
change — same rules as the rest of the engine (see CLAUDE.md).

The weekly data collection lives in
`.claude/skills/self-improve/scripts/retro_analysis.py`; its reports land in
`storage/retro/` (gitignored). This directory IS committed — the ledger and
proposals are part of the project's history.
