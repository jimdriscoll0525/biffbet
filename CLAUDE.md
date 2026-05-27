# CLAUDE.md — architecture notes for mlb_value_bot

Context for future Claude Code sessions. Read this before making changes.

## What this is

A personal, terminal-based **+EV moneyline finder for MLB**, starting-pitcher
focused, designed to be extended to NFL. **Primary goal: measure whether the
model has a real edge** (via disciplined EV accounting + CLV tracking), *not* to
maximize win rate. Optimize for transparency and correct bookkeeping over
predictive cleverness.

## Layering (strict, one job per layer)

```
cli.py  ─┐
         ├─> pipeline.py ─> analysis/* ─> data/* ─> (external APIs / cache)
tracking/* (DB) <─ cli.py
backtest/* reuses pipeline.evaluate_game
```

- **data/** — the *only* place that does external I/O. `odds_client` (The Odds
  API), `mlb_client` (MLB Stats API), `cache` (parquet TTL). All HTTP is wrapped
  in `tenacity` retry; `cache.cached_dataframe` falls back to a *stale* copy if a
  producer raises, so an API blip doesn't kill a run.
- **analysis/** — computation, mostly pure. `ev_calculator` is 100% pure stdlib
  math (so it's trivially testable and is the thing you least want to get wrong).
  `pitcher_metrics`/`team_metrics` wrap pybaseball; `win_probability` is the model.
- **tracking/** — durable SQLite record + reporting. Schema lives in
  `recommendations.py::_SCHEMA` and matches the spec exactly.
- **pipeline.py** — orchestration glue so `today` and `backtest` share the
  "match odds ↔ schedule, build metrics, run model, compute EV" flow. Also owns
  `save_value_bets()` (shared by the CLI and the web app — don't re-implement it).
- **web/app.py** — Streamlit viewer (`python -m mlb_value_bot serve`, which shells
  out to `streamlit run`). Three pages (Today/Results/Performance) reuse
  `analyze_slate`, `save_value_bets`, and the tracking/performance modules — it's
  a thin view layer with NO business logic of its own. Reads the DB by default;
  only the "Run analysis" button calls the metered pipeline. Inserts the repo
  root on `sys.path` at import so `streamlit run` resolves `mlb_value_bot`.
  Verify changes headlessly with `streamlit.testing.v1.AppTest` (see how the
  smoke test runs each page and checks `at.exception`).

## Key design decisions

1. **The model is a transparent weighted sum, not ML.** `win_probability.py`:
   ```
   home_wp = base_wp(log5 of regressed team win%)
           + Σ weight_i * delta_i   for i in {starter, bullpen, park, home_field, form}
   ```
   Every `delta_i` is a signed probability contribution where **positive favors
   home**, and every weight is in `config.yaml → model.weights`. The result
   carries a `Component` list so the CLI can print exactly why a side is favored.
   *Don't replace this with a black box* — transparency is a product requirement.

1b. **Market anchoring (calibration) — critical.** The raw model is then blended
   toward the de-vigged market in `pipeline.evaluate_game`:
   `final = market_blend*model + (1-market_blend)*market_devig` (default 0.35).
   This is NOT optional polish: a standalone heuristic "finds" +EV on ~every game
   (we observed +EV on 13/13 at up to +44% before adding it; ~5/13 at +3-8% after).
   The de-vigged market is the sharp baseline; the model expresses a *bounded
   tilt*. EV/Kelly use the blended prob; the raw model + blend are stored in
   `reasoning_json` and shown in the breakdown. Also: `statcast_rate` is regressed
   toward league avg by batters faced (`model.statcast_regression_pa`) so small
   samples don't hit the 2.0 floor. If output looks too edgy, LOWER `market_blend`.

2. **Config-driven, reloaded every run.** `utils.load_config()` is `lru_cache`d
   per-process. All thresholds/weights/park factors live in `config.yaml`. Adding
   a tunable = add a key there + read it; do not hardcode magic numbers in code.

3. **Graceful degradation everywhere.** pybaseball missing, no probable pitcher,
   postponed game, API outage, rate limit — each is handled by lowering data
   completeness / confidence or skipping with a reason, never crashing the slate.
   `PitcherProfile`/`TeamProfile` are valid even when empty.

4. **EV/units convention.** `kelly_stake` and `profit_loss` are **fractions of
   bankroll** (bankroll-independent). The CLI multiplies by `BANKROLL` only for
   display. `performance.py` also derives a flat (1u) ROI from stored odds+result.

5. **CLV via re-pricing.** There's no historical-odds source on the free tier, so
   CLV is captured by running `today` again near first pitch: `upsert_recommendation`
   keeps the original bet price/`opening_line` and updates `closing_line` +
   `clv_pct`. CLV% = `decimal(opening)/decimal(closing) − 1` (positive = beat close).

6. **Team-name normalization is centralized** in `constants.py::normalize_team`
   (handles Athletics relocation, abbreviations, historical names). Both API
   clients and FanGraphs matching route through it. `team_metrics._FG_ABBR` maps
   FanGraphs abbreviations.

## Data source specifics / gotchas

- **The Odds API:** v4 `/sports/baseball_mlb/odds`, `markets=h2h`,
  `oddsFormat=american`. We pick the **best (highest) American price** per side
  across configured books. Quota headers `x-requests-remaining`/`-used` are logged.
- **MLB Stats API:** `/v1/schedule?hydrate=probablePitcher,linescore,team,venue`
  for slate; `/v1/standings` for win%; final scores from the schedule linescore.
  `gamePk` is the canonical `game_id`. No key needed.
- **FanGraphs 403 (IMPORTANT).** pybaseball's FanGraphs scrapers (`pitching_stats`,
  `team_batting`) hit `leaders-legacy.aspx`, which Cloudflare guards with a **JS
  challenge** ("Just a moment…", `server: cloudflare`) → **403** for every
  scraper. Confirmed from a residential IP even with a browser UA, so it's NOT an
  IP ban and NOT a header fix; only a real browser solves it. So xFIP/SIERA/wRC+/
  bullpen-FIP can be unavailable. Handling:
  - **Primary path: `data/fangraphs_csv.py`** reads browser-exported leaderboard
    CSVs from `storage/fangraphs/` (`pitching_{season}.csv`, `batting_{season}.csv`).
    `get_season_pitching` + `team_metrics._team_batting` **prefer the CSV**;
    `_reliever_leaderboard` routes through `get_season_pitching`, so one pitching
    CSV feeds both starter rates and bullpen FIP. CSV is normalized to pybaseball's
    column shape (handles "18.0%" strings, team abbrevs). `data-status` CLI command
    reports presence/columns. Config: `fangraphs_csv.*`.
  - `cache.cached_dataframe` returns an **empty frame** (not raise) on a hard
    producer failure → the run degrades instead of crashing.
  - `get_season_pitching` is `@lru_cache`d so FanGraphs is hit ≤ once/season/run.
  - **Starter component fallback:** `pitcher_metrics` derives `statcast_rate`
    (runs/9) from full xwOBA-against (`_xwoba_against`) using Savant data, on the
    same ~4.00 scale as xFIP. `PitcherProfile.primary_rate` chain is
    xfip → siera → **statcast_rate** → fip → era; `primary_rate_source` labels it.
    Tunables: `league.avg_xwoba`, `league.xwoba_to_run9`.
  - **Starter** survives via `statcast_rate` even with no CSV; **bullpen/park/
    offense** have no Statcast fallback → 0 until the CSV is supplied.
  - If you add an as-of-date FanGraphs replacement (or a Cloudflare-solving
    fetch), it slots in at `get_season_pitching` / `team_metrics._team_batting`
    with no model changes.
- **pybaseball:**
  - `pitching_stats(season, season, qual=0)` → FanGraphs leaderboard (xFIP, SIERA,
    K-BB%, IP, GS, FIP; **Stuff+ only if present**). Matched to a pitcher **by
    normalized name** (FanGraphs ids ≠ MLBAM ids). **Subject to the 403 above.**
  - `statcast_pitcher(start, end, MLBAM_id)` → pitch-level; we derive Whiff%/CSW%/
    HardHit%/xwOBA-on-contact in `_statcast_rates`. The probable-pitcher `id` from
    MLB Stats API *is* the MLBAM id, used directly.
  - **Bullpen FIP is derived** (no clean team-bullpen endpoint): IP-weighted FIP of
    relievers (`GS/G < 0.5`) per team from the same leaderboard.
  - Everything is cached to `storage/cache/*.parquet` with a 24h TTL.

## Known limitations (see README §7 for the full list)

- Backtest uses **full-season** FanGraphs stats → look-ahead bias on ROI (CLV is
  fine). Wiring an as-of-date stat source is the main backtest improvement.
- Recent-form component is **pitching-only** (hook exists for rolling team offense).
- xwOBA-on-contact is a proxy, not full PA-weighted xwOBA-against.
- Doubleheaders can mis-match on the team-pair odds join.

## Testing

`tests/test_core.py` covers the correctness-critical bits (odds conversions,
de-vig, EV/Kelly, model breakdown + confidence, DB upsert/CLV) with **no network
and no pybaseball**. Run `pytest` or `python -m mlb_value_bot.tests.test_core`.
Add to these when touching `ev_calculator` or `win_probability`.

## Environment

- Windows + **64-bit Python 3.13** in `mlb_value_bot/.venv` with the full
  `requirements.txt` (pybaseball 2.2.7, pyarrow, pandas 3.0, scipy) installed and
  verified. (The only 3.12 on the box is 32-bit, which can't install pyarrow —
  hence 3.13.) Statcast works; FanGraphs is 403-blocked from here (see below).
- `cli.py` reconfigures stdout/stderr to UTF-8 at import (Windows console safety);
  Click help strings are kept ASCII.

## Extending to NFL (planned)

The structure is sport-agnostic on purpose. To add NFL:
1. New `data/nfl_client.py` (schedule/scores/QB designations) + an odds sport key
   (`americanfootball_nfl`) — reuse `OddsClient` by parameterizing `sport_key`.
2. New `analysis/nfl_win_probability.py` with its own components (QB rating,
   EPA/play, rest, weather, HFA) but the **same `Component`/breakdown contract**.
3. Reuse `ev_calculator`, `tracking/*`, `performance/*`, `backtest/*` unchanged —
   they're sport-agnostic (they only see odds, probabilities, and results).
4. Parameterize `pipeline.analyze_slate` by sport, or add `pipeline_nfl.py`.
Keep the same "transparent weighted components + CLV-first measurement" ethos.

## Conventions

- Type hints throughout; dataclasses for data carriers.
- Logging via `utils.get_logger(...)` (rich console + file at
  `storage/mlb_value_bot.log`). No bare `print` outside the CLI's rich output.
- Don't commit `.env` or `storage/` contents.
