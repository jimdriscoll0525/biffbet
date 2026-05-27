# mlb_value_bot

A transparent, terminal-based tool for finding **+EV moneyline bets** in MLB
games, focused on **starting-pitcher matchups**. It pulls live odds and team /
pitcher data, runs a tunable win-probability model, computes expected value and
Kelly stakes, and tracks every recommendation (including **closing-line value**)
so you can measure whether the model actually has an edge.

> **Design philosophy:** this tool is built to *measure edge*, not to *pick
> winners*. Win rate is noisy at small samples; CLV and disciplined EV
> accounting are the real scoreboard. The model is intentionally a transparent,
> weighted sum of interpretable components — never a black box.

---

## 1. Install

Requires **Python 3.11+** (developed/tested against 3.11–3.12; see the note on
pybaseball below if you're on 3.13/3.14).

```powershell
# from the directory that CONTAINS the mlb_value_bot folder (e.g. C:\Users\jim)
python -m venv mlb_value_bot\.venv
mlb_value_bot\.venv\Scripts\Activate.ps1        # PowerShell
# source mlb_value_bot/.venv/bin/activate       # macOS/Linux

pip install -r mlb_value_bot\requirements.txt
```

> **pybaseball + Python version:** use a **64-bit** interpreter. `pybaseball`
> installs cleanly on 64-bit **Python 3.13** (verified here: 3.13.7). Avoid
> 32-bit Python — `pyarrow` (used by the cache) ships no 32-bit Windows wheel.
> Python 3.14 may not yet have wheels for every scientific dependency; if pip
> struggles, use 3.13. The bot also runs *without* pybaseball — it just can't
> compute pitcher/team metrics, so confidence drops and the model leans on base
> rate + home field. **See §7 for an important note on FanGraphs availability.**

Run it as a module from the directory that contains the package:

```powershell
python -m mlb_value_bot --help
```

## 2. API key setup

Only **The Odds API** needs a key (the MLB Stats API is free and key-less).

1. Get a free key at <https://the-odds-api.com> (500 requests/month on the free
   tier; each `today` run costs **1** request).
2. Copy the example env file and fill it in:

   ```powershell
   Copy-Item mlb_value_bot\.env.example mlb_value_bot\.env
   ```

   ```dotenv
   ODDS_API_KEY=your_key_here
   BANKROLL=1000          # used for Kelly stake sizing / $ display
   LOG_LEVEL=INFO
   ```

The bot logs the **remaining request quota** from the API response headers after
every odds pull, so you can watch your monthly budget.

## FanGraphs data (CSV workflow)

**Why this exists:** FanGraphs leaderboards sit behind a Cloudflare JavaScript
challenge that returns HTTP **403** to every scraper (pybaseball included),
regardless of IP — so xFIP/SIERA/K-BB% (pitchers), wRC+ (offense), and bullpen
FIP can't be fetched automatically. The reliable fix is to **export the CSVs
from your browser** (where the challenge solves invisibly) and drop them in
`storage/fangraphs/`. The model prefers these files over live scraping; if
they're absent, the starter component falls back to a Statcast-derived rate and
the bullpen/park components sit out.

**One-time / weekly steps:**

1. **Pitching** — open FanGraphs → *Leaderboards → Pitching*, pick the season,
   set **Min IP = 0** (so relievers and low-IP arms are included), and make sure
   the table shows at least: **Name, Team, IP, G, GS, FIP**, and **xFIP and/or
   SIERA** (add **K-BB%** and **Stuff+** if you like — they're used if present).
   Click **Export Data** and save as:

   ```
   mlb_value_bot/storage/fangraphs/pitching_<season>.csv      e.g. pitching_2026.csv
   ```

   A single pitching export feeds **both** the starter rates and the derived
   bullpen FIP (relievers = `GS/G < 0.5`).

2. **Team batting (optional, for wRC+)** — *Leaderboards → Team Batting*, export,
   save as `storage/fangraphs/batting_<season>.csv`. Without it, the park
   component (which uses wRC+) simply stays at 0.

3. **Verify** what the bot can read:

   ```powershell
   python -m mlb_value_bot data-status            # current season
   python -m mlb_value_bot data-status --season 2026
   ```

   It prints whether each file was found, the row count, and which key columns
   were detected — and, if a file is missing, the exact filename to save.

Notes: a bare `pitching.csv` / `batting.csv` (no season suffix) is also accepted.
Percentage columns like `"18.0%"` and team abbreviations (`NYY`, `LAD`, …) are
handled automatically. Refresh the exports as often as you want fresh stats
(weekly is plenty mid-season). You can disable the whole mechanism with
`fangraphs_csv.enabled: false` in `config.yaml`.

## 3. Daily workflow

```powershell
# Morning: pull the slate, see +EV bets, save recommendations
python -m mlb_value_bot today

# (Optional) run `today` again ~10–15 min before first pitch.
# It re-prices the same games and records the CLOSING line + CLV for open bets.
python -m mlb_value_bot today

# Next morning: settle yesterday's bets against final scores
python -m mlb_value_bot results            # defaults to yesterday
python -m mlb_value_bot results --date 2026-05-24

# Anytime: review performance, segmented every which way
python -m mlb_value_bot performance
python -m mlb_value_bot performance --since 2026-04-01
```

Useful flags:

| Command | Flag | Effect |
|---|---|---|
| `today` | `--date YYYY-MM-DD` | analyze a specific date |
| `today` | `--all` | show every game, not just +EV ones |
| `today` | `--min-ev 0.05` | override the EV threshold for this run |
| `today` | `--no-save` | don't write recommendations to the DB |
| `performance` | `--since YYYY-MM-DD` | filter to recent games |
| (any) | `clear-cache` | drop cached pybaseball/Statcast parquet files |

## Web viewer

Prefer a browser to the terminal? Launch the local Streamlit viewer:

```powershell
python -m mlb_value_bot serve                 # opens http://localhost:8501
python -m mlb_value_bot serve --port 8600 --headless
```

Three views mirror the CLI:

- **Today** — the +EV table with expandable model breakdowns (base → weighted
  components → market blend). A **Run / refresh analysis** button triggers the
  live pipeline; that button is the *only* thing that spends an Odds API request.
  Otherwise the page just displays saved recommendations from the DB.
- **Results** — grade a date against final scores and see daily P/L.
- **Performance** — ROI / hit-rate / **CLV**, segmented, with per-bucket CLV charts.

Viewing reads the local SQLite DB (fast, free); nothing is fetched from an API
until you click **Run / refresh** (Today) or **Grade** (Results).

## 4. Interpreting the output

### The slate table (`today`)

| Column | Meaning |
|---|---|
| **Matchup** | `Away @ Home` |
| **Pick** | model's best-EV side (the team + home/away) |
| **Odds** | best available American price across your configured books |
| **Model%** | model's win probability for the pick |
| **Mkt%** | de-vigged ("fair") market probability for the pick |
| **EV%** | expected profit per unit staked = `model_prob × decimal_odds − 1` |
| **Kelly%** | fractional-Kelly stake as a **% of bankroll** (capped at 2%) |
| **Stake** | Kelly% × your `BANKROLL` (display only) |
| **Conf** | confidence score, 0–100 (see below) |

Rows in **green** clear your EV threshold and get saved. Below the table, each
+EV bet gets a **per-component breakdown panel** so you can see *why* the model
favors that side — the base rate and each weighted adjustment (`Δ` = points of
home win probability added). Components marked `(missing)` had no data and
contributed 0.

### Confidence score (0–100)

A conservative composite (weights in `config.yaml → confidence.weights`):

- **data_completeness** — did we actually get both pitchers + team data?
- **sample_size** — innings pitched of the *weaker-link* starter vs a "trustworthy" threshold.
- **edge_magnitude** — bigger measured EV ⇒ (weakly) more confidence, capped.
- **component_agreement** — do the skill components all lean the same way, or does
  the edge hinge on one factor while others disagree? A coin-flip edge scores low.

Low confidence on a juicy-looking EV usually means *thin data or internal
disagreement* — treat those with suspicion.

### Results & performance

`results` prints each settled bet (win/loss/void) and the day's P/L + ROI in
**bankroll-fraction units** (1.0 = your whole bankroll). `performance` reports:

- **Kelly ROI** (how the bankroll actually moved) and **Flat ROI** (every bet = 1u),
- **Hit rate** (secondary at low N),
- **Avg CLV%** — the most stable early-sample signal of genuine edge,

…segmented by **confidence bucket**, **EV bucket**, **favorite/underdog**,
**home/road**, and **CLV positive/negative**.

## 5. Tuning `config.yaml`

Everything that controls behavior lives in `config.yaml` — no code changes
needed. Highlights:

- `model.weights.*` — multiplier on each model component. **Set any to `0.0` to
  disable it.** This is the main knob for changing how the model thinks.
- `model.pitcher_run_to_winpct` / `bullpen_run_to_winpct` — how strongly a
  run/9 of pitcher/bullpen quality moves win probability.
- `model.pitcher_stat` — `"xfip"` or `"siera"` as the primary starter rate stat.
- `model.team_regression_games` — how hard early-season win% is pulled to .500.
- `model.home_field_advantage` — baseline home boost (default 3.5%).
- `ev.threshold` — minimum EV% to flag a bet (default **0.03**; deliberately not
  0.05, which filters out almost everything early in the year).
- `ev.devig_method` — `"power"` (favorite-longshot aware) or `"proportional"`.
- `kelly.fraction` / `kelly.max_bankroll_fraction` — staking aggressiveness + cap.
- `park_factors` — hardcoded, editable per-park run factors.
- `league.*` — league-average anchors (update these seasonally).

After editing, just re-run `today`; the config is reloaded every run.

## 6. Backtesting

Historical odds from The Odds API require a **paid** plan, so backtesting uses a
**CSV fallback**. Provide a CSV of past lines (see
`backtest/sample_odds_template.csv`):

```csv
date,home_team,away_team,home_odds,away_odds,home_closing,away_closing
2024-04-01,New York Yankees,Houston Astros,-145,125,-150,130
```

`home_closing`/`away_closing` are optional (enable CLV in the backtest). Run:

```powershell
python -m mlb_value_bot backtest --start 2024-04-01 --end 2024-09-30 --csv my_odds.csv
```

The backtester pulls each day's real schedule, probable pitchers, and final
score from the MLB Stats API and grades the model's picks.

## 7. Known limitations

- **FanGraphs scraping is blocked (HTTP 403) — use the CSV workflow.** This is
  the big one. pybaseball pulls xFIP, SIERA, K-BB% (pitchers), wRC+ (offense),
  and bullpen FIP from FanGraphs, which sits behind a Cloudflare **JavaScript
  challenge** ("Just a moment…"). That returns **403** to every scraper
  regardless of IP — confirmed here from a residential connection, even with a
  browser User-Agent. It is not an IP ban and not a header fix; only a real
  browser solves the challenge. Handling:
  - **Primary fix:** export the leaderboard CSVs from your browser into
    `storage/fangraphs/` — see the **“FanGraphs data (CSV workflow)”** section
    above and `python -m mlb_value_bot data-status`. With the CSVs present,
    xFIP/SIERA/wRC+/bullpen-FIP all work normally.
  - A FanGraphs outage **never crashes a run** — the cache degrades to empty and
    the affected components drop out (logged as a warning).
  - **Without the CSVs, the starter component still works** via a Statcast-derived
    `statcast_rate` (runs/9 from full xwOBA-against on Baseball Savant, which *is*
    reachable); the breakdown shows e.g. `rate H=2.04(statcast_rate)`. The
    bullpen and park/offense components have no Statcast fallback yet, so they
    sit at 0 until you supply the CSVs. Tune the fallback via `league.avg_xwoba`
    / `league.xwoba_to_run9` in `config.yaml`.
- **Statcast metric availability.** The bot only surfaces metrics `pybaseball`
  actually returns (xFIP, SIERA, K-BB%, Whiff%, CSW%, HardHit%, xwOBA-on-contact,
  recent form). **Stuff+** is included *only if* your FanGraphs leaderboard pull
  exposes it; otherwise it's omitted rather than fabricated. `xwOBA-on-contact`
  is a contact-quality proxy (mean estimated wOBA on balls in play), not a full
  PA-weighted xwOBA-against.
- **Bullpen rating is a derived proxy.** pybaseball has no clean "team bullpen"
  endpoint, so bullpen FIP is built by IP-weighting *relievers only* (GS/G < 0.5)
  from the FanGraphs pitcher leaderboard. It is not leverage-weighted.
- **Recent-form component is pitching-only for now.** It reflects starters' last
  ~5 starts (Statcast). Rolling *team-offense* form (last 14 days) isn't pulled
  yet — there's a clearly marked hook to add it.
- **Park factor has a deliberately small moneyline effect.** Park primarily moves
  totals; here it only modulates the offense gap, with a low default weight.
- **Historical odds need a paid Odds API tier.** Hence the CSV backtest fallback.
- **Backtest look-ahead bias.** The FanGraphs *season* stat path uses full-season
  numbers, so backtest ROI is optimistic. Statcast pulls are correctly windowed
  to ≤ the game date, and supplied closing lines make **CLV** in backtests
  unbiased. Treat backtest win/ROI as directional until an as-of-date stat source
  is wired in.
- **Doubleheaders / line shopping.** Odds↔schedule matching is by team pair, so a
  doubleheader can mis-match; the bot records the single best price across your
  configured books (not a true no-vig consensus of all books).
- **This is a personal analysis tool. It is not betting advice. Bet responsibly
  and only what you can afford to lose.**

## 8. Project layout

```
mlb_value_bot/
├── cli.py                 # Click entry point (python -m mlb_value_bot ...)
├── pipeline.py            # orchestration: odds + schedule + metrics -> analyses
├── config.yaml            # all thresholds, weights, park factors
├── constants.py           # canonical team names + normalization
├── utils.py               # paths, config loading, logging
├── data/
│   ├── odds_client.py     # The Odds API wrapper (best price, quota logging)
│   ├── mlb_client.py      # MLB Stats API (schedule, pitchers, scores, standings)
│   ├── fangraphs_csv.py   # loads browser-exported FanGraphs CSVs (Cloudflare workaround)
│   └── cache.py           # parquet cache w/ 24h TTL + stale-fallback
├── analysis/
│   ├── pitcher_metrics.py # FanGraphs + Statcast pitcher profiles
│   ├── team_metrics.py    # offense (wRC+), bullpen FIP, park factors, win%
│   ├── win_probability.py # transparent weighted model + confidence score
│   └── ev_calculator.py   # odds conversions, de-vig, EV, Kelly
├── tracking/
│   ├── recommendations.py # SQLite store (full schema, CLV)
│   ├── results.py         # settle bets vs final scores
│   └── performance.py     # ROI / hit rate / CLV, segmented
├── backtest/
│   └── backtester.py      # CSV-odds historical re-run
├── web/
│   └── app.py             # Streamlit viewer (python -m mlb_value_bot serve)
├── tests/test_core.py     # EV math, model, tracking (no network)
└── storage/               # SQLite DB + cached parquet + logs (auto-created)
```

See `CLAUDE.md` for architecture decisions and extension notes (incl. NFL).
