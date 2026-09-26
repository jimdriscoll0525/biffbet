-- BiffBet — Supabase schema for the public web dashboard.
-- Run this once in the Supabase SQL editor (Dashboard -> SQL Editor -> New query).
--
-- Design: the site is a FREE, READ-ONLY dashboard. The Python engine writes via
-- the service-role key (which bypasses RLS); anonymous/public visitors can only
-- SELECT. The tables mirror the engine's local SQLite so `sync` is a 1:1 push.

-- ---------------------------------------------------------------------------
-- recommendations — one row per (date, game_id, recommended_side).
-- Mirrors tracking/recommendations.py::_SCHEMA; `reasoning` is the parsed
-- reasoning_json (model components + market-anchor blend) as jsonb.
-- ---------------------------------------------------------------------------
create table if not exists public.recommendations (
    id                    bigint generated always as identity primary key,
    date                  date             not null,   -- game date
    game_id               bigint           not null,
    home_team             text             not null,
    away_team             text             not null,
    recommended_side      text             not null check (recommended_side in ('home', 'away')),
    model_prob            double precision not null,
    market_prob_devigged  double precision not null,
    american_odds         integer          not null,   -- bet price used for EV
    decimal_odds          double precision not null,
    ev_pct                double precision not null,
    kelly_stake           double precision not null,   -- fraction of bankroll
    confidence            double precision not null,   -- 0..100
    reasoning             jsonb,                        -- full model breakdown
    opening_line          integer,
    closing_line          integer,
    clv_pct               double precision,             -- open->close CLV, %
    result                text             not null default 'pending',  -- pending|win|loss|push|void
    profit_loss           double precision,             -- realized, bankroll-fraction units
    -- TRUE = an actual bet (>= EV threshold); FALSE = an analysis breadcrumb
    -- kept so the home page can show the full slate even on quiet days.
    -- Only is_value=TRUE rows count toward performance / grading.
    is_value              boolean          not null default true,
    created_at            timestamptz      not null default now(),
    updated_at            timestamptz      not null default now(),
    -- One row per game per date. We persist whichever side is the best play
    -- of the moment; the side can flip between runs on non-bet analyses, so
    -- the natural key has to be (date, game_id), not include the side.
    constraint recommendations_date_game_id_key unique (date, game_id)
);

create index if not exists recommendations_date_idx     on public.recommendations (date desc);
create index if not exists recommendations_result_idx   on public.recommendations (result);
create index if not exists recommendations_is_value_idx on public.recommendations (is_value);

-- Migration for existing projects (no-op once applied; safe to re-run):
alter table public.recommendations
    add column if not exists is_value boolean not null default true;

-- ---------------------------------------------------------------------------
-- 2026-05-28: tighten unique key from (date, game_id, side) to (date, game_id).
--
-- We only ever recommend one side per game, but the per-side unique key let
-- the sync push create duplicate rows whenever the engine's "best side"
-- flipped between runs (local DB updated in place; Supabase saw it as a new
-- (date, game_id, new_side) tuple). The natural key IS (date, game_id).
--
-- Order: dedupe first (keep the most recent / is_value=true row per game),
-- then swap the constraint. Idempotent and safe to re-run.
-- ---------------------------------------------------------------------------
delete from public.recommendations
where id in (
    select id from (
        select id,
            row_number() over (
                partition by date, game_id
                order by is_value desc, updated_at desc nulls last, id desc
            ) as rn
        from public.recommendations
    ) ranked
    where rn > 1
);

alter table public.recommendations
    drop constraint if exists recommendations_date_game_id_recommended_side_key;

-- Add as unique constraint (named so we can ON CONFLICT against it).
do $$
begin
    if not exists (
        select 1 from pg_constraint
        where conrelid = 'public.recommendations'::regclass
          and conname = 'recommendations_date_game_id_key'
    ) then
        alter table public.recommendations
            add constraint recommendations_date_game_id_key unique (date, game_id);
    end if;
end $$;

-- ---------------------------------------------------------------------------
-- performance_snapshot — precomputed analytics so the site never recomputes.
-- `overall` and `segments` are jsonb shaped exactly like the Python
-- PerformanceReport (tracking/performance.py). One row per scope:
--   'all'              = lifetime
--   'since:2026-04-01' = filtered (optional, future use)
-- ---------------------------------------------------------------------------
create table if not exists public.performance_snapshot (
    scope        text        primary key,
    overall      jsonb       not null,
    segments     jsonb       not null,
    computed_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- Row-level security: public read, no public writes.
-- The engine's service-role key bypasses RLS entirely, so `sync` still writes.
-- ---------------------------------------------------------------------------
alter table public.recommendations      enable row level security;
alter table public.performance_snapshot enable row level security;

drop policy if exists "public read recommendations" on public.recommendations;
create policy "public read recommendations"
    on public.recommendations for select
    to anon, authenticated
    using (true);

drop policy if exists "public read performance" on public.performance_snapshot;
create policy "public read performance"
    on public.performance_snapshot for select
    to anon, authenticated
    using (true);

-- ===========================================================================
-- TOTALS (over/under) — ADDITIVE, PAPER-ONLY. A parallel BiffBet market: its own
-- line, de-vig, sharp consensus, and close. Keyed on (date, game_id) in its OWN
-- table so it never collides with the moneyline `recommendations`. CLV is stored
-- in PROBABILITY POINTS vs the sharp totals close (`clv_pp`), because the totals
-- line moves (the moneyline's decimal-ratio CLV doesn't transfer). `paper` marks
-- a simulated pick; the public site labels these PAPER until CLV proves out.
-- ===========================================================================
create table if not exists public.totals_recommendations (
    id                       bigint generated always as identity primary key,
    date                     date             not null,
    game_id                  bigint           not null,
    home_team                text             not null,
    away_team                text             not null,
    pick_side                text             not null check (pick_side in ('over', 'under')),
    market_total             double precision,
    over_odds                integer,
    under_odds               integer,
    bet_odds                 integer          not null,   -- picked-side price (EV basis)
    decimal_odds             double precision not null,
    model_p_over             double precision,            -- conditional model P(over)
    market_devig_over        double precision,            -- de-vigged market P(over)
    blended_p_over           double precision,
    model_prob               double precision not null,   -- blended P(picked side)
    market_prob_devigged     double precision not null,
    ev_pct                   double precision not null,
    kelly_stake              double precision not null,   -- fraction of bankroll (paper)
    confidence               double precision not null,
    tier                     text,
    stability                text,
    raw_model_total          double precision,
    expected_total           double precision,
    paper                    boolean          not null default true,
    reasoning                jsonb,
    -- CLV (probability-pp move vs the sharp totals close) --------------------
    opening_line             double precision,
    opening_price            integer,
    opening_devig_p_side     double precision,            -- de-vig P(side) at commit
    closing_line             double precision,
    closing_price            integer,
    sharp_close_book         text,
    sharp_close_line         double precision,
    sharp_close_over         integer,
    sharp_close_under        integer,
    sharp_close_devig_p_side double precision,            -- de-vig P(side) at sharp close
    clv_pp                   double precision,            -- (close - entry) * 100 pp
    result                   text             not null default 'pending',  -- pending|win|loss|push|void
    profit_loss              double precision,            -- paper, bankroll-fraction units
    is_value                 boolean          not null default true,
    created_at               timestamptz      not null default now(),
    updated_at               timestamptz      not null default now(),
    constraint totals_recommendations_date_game_id_key unique (date, game_id)
);

create index if not exists totals_recommendations_date_idx     on public.totals_recommendations (date desc);
create index if not exists totals_recommendations_result_idx   on public.totals_recommendations (result);
create index if not exists totals_recommendations_is_value_idx on public.totals_recommendations (is_value);

alter table public.totals_recommendations enable row level security;
drop policy if exists "public read totals" on public.totals_recommendations;
create policy "public read totals"
    on public.totals_recommendations for select
    to anon, authenticated
    using (true);

-- ===========================================================================
-- GriffBet (the challenger) — ADDITIVE. These tables are entirely separate
-- from BiffBet's above; BiffBet's schema is unchanged. Run this in the same
-- Supabase project. The GriffBet engine writes via the same service-role key
-- (bypasses RLS); the public site reads via the anon key under SELECT-only RLS.
-- ===========================================================================

-- GriffBet recommendations: a structural SUPERSET of public.recommendations
-- with the CLV split (raw-model vs blended pick streams) and the sharp closing
-- line + two sharp CLV streams. Keyed on (date, game_id) like BiffBet, but in
-- its OWN table so the two models never collide.
create table if not exists public.griffbet_recommendations (
    id                    bigint generated always as identity primary key,
    date                  date             not null,
    game_id               bigint           not null,
    home_team             text             not null,
    away_team             text             not null,
    recommended_side      text             not null check (recommended_side in ('home', 'away')),
    model_prob            double precision not null,   -- blended pick-side prob (EV basis)
    market_prob_devigged  double precision not null,
    american_odds         integer          not null,   -- blended bet price
    decimal_odds          double precision not null,
    ev_pct                double precision not null,
    kelly_stake           double precision not null,   -- AFTER discipline
    confidence            double precision not null,
    reasoning             jsonb,
    -- CLV split -----------------------------------------------------------
    raw_model_prob        double precision,            -- raw (pre-blend) home prob
    blended_prob          double precision,            -- blended home prob
    raw_pick_side         text,                        -- side the raw model would bet
    raw_pick_open         integer,                     -- raw-pick price at commit
    -- Opening / best-available ("obtainable") close on the blended side ----
    opening_line          integer,
    closing_line          integer,
    clv_pct               double precision,            -- blended open vs best close
    -- Sharp close (Pinnacle-preferred) + the two sharp CLV streams --------
    sharp_close_book      text,
    sharp_close_home_line integer,
    sharp_close_away_line integer,
    clv_raw_vs_sharp      double precision,
    clv_blended_vs_sharp  double precision,
    -- Grading -------------------------------------------------------------
    result                text             not null default 'pending',
    profit_loss           double precision,
    is_value              boolean          not null default true,
    created_at            timestamptz      not null default now(),
    updated_at            timestamptz      not null default now(),
    constraint griffbet_recommendations_date_game_id_key unique (date, game_id)
);

create index if not exists griffbet_recommendations_date_idx     on public.griffbet_recommendations (date desc);
create index if not exists griffbet_recommendations_result_idx   on public.griffbet_recommendations (result);
create index if not exists griffbet_recommendations_is_value_idx on public.griffbet_recommendations (is_value);

-- Referee snapshot: the cross-model report (calibration, EV monotonicity, CLV)
-- produced by griffbet.referee. One row per scope ('all' = lifetime).
create table if not exists public.referee_snapshot (
    scope        text        primary key,
    data         jsonb       not null,
    computed_at  timestamptz not null default now()
);

-- GriffBet feature store: Stage-4 features (incl. historical backfill) for any
-- (date, game_id), so the production trainer sees backfilled features. Read by
-- the engine only (not the public site).
create table if not exists public.griffbet_game_features (
    date          date        not null,
    game_id       bigint      not null,
    features      jsonb       not null,
    source        text,
    updated_at    timestamptz not null default now(),
    primary key (date, game_id)
);

alter table public.griffbet_recommendations enable row level security;
alter table public.referee_snapshot         enable row level security;
alter table public.griffbet_game_features   enable row level security;
drop policy if exists "public read griffbet features" on public.griffbet_game_features;
create policy "public read griffbet features"
    on public.griffbet_game_features for select
    to anon, authenticated using (true);

drop policy if exists "public read griffbet recs" on public.griffbet_recommendations;
create policy "public read griffbet recs"
    on public.griffbet_recommendations for select
    to anon, authenticated
    using (true);

drop policy if exists "public read referee" on public.referee_snapshot;
create policy "public read referee"
    on public.referee_snapshot for select
    to anon, authenticated
    using (true);

-- ============================================================================
-- FOOTBALL (NFL + college FBS) — the matchup-exploitation model's tables.
-- One row per (league, date, game_id, market). PAPER-ONLY until CLV proves
-- out. Records are ALWAYS computed filtered by model_tag x league x market
-- (engine-side, in football_snapshot scopes) — never from this table raw.
-- ============================================================================
create table if not exists public.football_recommendations (
    id                       bigint generated always as identity primary key,
    league                   text not null check (league in ('nfl','cfb')),
    date                     date not null,
    week                     integer,
    game_id                  text not null,
    market                   text not null check (market in ('spread','total','moneyline')),
    home_team                text not null,
    away_team                text not null,
    pick_side                text not null check (pick_side in ('home','away','over','under')),
    line                     double precision,      -- picked-side line (spread) / total
    bet_odds                 integer not null,
    decimal_odds             double precision not null,
    model_prob               double precision not null,
    market_prob_devigged     double precision not null,
    p_push                   double precision,
    ev_pct                   double precision not null,
    adjusted_ev_pct          double precision,
    flat_stake               double precision not null,
    confidence               double precision not null,
    tier                     text,
    stability                text,
    edge_score               double precision,
    archetype                text,
    projected_margin         double precision,
    projected_total          double precision,
    paper                    boolean not null default true,
    model_tag                text not null default 'matchup_v1',
    reasoning                jsonb,
    opening_line             double precision,
    opening_price            integer,
    opening_devig_p_side     double precision,
    closing_line             double precision,
    closing_price            integer,
    sharp_close_line         double precision,
    sharp_close_devig_p_side double precision,
    clv_pp                   double precision,      -- probability points vs sharp close
    result                   text not null default 'pending'
                             check (result in ('pending','win','loss','push','void')),
    home_score               integer,
    away_score               integer,
    profit_loss              double precision,      -- bankroll-fraction units
    is_value                 boolean not null default false,
    created_at               timestamptz,
    updated_at               timestamptz,
    constraint football_recs_key unique (league, date, game_id, market)
);
create index if not exists football_recs_date_idx   on public.football_recommendations (date desc);
create index if not exists football_recs_league_idx on public.football_recommendations (league);
create index if not exists football_recs_result_idx on public.football_recommendations (result);

-- Precomputed aggregates the site reads: record:<league>:<market>,
-- distribution:<league>:total, calibration:<model_tag>, record:all:all.
create table if not exists public.football_snapshot (
    scope        text        primary key,
    payload      jsonb       not null,
    updated_at   timestamptz not null default now()
);

alter table public.football_recommendations enable row level security;
alter table public.football_snapshot        enable row level security;

drop policy if exists "public read football recs" on public.football_recommendations;
create policy "public read football recs"
    on public.football_recommendations for select
    to anon, authenticated
    using (true);

drop policy if exists "public read football snapshot" on public.football_snapshot;
create policy "public read football snapshot"
    on public.football_snapshot for select
    to anon, authenticated
    using (true);

-- ============================================================================
-- LIVE TOTALS (live_total_v1) — the in-game NFL over/under projection tool.
-- Priors are engine-computed (football/analysis/drive_stats.py, synced daily);
-- snapshots + recommendations are written by the biffbet site's server routes
-- (service-role key) on each "Update Over/Under" press. Grading is engine-side
-- (football/tracking/football_live_results.py). CLV NOTE: clv_pts here is
-- POINTS vs the self-captured closing live line — a deliberately different
-- name/metric from the pregame clv_pp (probability points vs sharp close);
-- never aggregate the two.
-- ============================================================================

-- (1) Team drive-stat priors — engine writes, site reads.
create table if not exists public.football_team_drive_stats (
    league               text not null check (league in ('nfl','cfb')),
    season               integer not null,
    team                 text not null,           -- nflverse abbr (KC, BUF, LA, ...)
    games                integer not null,
    ppd_off              double precision,        -- points per offensive drive
    ppd_def_allowed      double precision,        -- points allowed per defensive drive
    drives_pg            double precision,
    plays_per_drive      double precision,
    sec_per_play         double precision,
    drive_sec_avg        double precision,
    explosive_play_rate  double precision,        -- (pass>=20yd + rush>=10yd) / plays
    pts_per_min_trailing double precision,        -- scoring rate while trailing
    quick_td_rate        double precision,        -- TD drives <= 2:00 TOP / drives (2026-09)
    quick_td_allowed_rate double precision,       -- same, allowed by the defense
    updated_at           timestamptz not null default now(),
    primary key (league, season, team)
);
-- 2026-09-10: quick-strike columns for the site's data-derived X-factor
-- (idempotent for tables created before they existed).
alter table public.football_team_drive_stats
    add column if not exists quick_td_rate double precision,
    add column if not exists quick_td_allowed_rate double precision;

-- (2) One row per "Update Over/Under" press on the site.
create table if not exists public.football_live_snapshots (
    id                bigint generated always as identity primary key,
    league            text not null default 'nfl' check (league in ('nfl','cfb')),
    game_key          text not null,               -- ESPN event id (canonical live key)
    date              date not null,
    home_team         text not null,               -- ESPN abbr
    away_team         text not null,
    captured_at       timestamptz not null default now(),
    quarter           integer,                     -- 1-4, 5 = OT
    clock             text,                        -- "MM:SS" as displayed
    seconds_remaining integer,                     -- regulation seconds left
    home_score        integer not null,
    away_score        integer not null,
    possession        text check (possession in ('home','away')),
    home_timeouts     integer,
    away_timeouts     integer,
    live_line         double precision,            -- manual entry; null = not entered
    erp               double precision not null,   -- expected remaining points
    projected_total   double precision not null,   -- current pts + ERP + OT component
    edge              double precision,            -- projected_total - live_line
    lean              text check (lean in ('over','under','pass')),
    confidence_tier   text check (confidence_tier in ('high','medium','low')),
    sentiment_delta   double precision,            -- vs previous snapshot
    sentiment_cum     double precision,            -- rolling per-game cumulative
    scoring_event     text,                        -- 'td7','td6','td8','fg3','safety2','multi','none'
    flags             jsonb,                       -- badges: x_factor_alive, kneel_down_range, ...
    inputs            jsonb,                       -- manual overrides + data_health provenance
    model_tag         text not null default 'live_total_v1'
);
create index if not exists football_live_snap_game_idx
    on public.football_live_snapshots (game_key, captured_at desc);
create index if not exists football_live_snap_date_idx
    on public.football_live_snapshots (date desc);

-- (3) Recommendation log — one row per press that produced an over/under lean
--     (passes live only in snapshots). PAPER-ONLY.
create table if not exists public.football_live_recommendations (
    id                bigint generated always as identity primary key,
    league            text not null default 'nfl' check (league in ('nfl','cfb')),
    date              date not null,
    game_key          text not null,
    home_team         text not null,
    away_team         text not null,
    snapshot_id       bigint references public.football_live_snapshots(id),
    captured_at       timestamptz not null default now(),
    live_line         double precision not null,
    projected_total   double precision not null,
    edge              double precision not null,
    lean              text not null check (lean in ('over','under')),
    confidence_tier   text not null check (confidence_tier in ('high','medium','low')),
    badges            jsonb,
    paper             boolean not null default true,
    model_tag         text not null default 'live_total_v1',
    result            text not null default 'pending'
                      check (result in ('pending','win','loss','push','void')),
    final_total       integer,
    closing_live_line double precision,            -- last captured live_line for the game
    clv_pts           double precision,            -- signed pts of line move in lean's favor
    graded_at         timestamptz
);
create index if not exists football_live_recs_result_idx
    on public.football_live_recommendations (result);
create index if not exists football_live_recs_date_idx
    on public.football_live_recommendations (date desc);

alter table public.football_team_drive_stats     enable row level security;
alter table public.football_live_snapshots       enable row level security;
alter table public.football_live_recommendations enable row level security;

drop policy if exists "public read drive stats" on public.football_team_drive_stats;
create policy "public read drive stats"
    on public.football_team_drive_stats for select
    to anon, authenticated
    using (true);

drop policy if exists "public read live snapshots" on public.football_live_snapshots;
create policy "public read live snapshots"
    on public.football_live_snapshots for select
    to anon, authenticated
    using (true);

drop policy if exists "public read live recs" on public.football_live_recommendations;
create policy "public read live recs"
    on public.football_live_recommendations for select
    to anon, authenticated
    using (true);

-- ============================================================================
-- RSI (Recursive Self-Improvement) — 2026-09-26. ADDITIVE.
--
-- The engine mines its OWN recommendation history (picks AND passes, all
-- engines) every Monday, writes Proposals for Jim's approval, runs approved
-- changes as shadow challengers, and lets the Proposals tab promote / roll
-- back model versions. Nothing here changes the live model on its own: the
-- engine only ever reads rsi_model_versions(status='active') and
-- rsi_proposals(status='approved'); every transition beyond watch/pending is
-- a button press in the tab.
--
-- Invariants:
--   * recommendations / totals_recommendations / football_recommendations
--     only ever hold CHAMPION rows; challengers write to rsi_shadow_picks.
--   * CLV metrics never mix: clv_pct (MLB ML, % re-pricing), clv_pp (totals +
--     football, probability points vs sharp close), clv_blended_vs_sharp
--     (GriffBet). Every RSI aggregate carries a clv_metric label.
--   * rsi_* tables are personal-only: RLS enabled with NO policies, so the anon
--     key sees nothing; the site's service-role routes (passcode-gated) and the
--     engine's service key are the only readers/writers.
-- ============================================================================

-- --- 1. Additive columns on the MLB tables (football already has model_tag) --
alter table public.recommendations
    add column if not exists model_tag   text not null default 'biff_v1',
    add column if not exists pass_reason text;   -- null on bets; see vocabulary below
alter table public.totals_recommendations
    add column if not exists model_tag   text not null default 'totals_v1',
    add column if not exists pass_reason text;
create index if not exists recommendations_model_tag_idx on public.recommendations (model_tag);
create index if not exists totals_recs_model_tag_idx    on public.totals_recommendations (model_tag);
-- pass_reason vocabulary (MLB ML): below_threshold | filter:heavy_favorite |
--   filter:min_model_prob | filter:no_sharp_coverage | skip:divergence |
--   skip:max_ev | skip:sharp_fade   (several joined with '+').
-- Totals add: weather_held. Football keeps hold_reason inside reasoning.

-- --- 2. Proposals -------------------------------------------------------------
create table if not exists public.rsi_proposals (
    id                 bigint generated always as identity primary key,
    engine             text not null check (engine in ('mlb','mlb_totals','football','griffbet')),
    sport              text not null,      -- mlb | mlb_totals | nfl | cfb | griffbet
    finding_key        text not null,      -- ledger key format: '<pool>|<dim>=<value>'
    kind               text not null default 'overlay' check (kind in ('overlay','insight')),
    title              text not null,
    description        text not null,      -- plain English pattern
    suggested_change   text,               -- plain English rule / weight change
    overlay            jsonb,              -- {"filters.min_model_prob": 0.5} dotted config keys; null = insight
    status             text not null default 'watch'
                       check (status in ('watch','pending','snoozed','approved','promoted','rejected','dropped')),
    confidence         text not null default 'low' check (confidence in ('low','medium','high')),
    direction          text,               -- positive | negative | structural
    latest_stats       jsonb not null,     -- {n, settled, wins, losses, hit_rate, flat_roi, avg_clv, clv_metric,
                                           --  clv_tracked, t_stat, p_value, bonferroni, clv_agrees, date_from, date_to}
    evidence           jsonb not null default '[]'::jsonb,   -- one entry per review run
    consecutive_hits   integer not null default 0,
    consecutive_misses integer not null default 0,
    first_seen         date not null,
    last_seen          date not null,
    snoozed_until      date,
    decided_at         timestamptz,
    decision_reason    text,
    challenger_tag     text unique,        -- set on approve: biff_p<id> / totals_p<id> / matchup_p<id>
    shadow_started_at  date,
    shadow_ends_at     date,
    shadow_stats       jsonb,              -- engine-computed champion vs challenger + gate
    version_id         bigint,             -- rsi_model_versions.id once promoted
    created_at         timestamptz not null default now(),
    updated_at         timestamptz not null default now(),
    constraint rsi_proposals_engine_key unique (engine, finding_key)
);
create index if not exists rsi_proposals_status_idx on public.rsi_proposals (status, sport);

-- Permanent decision / lifecycle history (never deleted).
create table if not exists public.rsi_proposal_events (
    id           bigint generated always as identity primary key,
    proposal_id  bigint not null references public.rsi_proposals(id) on delete cascade,
    event        text not null,   -- seen|bar_met|pending|approved|rejected|snoozed|unsnoozed|dropped|promoted|rolled_back|imported|shadow_stats
    actor        text not null default 'engine',   -- engine | jim
    payload      jsonb,
    created_at   timestamptz not null default now()
);
create index if not exists rsi_events_proposal_idx on public.rsi_proposal_events (proposal_id, created_at desc);

-- --- 3. Model versions (one ACTIVE per engine; the engine stamps rows with tag)
create table if not exists public.rsi_model_versions (
    id               bigint generated always as identity primary key,
    engine           text not null check (engine in ('mlb','mlb_totals','football')),
    tag              text not null unique,        -- biff_v1, biff_v2, totals_v1, matchup_v1 ...
    parent_tag       text,
    proposal_id      bigint references public.rsi_proposals(id),
    overlay          jsonb not null default '{}'::jsonb,   -- CUMULATIVE dotted-key overlay vs base yaml
    status           text not null default 'active' check (status in ('active','superseded','rolled_back')),
    promoted_at      timestamptz not null default now(),
    retired_at       timestamptz,
    baseline         jsonb,    -- champion stats over the shadow window at promotion time
    rolling          jsonb,    -- latest rolling-window stats {n, avg_clv, clv_metric, flat_roi, window, computed_at}
    rollback_flagged boolean not null default false,
    rollback_reason  text,
    notes            text
);
create unique index if not exists rsi_versions_one_active_per_engine
    on public.rsi_model_versions (engine) where status = 'active';

insert into public.rsi_model_versions (engine, tag, overlay, notes) values
    ('mlb',        'biff_v1',    '{}', 'baseline (includes the filters.min_model_prob ability, 2026-08-30)'),
    ('mlb_totals', 'totals_v1',  '{}', 'baseline'),
    ('football',   'matchup_v1', '{}', 'baseline (includes the CFB elo margin anchor, 2026-08-31)')
on conflict (tag) do nothing;

-- --- 4. Shadow picks (challenger output; never in the public record) ---------
create table if not exists public.rsi_shadow_picks (
    id                       bigint generated always as identity primary key,
    challenger_tag           text not null,
    engine                   text not null,
    sport                    text not null,   -- mlb | mlb_totals | nfl | cfb
    league                   text,
    date                     date not null,
    game_id                  text not null,
    market                   text not null,   -- moneyline | total | spread
    home_team                text not null,
    away_team                text not null,
    pick_side                text not null,
    is_value                 boolean not null,
    pass_reason              text,
    champion_is_value        boolean,         -- the champion's decision on the same game/market that run
    ev_pct                   double precision,
    adjusted_ev_pct          double precision,
    confidence               double precision,
    model_prob               double precision,
    market_prob_devigged     double precision,
    bet_odds                 integer,
    decimal_odds             double precision,
    line                     double precision,
    opening_price            integer,
    opening_line             double precision,
    opening_devig_p_side     double precision,
    closing_price            integer,
    closing_line             double precision,
    sharp_close_devig_p_side double precision,
    clv                      double precision,
    clv_metric               text not null,   -- clv_pct | clv_pp
    result                   text not null default 'pending',
    flat_pl                  double precision, -- flat 1u units (win = decimal-1, loss = -1)
    reasoning                jsonb,
    created_at               timestamptz not null default now(),
    updated_at               timestamptz not null default now(),
    constraint rsi_shadow_picks_key unique (challenger_tag, sport, date, game_id, market)
);
create index if not exists rsi_shadow_tag_date_idx on public.rsi_shadow_picks (challenger_tag, date desc);

-- --- 5. Review runs (one row per weekly run; summary feeds the email + tab) --
create table if not exists public.rsi_review_runs (
    id            bigint generated always as identity primary key,
    run_date      date not null,
    sport         text not null,          -- all | mlb | mlb_totals | football | griffbet
    started_at    timestamptz not null default now(),
    finished_at   timestamptz,
    status        text not null default 'running',   -- running | ok | error
    params        jsonb,                  -- min_sample, t_threshold, holdout_pct ...
    row_counts    jsonb,
    cells_tested  integer,
    summary       jsonb,                  -- per-sport headline block
    email_sent_at timestamptz,
    error         text
);
create index if not exists rsi_review_runs_date_idx on public.rsi_review_runs (run_date desc);

alter table public.rsi_proposals       enable row level security;
alter table public.rsi_proposal_events enable row level security;
alter table public.rsi_model_versions  enable row level security;
alter table public.rsi_shadow_picks    enable row level security;
alter table public.rsi_review_runs     enable row level security;
-- Deliberately NO policies: service-role only.

-- --- 6. rsi_scored_games — every scored game across engines, one shape --------
-- No data duplication: a UNION over the four recommendation tables. decision is
-- 'pick' (is_value) or 'pass'; features is the full reasoning jsonb; clv_metric
-- labels the unit. is_holdout is a deterministic 20% split on the game key
-- (md5 of '<sport_key>:<game_id>', first 7 hex chars as an int, mod 100 < 20)
-- that the weekly review NEVER uses for discovery — only promotion validation.
-- Python twin: rsi/holdout.py::is_holdout (a test pins both to the same values).
-- security_invoker: readable by whoever can read the underlying tables (all
-- four are public-read), so the site could read it with the anon key too.
create or replace view public.rsi_scored_games with (security_invoker = on) as
select
    'mlb'::text                   as engine,
    'mlb'::text                   as sport,
    null::text                    as league,
    r.date,
    r.game_id::text               as game_key,
    r.home_team, r.away_team,
    'moneyline'::text             as market,
    r.recommended_side            as pick_side,
    r.reasoning                   as features,
    r.model_tag                   as model_version,
    r.ev_pct,
    (r.reasoning->'adjusted_ev'->>'adjusted_ev_pct')::double precision as adjusted_ev_pct,
    r.confidence,
    case when r.is_value then 'pick' else 'pass' end as decision,
    r.pass_reason,
    r.opening_line::double precision as opening_line,
    r.opening_line                as opening_price,
    r.american_odds               as pick_price,
    null::double precision        as pick_line,
    r.closing_line::double precision as closing_line,
    r.closing_line                as closing_price,
    r.result,
    r.clv_pct                     as clv,
    'clv_pct'::text               as clv_metric,
    case r.result when 'win' then r.decimal_odds - 1 when 'loss' then -1 when 'push' then 0 when 'void' then 0 end as flat_pl_units,
    r.profit_loss                 as kelly_pl_units,
    (('x' || substr(md5('mlb:' || r.game_id::text), 1, 7))::bit(28)::int % 100) < 20 as is_holdout,
    r.created_at, r.updated_at
from public.recommendations r
union all
select
    'mlb_totals', 'mlb_totals', null, t.date, t.game_id::text, t.home_team, t.away_team,
    'total', t.pick_side, t.reasoning, t.model_tag,
    t.ev_pct, null, t.confidence,
    case when t.is_value then 'pick' else 'pass' end, t.pass_reason,
    t.opening_line, t.opening_price, t.bet_odds, t.market_total,
    t.closing_line, t.closing_price,
    t.result, t.clv_pp, 'clv_pp',
    case t.result when 'win' then t.decimal_odds - 1 when 'loss' then -1 when 'push' then 0 when 'void' then 0 end,
    t.profit_loss,
    (('x' || substr(md5('mlb_totals:' || t.game_id::text), 1, 7))::bit(28)::int % 100) < 20,
    t.created_at, t.updated_at
from public.totals_recommendations t
union all
select
    'griffbet', 'griffbet', null, g.date, g.game_id::text, g.home_team, g.away_team,
    'moneyline', g.recommended_side, g.reasoning, 'griff_v1',
    g.ev_pct, null, g.confidence,
    case when g.is_value then 'pick' else 'pass' end, null,
    g.opening_line::double precision, g.opening_line, g.american_odds, null,
    g.closing_line::double precision, g.closing_line,
    g.result, g.clv_blended_vs_sharp, 'clv_blended_vs_sharp',
    case g.result when 'win' then g.decimal_odds - 1 when 'loss' then -1 when 'push' then 0 when 'void' then 0 end,
    g.profit_loss,
    (('x' || substr(md5('griffbet:' || g.game_id::text), 1, 7))::bit(28)::int % 100) < 20,
    g.created_at, g.updated_at
from public.griffbet_recommendations g
union all
select
    'football', f.league, f.league, f.date, f.game_id, f.home_team, f.away_team,
    f.market, f.pick_side, f.reasoning, f.model_tag,
    f.ev_pct, f.adjusted_ev_pct, f.confidence,
    case when f.is_value then 'pick' else 'pass' end, f.reasoning->>'hold_reason',
    f.opening_line, f.opening_price, f.bet_odds, f.line,
    f.closing_line, f.closing_price,
    f.result, f.clv_pp, 'clv_pp',
    case f.result when 'win' then f.decimal_odds - 1 when 'loss' then -1 when 'push' then 0 when 'void' then 0 end,
    f.profit_loss,
    (('x' || substr(md5(f.league || ':' || f.game_id), 1, 7))::bit(28)::int % 100) < 20,
    f.created_at, f.updated_at
from public.football_recommendations f;
