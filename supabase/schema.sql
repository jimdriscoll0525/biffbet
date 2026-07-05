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
