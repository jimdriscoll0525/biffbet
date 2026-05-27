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
    created_at            timestamptz      not null default now(),
    updated_at            timestamptz      not null default now(),
    unique (date, game_id, recommended_side)
);

create index if not exists recommendations_date_idx   on public.recommendations (date desc);
create index if not exists recommendations_result_idx on public.recommendations (result);

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
