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
