-- PASO 4: LIVE timing, technical runs and later observed outcomes.
-- Existing predictions remain readable; their historical timestamp still means source open.
alter table public.predictions
    add column generated_at timestamptz,
    add column valid_until timestamptz,
    add constraint predictions_timing_pair check (
        (generated_at is null and valid_until is null)
        or (generated_at is not null and valid_until is not null)
    ),
    add constraint predictions_valid_until_after_source check (
        valid_until is null or valid_until > candle_open_time
    );

create table public.pipeline_runs (
    run_id uuid primary key,
    started_at timestamptz not null,
    finished_at timestamptz not null,
    duration_ms numeric not null check (duration_ms >= 0),
    market_data_ms numeric check (market_data_ms >= 0),
    prediction_ms numeric check (prediction_ms >= 0),
    evaluation_ms numeric check (evaluation_ms >= 0),
    status text not null check (status in (
        'SUCCESS', 'NO_NEW_CANDLE', 'PREDICTION_ALREADY_EXISTS',
        'INSUFFICIENT_HISTORY', 'FAILED'
    )),
    symbol text not null check (symbol ~ '^[A-Z0-9]{3,20}$'),
    interval text not null check (interval ~ '^[1-9][0-9]*[mhd]$'),
    candle_open_time timestamptz,
    prediction_id text references public.predictions (prediction_id),
    new_candle boolean not null default false,
    new_prediction boolean not null default false,
    candles_ingested integer not null default 0 check (candles_ingested >= 0),
    error_type text,
    error_message text,
    evaluation_error_type text,
    evaluation_error_message text,
    created_at timestamptz not null default now(),
    constraint pipeline_runs_time_order check (finished_at >= started_at),
    constraint pipeline_runs_error_consistency check (
        (status = 'FAILED' and error_type is not null)
        or (status <> 'FAILED' and error_type is null and error_message is null)
    )
);
create index pipeline_runs_recent on public.pipeline_runs (symbol, started_at desc);
create index pipeline_runs_status_recent on public.pipeline_runs (symbol, status, started_at desc);

create table public.live_evaluations (
    evaluation_id uuid primary key,
    prediction_id text not null unique references public.predictions (prediction_id),
    symbol text not null check (symbol ~ '^[A-Z0-9]{3,20}$'),
    evaluated_at timestamptz not null,
    source_close numeric not null check (source_close > 0),
    target_open_time timestamptz not null,
    target_close_time timestamptz not null,
    target_close numeric not null check (target_close > 0),
    observed_move text not null check (observed_move in ('UP', 'DOWN', 'FLAT')),
    observed_class text not null check (observed_class in ('UP', 'DOWN')),
    correct boolean not null,
    created_at timestamptz not null default now(),
    constraint live_evaluations_closed_target check (
        target_close_time > target_open_time and evaluated_at > target_close_time
    ),
    constraint live_evaluations_binary_target check (
        observed_class = case when target_close > source_close then 'UP' else 'DOWN' end
    ),
    constraint live_evaluations_move check (
        observed_move = case when target_close > source_close then 'UP'
            when target_close < source_close then 'DOWN' else 'FLAT' end
    )
);
create index live_evaluations_recent on public.live_evaluations (symbol, evaluated_at desc);

alter table public.pipeline_runs enable row level security;
alter table public.live_evaluations enable row level security;
revoke all on public.pipeline_runs, public.live_evaluations from public, anon, authenticated;
grant all on public.pipeline_runs, public.live_evaluations to service_role;
-- Evaluations contain observed outcomes only. Technical errors in pipeline_runs stay private.
grant select on public.live_evaluations to anon, authenticated;
create policy live_evaluations_public_read on public.live_evaluations
for select to anon, authenticated using (true);
