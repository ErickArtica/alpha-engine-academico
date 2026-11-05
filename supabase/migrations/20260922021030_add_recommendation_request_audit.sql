-- PASO 5: immutable audit of every on-demand prescriptive request.
-- Context and raw validated response remain in private recommendation tables.
alter table public.recommendations
    add column requested_at timestamptz,
    add column valid_until timestamptz,
    add column llm_latency_ms numeric check (llm_latency_ms >= 0),
    add column total_latency_ms numeric check (total_latency_ms >= 0),
    add constraint recommendations_request_before_generation check (
        requested_at is null or generated_at >= requested_at
    );

alter table public.recommendation_history
    add column requested_at timestamptz,
    add column valid_until timestamptz,
    add column llm_latency_ms numeric check (llm_latency_ms >= 0),
    add column total_latency_ms numeric check (total_latency_ms >= 0);

create or replace function public.record_recommendation_revision()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    insert into public.recommendation_history (
        prediction_id, prompt_version, llm_model, candle_open_time, generated_at,
        requested_at, valid_until, llm_latency_ms, total_latency_ms,
        candidate_signal, llm_signal, final_signal, risk_level,
        context_json, response_json, guardrails_json
    ) values (
        new.prediction_id, new.prompt_version, new.llm_model, new.candle_open_time,
        new.generated_at, new.requested_at, new.valid_until, new.llm_latency_ms,
        new.total_latency_ms, new.candidate_signal, new.llm_signal, new.final_signal,
        new.risk_level, new.context_json, new.response_json, new.guardrails_json
    );
    return new;
end;
$$;

create table public.recommendation_requests (
    request_id uuid primary key,
    prediction_id text references public.predictions (prediction_id),
    symbol text not null check (symbol ~ '^[A-Z0-9]{3,20}$'),
    interval text not null check (interval = '1m'),
    provider text not null,
    llm_model text not null,
    prompt_version text not null,
    requested_at timestamptz not null,
    generated_at timestamptz,
    finished_at timestamptz not null,
    valid_until timestamptz,
    llm_latency_ms numeric check (llm_latency_ms >= 0),
    total_latency_ms numeric not null check (total_latency_ms >= 0),
    status text not null check (status in (
        'SUCCESS', 'CACHED', 'EXPIRED_BEFORE_REQUEST', 'DEADLINE_TOO_CLOSE',
        'LATE_RESULT', 'LLM_TIMEOUT', 'PROVIDER_ERROR', 'INVALID_RESPONSE',
        'GUARDRAIL_REJECTED', 'PREDICTION_NOT_FOUND', 'NO_ACTIVE_PREDICTION'
        , 'PREDICTION_INVALID', 'PERSISTENCE_ERROR'
    )),
    timing_status text check (timing_status in ('TIMELY', 'LATE')),
    cached boolean not null default false,
    error_type text,
    error_message text,
    created_at timestamptz not null default now(),
    constraint recommendation_requests_time_order check (finished_at >= requested_at),
    constraint recommendation_requests_generation_order check (
        generated_at is null or generated_at >= requested_at
    ),
    constraint recommendation_requests_timing_fields check (
        timing_status is null or (generated_at is not null and valid_until is not null)
    )
);
create index recommendation_requests_recent
    on public.recommendation_requests (symbol, requested_at desc);
create index recommendation_requests_status_recent
    on public.recommendation_requests (symbol, status, requested_at desc);

alter table public.recommendation_requests enable row level security;
revoke all on public.recommendation_requests from public, anon, authenticated;
grant all on public.recommendation_requests to service_role;
-- No public policy: timestamps, errors and provider audit remain server-only.
