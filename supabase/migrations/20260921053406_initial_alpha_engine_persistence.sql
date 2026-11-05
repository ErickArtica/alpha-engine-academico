-- Alpha Engine persistence. All instants are timestamptz; the application sends UTC.
-- Only the trusted server key can write. Public API readers see market data and
-- predictions, never LLM context or response payloads.

create table public.candles (
    symbol text not null check (symbol ~ '^[A-Z0-9]{3,20}$'),
    interval text not null check (interval ~ '^[1-9][0-9]*[mhd]$'),
    open_time timestamptz not null,
    close_time timestamptz not null check (close_time > open_time),
    open numeric not null check (open > 0),
    high numeric not null check (high > 0),
    low numeric not null check (low > 0),
    close numeric not null check (close > 0),
    volume numeric not null check (volume >= 0),
    primary key (symbol, interval, open_time),
    check (high >= greatest(open, low, close)),
    check (low <= least(open, high, close))
);

-- The PK covers latest-candle and [start, stop) range lookups by symbol/interval.
create table public.predictions (
    prediction_id text primary key,
    symbol text not null,
    interval text not null default '1m' check (interval = '1m'),
    candle_open_time timestamptz not null,
    horizon text not null check (horizon = '1m'),
    prob_up numeric not null check (prob_up between 0 and 1),
    prob_down numeric not null check (prob_down between 0 and 1),
    predicted_class text not null check (predicted_class in ('UP', 'DOWN')),
    model_version text not null,
    trained_at timestamptz,
    foreign key (symbol, interval, candle_open_time)
        references public.candles (symbol, interval, open_time),
    constraint predictions_logical_identity
        unique nulls not distinct (symbol, interval, candle_open_time, model_version, trained_at),
    check (abs(prob_up + prob_down - 1) <= 0.000001),
    check (predicted_class = case when prob_up > prob_down then 'UP' else 'DOWN' end)
);

create index predictions_latest_by_symbol
    on public.predictions (symbol, candle_open_time desc);

create table public.recommendations (
    prediction_id text not null references public.predictions (prediction_id),
    prompt_version text not null,
    llm_model text not null,
    candle_open_time timestamptz not null,
    generated_at timestamptz not null,
    candidate_signal text not null check (candidate_signal in ('BUY', 'SELL', 'HOLD')),
    llm_signal text check (llm_signal in ('BUY', 'SELL', 'HOLD')),
    final_signal text not null check (final_signal in ('BUY', 'SELL', 'HOLD')),
    risk_level text not null check (risk_level in ('LOW', 'MEDIUM', 'HIGH')),
    context_json text not null,
    response_json text,
    guardrails_json text not null default '[]',
    primary key (prediction_id, prompt_version, llm_model)
);

create table public.recommendation_history (
    revision_id uuid primary key default gen_random_uuid(),
    prediction_id text not null,
    prompt_version text not null,
    llm_model text not null,
    candle_open_time timestamptz not null,
    generated_at timestamptz not null,
    candidate_signal text not null check (candidate_signal in ('BUY', 'SELL', 'HOLD')),
    llm_signal text check (llm_signal in ('BUY', 'SELL', 'HOLD')),
    final_signal text not null check (final_signal in ('BUY', 'SELL', 'HOLD')),
    risk_level text not null check (risk_level in ('LOW', 'MEDIUM', 'HIGH')),
    context_json text not null,
    response_json text,
    guardrails_json text not null default '[]',
    foreign key (prediction_id, prompt_version, llm_model)
        references public.recommendations (prediction_id, prompt_version, llm_model)
);

create index recommendation_history_recent
    on public.recommendation_history (generated_at desc, revision_id desc);

-- One API upsert updates the cache and records its revision in the same DB transaction.
create function public.record_recommendation_revision()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    insert into public.recommendation_history (
        prediction_id, prompt_version, llm_model, candle_open_time, generated_at,
        candidate_signal, llm_signal, final_signal, risk_level,
        context_json, response_json, guardrails_json
    ) values (
        new.prediction_id, new.prompt_version, new.llm_model, new.candle_open_time,
        new.generated_at, new.candidate_signal, new.llm_signal, new.final_signal,
        new.risk_level, new.context_json, new.response_json, new.guardrails_json
    );
    return new;
end;
$$;

create trigger recommendation_insert_revision
after insert on public.recommendations
for each row execute function public.record_recommendation_revision();

create trigger recommendation_update_revision
after update on public.recommendations
for each row when (old is distinct from new)
execute function public.record_recommendation_revision();

revoke all on function public.record_recommendation_revision()
    from public, anon, authenticated;

alter table public.candles enable row level security;
alter table public.predictions enable row level security;
alter table public.recommendations enable row level security;
alter table public.recommendation_history enable row level security;

-- Supabase may apply broad default grants in public; revoke them explicitly.
revoke all on public.candles, public.predictions, public.recommendations,
    public.recommendation_history from public, anon, authenticated;
grant select on public.candles, public.predictions to anon, authenticated;
grant all on public.candles, public.predictions, public.recommendations,
    public.recommendation_history to service_role;

create policy candles_public_read on public.candles
for select to anon, authenticated using (true);
create policy predictions_public_read on public.predictions
for select to anon, authenticated using (true);

-- No anon/authenticated policies or grants exist for LLM records or writes.
