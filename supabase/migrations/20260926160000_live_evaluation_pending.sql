-- Return only LIVE predictions with both persisted H1 candles available.
-- The strict close_time comparison matches LiveEvaluationService._closed_pair.
create or replace function public.pending_live_evaluations(
    p_symbol text, p_now timestamptz, p_limit integer default 100
)
returns table (
    prediction_id text,
    symbol text,
    predicted_class text,
    source_close numeric,
    target_open_time timestamptz,
    target_close_time timestamptz,
    target_close numeric
)
language sql stable
set search_path = ''
as $$
    select p.prediction_id, p.symbol, p.predicted_class,
           source.close, target.open_time, target.close_time, target.close
    from public.predictions as p
    join public.candles as source
      on source.symbol = p.symbol and source.interval = p.interval
     and source.open_time = p.candle_open_time
    join public.candles as target
      on target.symbol = p.symbol and target.interval = p.interval
     and target.open_time = p.candle_open_time + interval '1 minute'
    where p.symbol = p_symbol and p.interval = '1m'
      and p.generated_at is not null
      and target.close_time < p_now
      and not exists (
          select 1 from public.live_evaluations as e
          where e.prediction_id = p.prediction_id
      )
      and p_limit between 1 and 1000
    -- Oldest first guarantees progress through a backlog under a fixed limit.
    order by p.candle_open_time asc, p.prediction_id asc
    limit p_limit;
$$;

revoke all on function public.pending_live_evaluations(text, timestamptz, integer)
    from public, anon, authenticated;
grant execute on function public.pending_live_evaluations(text, timestamptz, integer)
    to service_role;
