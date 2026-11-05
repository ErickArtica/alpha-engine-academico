-- PASO 6: narrow read projections for a public Streamlit dashboard.
-- Underlying audit tables remain private; these views omit messages, prompts,
-- LLM context and raw provider responses.

create view public.public_pipeline_runs
with (security_invoker = false)
as
select
    run_id, started_at, finished_at, duration_ms, market_data_ms, prediction_ms,
    evaluation_ms, status, symbol, interval, candle_open_time, prediction_id,
    new_candle, new_prediction, candles_ingested,
    case
        when error_type is null then null
        when lower(error_type) like '%market%' or lower(error_type) like '%binance%' then 'MARKET_DATA'
        when lower(error_type) like '%storage%' or lower(error_type) like '%database%' then 'STORAGE'
        when lower(error_type) like '%model%' or lower(error_type) like '%prediction%' then 'MODEL'
        when lower(error_type) like '%validation%' or lower(error_type) like '%value%' then 'VALIDATION'
        else 'UNEXPECTED'
    end as error_type,
    null::text as error_message,
    null::text as evaluation_error_type,
    null::text as evaluation_error_message,
    created_at
from public.pipeline_runs;

create view public.public_recommendation_request_summary
with (security_invoker = false)
as
select
    request_id, prediction_id, symbol, interval, provider, llm_model, prompt_version,
    requested_at, generated_at, finished_at, valid_until, llm_latency_ms,
    total_latency_ms, status, timing_status, cached,
    case
        when error_type is null then null
        when lower(error_type) like '%timeout%' then 'LLM_TIMEOUT'
        when lower(error_type) like '%provider%' then 'PROVIDER_ERROR'
        when lower(error_type) like '%validation%' then 'VALIDATION'
        else 'UNEXPECTED'
    end as error_type,
    null::text as error_message,
    created_at
from public.recommendation_requests;

revoke all on public.public_pipeline_runs from public, anon, authenticated;
revoke all on public.public_recommendation_request_summary from public, anon, authenticated;
grant select on public.public_pipeline_runs to anon, authenticated;
grant select on public.public_recommendation_request_summary to anon, authenticated;
