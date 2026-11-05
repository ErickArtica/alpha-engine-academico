-- A denied repeat request is an audited prescriptive attempt, not an LLM result.
alter table public.recommendation_requests
    drop constraint recommendation_requests_status_check;
alter table public.recommendation_requests
    add constraint recommendation_requests_status_check check (status in (
        'SUCCESS', 'CACHED', 'EXPIRED_BEFORE_REQUEST', 'DEADLINE_TOO_CLOSE',
        'LATE_RESULT', 'LLM_TIMEOUT', 'PROVIDER_ERROR', 'INVALID_RESPONSE',
        'GUARDRAIL_REJECTED', 'PREDICTION_NOT_FOUND', 'NO_ACTIVE_PREDICTION',
        'PREDICTION_INVALID', 'PERSISTENCE_ERROR', 'RATE_LIMITED'
    ));
