-- The worker uses service_role to read the same narrow dashboard projections.
-- The preceding migration revoked the default view grant before granting anon
-- and authenticated. Grant server read explicitly; underlying audit tables stay private.
grant select on public.public_pipeline_runs to service_role;
grant select on public.public_recommendation_request_summary to service_role;
