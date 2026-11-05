-- REVIEW ONLY: disables DEMO first, then removes ONLY its new private state.
-- Discards quotas, revocations and cached explanations. Rotate session secret and
-- approve a NEW bounded presentation window before any later reactivation.
BEGIN;
DROP FUNCTION IF EXISTS public.academic_demo_state_cas(bigint, jsonb);
DROP FUNCTION IF EXISTS public.academic_demo_state_read();
DROP TABLE IF EXISTS public.academic_demo_state;
COMMIT;
