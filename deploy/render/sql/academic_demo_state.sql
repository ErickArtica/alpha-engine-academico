-- REVIEW ONLY. Never run from application startup or the prediction pipeline.
-- Requires separate approval, inspection of existing names, and service_role backend credentials.
-- Only new academic_demo_* objects. No grants/revokes on existing project objects.
BEGIN;
CREATE TABLE IF NOT EXISTS public.academic_demo_state (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    version bigint NOT NULL DEFAULT 0 CHECK (version >= 0),
    payload jsonb NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(payload) = 'object' AND octet_length(payload::text) <= 262144)
);
ALTER TABLE public.academic_demo_state ENABLE ROW LEVEL SECURITY;
-- Supabase defaults can grant ALL to service_role on a newly created table.
-- Reset ONLY this new object's ACL before granting the two required operations.
REVOKE ALL ON TABLE public.academic_demo_state FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT, UPDATE ON TABLE public.academic_demo_state TO service_role;
INSERT INTO public.academic_demo_state(singleton) VALUES (true) ON CONFLICT DO NOTHING;

CREATE OR REPLACE FUNCTION public.academic_demo_state_read()
RETURNS jsonb LANGUAGE sql STABLE SECURITY INVOKER SET search_path = pg_catalog
AS $function$
    SELECT jsonb_build_object('version', version, 'payload', payload)
    FROM public.academic_demo_state WHERE singleton;
$function$;

CREATE OR REPLACE FUNCTION public.academic_demo_state_cas(p_version bigint, p_payload jsonb)
RETURNS boolean LANGUAGE plpgsql VOLATILE SECURITY INVOKER SET search_path = pg_catalog
AS $function$
DECLARE changed integer;
BEGIN
    IF p_version < 0 OR p_payload IS NULL OR jsonb_typeof(p_payload) <> 'object'
       OR octet_length(p_payload::text) > 262144 THEN
        RAISE EXCEPTION 'Invalid academic demo state';
    END IF;
    UPDATE public.academic_demo_state
       SET version = version + 1, payload = p_payload
     WHERE singleton AND version = p_version;
    GET DIAGNOSTICS changed = ROW_COUNT;
    RETURN changed = 1;
END;
$function$;
REVOKE ALL ON FUNCTION public.academic_demo_state_read() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.academic_demo_state_cas(bigint, jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.academic_demo_state_read() TO service_role;
GRANT EXECUTE ON FUNCTION public.academic_demo_state_cas(bigint, jsonb) TO service_role;
COMMIT;
