-- Optional targeted inspection only. No provider calls or mutating application RPC.
BEGIN READ ONLY;
SET LOCAL statement_timeout = '10s';
SET LOCAL lock_timeout = '2s';
SELECT current_database() AS database_name, current_user AS inspection_role;
SELECT c.relname, c.relkind, pg_get_userbyid(c.relowner) AS owner, c.relrowsecurity AS rls,
       has_table_privilege('anon', c.oid, 'SELECT') AS anon_select,
       has_table_privilege('authenticated', c.oid, 'SELECT') AS authenticated_select,
       has_table_privilege('service_role', c.oid, 'SELECT,UPDATE') AS server_any_select_update
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relname IN ('prescriptive_news_fetches', 'academic_demo_state');
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public' AND
      ((table_name = 'prescriptive_news_fetches' AND column_name IN ('provider','snapshot_id','marketaux_payload','status'))
       OR table_name = 'academic_demo_state');
SELECT p.proname, pg_get_function_identity_arguments(p.oid) AS arguments,
       pg_get_userbyid(p.proowner) AS owner, p.prosecdef AS security_definer,
       p.provolatile, p.proconfig, md5(p.prosrc) AS body_fingerprint,
       has_function_privilege('anon', p.oid, 'EXECUTE') AS anon_execute,
       has_function_privilege('authenticated', p.oid, 'EXECUTE') AS authenticated_execute,
       has_function_privilege('service_role', p.oid, 'EXECUTE') AS server_execute
FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public' AND p.proname IN
      ('claim_marketaux_news','academic_demo_state_read','academic_demo_state_cas');
SELECT tablename, policyname, roles, cmd, qual, with_check
FROM pg_policies WHERE schemaname = 'public'
AND tablename IN ('prescriptive_news_fetches','academic_demo_state');
ROLLBACK;
