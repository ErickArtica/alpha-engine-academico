-- One-minute trigger only. LiveCycle remains in the Render web service.
-- The destination and bearer token are provisioned separately in Supabase Vault.
-- No credential or project-specific URL is embedded in this migration.
create schema if not exists extensions;
create schema if not exists vault;
create extension if not exists supabase_vault with schema vault;
create extension if not exists pg_net with schema extensions;
create extension if not exists pg_cron;

revoke all on vault.secrets, vault.decrypted_secrets from public, anon, authenticated;

select cron.schedule(
    'alpha-engine-live-cycle',
    '* * * * *',
    $job$
    do $cycle$
    declare
        endpoint_url text;
        bearer_token text;
    begin
        select decrypted_secret into endpoint_url
        from vault.decrypted_secrets
        where name = 'alpha_engine_render_cycle_url';

        select decrypted_secret into bearer_token
        from vault.decrypted_secrets
        where name = 'alpha_engine_internal_cron_secret';

        if endpoint_url is null or endpoint_url !~ '^https://[a-z0-9-]+\.onrender\.com/internal/run-cycle$' then
            raise exception 'Alpha Engine Cron endpoint missing or invalid in Vault';
        end if;
        if bearer_token is null or length(bearer_token) < 32 then
            raise exception 'Alpha Engine Cron token missing or invalid in Vault';
        end if;

        perform net.http_post(
            url := endpoint_url,
            body := '{}'::jsonb,
            headers := jsonb_build_object(
                'Authorization', 'Bearer ' || bearer_token,
                'Content-Type', 'application/json'
            ),
            timeout_milliseconds := 90000
        );
    end
    $cycle$;
    $job$
);
