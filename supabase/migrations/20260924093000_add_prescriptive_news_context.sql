-- Private, optional NEWS_SENTIMENT context for on-demand recommendations.
-- No NEWS_SENTIMENT data enters candles, predictions or public dashboard views.
create table public.prescriptive_news_fetches (
    snapshot_id uuid primary key,
    slot_start timestamptz not null unique,
    attempted_at timestamptz not null,
    fetched_at timestamptz,
    status text not null check (status in ('CLAIMED', 'SUCCESS', 'EMPTY', 'RATE_LIMIT', 'FAILED')),
    article_count integer not null default 0 check (article_count >= 0),
    check (status not in ('SUCCESS', 'EMPTY') or fetched_at is not null),
    check (fetched_at is null or fetched_at >= attempted_at)
);

create index prescriptive_news_fetches_recent
    on public.prescriptive_news_fetches (fetched_at desc)
    where status in ('SUCCESS', 'EMPTY');

create table public.prescriptive_news_articles (
    article_identifier text primary key check (article_identifier ~ '^[0-9a-f]{64}$'),
    provider text not null check (provider = 'ALPHA_VANTAGE'),
    published_at timestamptz not null,
    fetched_at timestamptz not null,
    first_observed_at timestamptz not null,
    btc_sentiment_score numeric not null check (btc_sentiment_score between -1 and 1),
    btc_relevance_score numeric not null check (btc_relevance_score > 0 and btc_relevance_score <= 1),
    source text not null default '',
    check (published_at <= first_observed_at),
    check (first_observed_at <= fetched_at)
);

create index prescriptive_news_articles_published
    on public.prescriptive_news_articles (published_at desc);

alter table public.prescriptive_news_fetches enable row level security;
alter table public.prescriptive_news_articles enable row level security;
revoke all on public.prescriptive_news_fetches, public.prescriptive_news_articles
    from public, anon, authenticated;
grant select, insert, update on public.prescriptive_news_fetches,
    public.prescriptive_news_articles to service_role;
create policy prescriptive_news_fetches_server on public.prescriptive_news_fetches
    for all to service_role using (true) with check (true);
create policy prescriptive_news_articles_server on public.prescriptive_news_articles
    for all to service_role using (true) with check (true);

-- Recommendation context_json/history already store the private snapshot for
-- successful recommendations. These nullable fields also audit failed attempts.
alter table public.recommendation_requests
    add column news_context_id uuid references public.prescriptive_news_fetches (snapshot_id),
    add column news_context_used boolean not null default false,
    add column news_context_fetched_at timestamptz,
    add column news_context_count integer check (news_context_count >= 0),
    add column news_context_sentiment numeric check (news_context_sentiment between -1 and 1);
create index recommendation_requests_news_context
    on public.recommendation_requests (news_context_id)
    where news_context_id is not null;
