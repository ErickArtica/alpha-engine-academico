-- Local-only PMV storage. Executed once by the official PostgreSQL image.
CREATE TABLE IF NOT EXISTS candles (
  symbol text NOT NULL, interval text NOT NULL, open_time timestamptz NOT NULL,
  close_time timestamptz NOT NULL, open double precision NOT NULL,
  high double precision NOT NULL, low double precision NOT NULL,
  close double precision NOT NULL, volume double precision NOT NULL,
  PRIMARY KEY (symbol, interval, open_time)
);
CREATE TABLE IF NOT EXISTS predictions (
  prediction_id text PRIMARY KEY, symbol text NOT NULL, interval text NOT NULL,
  candle_open_time timestamptz NOT NULL, horizon text NOT NULL,
  prob_up double precision NOT NULL, prob_down double precision NOT NULL,
  predicted_class text NOT NULL, model_version text NOT NULL,
  trained_at timestamptz, generated_at timestamptz, valid_until timestamptz,
  UNIQUE (symbol, interval, candle_open_time, model_version, trained_at)
);
CREATE INDEX IF NOT EXISTS predictions_recent ON predictions (symbol, candle_open_time DESC);
CREATE TABLE IF NOT EXISTS live_evaluations (
  evaluation_id uuid PRIMARY KEY, prediction_id text UNIQUE NOT NULL REFERENCES predictions(prediction_id),
  symbol text NOT NULL, evaluated_at timestamptz NOT NULL,
  source_close double precision NOT NULL, target_open_time timestamptz NOT NULL,
  target_close_time timestamptz NOT NULL, target_close double precision NOT NULL,
  observed_move text NOT NULL, observed_class text NOT NULL, correct boolean NOT NULL
);
CREATE INDEX IF NOT EXISTS evaluations_recent ON live_evaluations (symbol, evaluated_at DESC);
CREATE TABLE IF NOT EXISTS pipeline_runs (
  run_id uuid PRIMARY KEY, started_at timestamptz NOT NULL, finished_at timestamptz NOT NULL,
  duration_ms double precision NOT NULL, market_data_ms double precision,
  prediction_ms double precision, evaluation_ms double precision,
  status text NOT NULL, symbol text NOT NULL, interval text NOT NULL,
  candle_open_time timestamptz, prediction_id text, new_candle boolean NOT NULL DEFAULT false,
  new_prediction boolean NOT NULL DEFAULT false, candles_ingested integer NOT NULL DEFAULT 0,
  error_type text, error_message text, evaluation_error_type text,
  evaluation_error_message text, created_at timestamptz
);
CREATE INDEX IF NOT EXISTS runs_recent ON pipeline_runs (symbol, started_at DESC);
CREATE TABLE IF NOT EXISTS recommendations (
  prediction_id text NOT NULL, prompt_version text NOT NULL, llm_model text NOT NULL,
  candle_open_time timestamptz NOT NULL, generated_at timestamptz NOT NULL,
  requested_at timestamptz, valid_until timestamptz, llm_latency_ms double precision,
  total_latency_ms double precision, candidate_signal text NOT NULL,
  llm_signal text, final_signal text NOT NULL, risk_level text NOT NULL,
  context_json text NOT NULL, response_json text, guardrails_json text NOT NULL,
  PRIMARY KEY (prediction_id, prompt_version, llm_model)
);
CREATE TABLE IF NOT EXISTS recommendation_requests (
  request_id uuid PRIMARY KEY, prediction_id text, symbol text NOT NULL,
  interval text NOT NULL, provider text NOT NULL, llm_model text NOT NULL,
  prompt_version text NOT NULL, requested_at timestamptz NOT NULL,
  generated_at timestamptz, finished_at timestamptz NOT NULL,
  valid_until timestamptz, llm_latency_ms double precision,
  total_latency_ms double precision NOT NULL, status text NOT NULL,
  timing_status text, cached boolean NOT NULL DEFAULT false,
  error_type text, error_message text, created_at timestamptz,
  news_context_id uuid, news_context_used boolean NOT NULL DEFAULT false,
  news_context_fetched_at timestamptz, news_context_count integer,
  news_context_sentiment double precision
);
CREATE INDEX IF NOT EXISTS requests_recent ON recommendation_requests (symbol, requested_at DESC);
CREATE TABLE IF NOT EXISTS prescriptive_news_fetches (
  snapshot_id uuid PRIMARY KEY, slot_start timestamptz UNIQUE NOT NULL,
  attempted_at timestamptz NOT NULL, fetched_at timestamptz,
  status text NOT NULL, article_count integer NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS prescriptive_news_articles (
  article_identifier text PRIMARY KEY, provider text NOT NULL,
  published_at timestamptz NOT NULL, fetched_at timestamptz NOT NULL,
  first_observed_at timestamptz NOT NULL, btc_sentiment_score double precision NOT NULL,
  btc_relevance_score double precision NOT NULL, source text
);
CREATE INDEX IF NOT EXISTS news_recent ON prescriptive_news_articles (published_at DESC);
CREATE TABLE IF NOT EXISTS local_bootstrap_runs (
  id integer PRIMARY KEY CHECK (id=1), completed_at timestamptz NOT NULL,
  source_start timestamptz NOT NULL, source_end timestamptz NOT NULL,
  candle_count integer NOT NULL, prediction_count integer NOT NULL,
  evaluation_count integer NOT NULL, kind text NOT NULL DEFAULT 'LOCAL_BOOTSTRAP'
);
CREATE TABLE IF NOT EXISTS local_demo_requests (
  request_id uuid PRIMARY KEY, prediction_id text NOT NULL,
  requested_at timestamptz NOT NULL, status text NOT NULL DEFAULT 'DEMO_ONLY',
  news_status text, provider text, error_type text
);
