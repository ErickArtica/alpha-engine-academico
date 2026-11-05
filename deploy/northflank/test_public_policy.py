"""Offline execution-path tests: no real Supabase, DNS, provider or sockets."""
import base64
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import patch

import httpx

from deploy.northflank.public_config import PublicConfig, ConfigurationError
from deploy.northflank import public_data as data
from public_test_support import PUBLIC_ENV, mock_transport, synthetic_rows


class PublicPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PublicConfig.from_environment(PUBLIC_ENV)
        self.rows = synthetic_rows()
        self.requests = []

    def reader(self, **kw):
        return data.PublicReader(self.cfg, transport=mock_transport(self.rows, self.requests, **kw))

    def test_explicit_profile_data_approval_and_no_action_credentials(self):
        invalid = [("NORTHFLANK_PUBLIC_READ_ONLY", "false"), ("NORTHFLANK_PUBLIC_DATA_APPROVED", "false"),
                   ("RECOMMENDATION_API_URL", "https://synthetic.invalid/api/recommendation"),
                   ("RECOMMENDATION_API_KEY", "synthetic"), ("NVIDIA_API_KEY", "synthetic"),
                   ("RECOMMENDATION_API_SECRET", "synthetic"), ("GITHUB_TOKEN", "synthetic"),
                   ("ALPHA_VANTAGE_API_KEY", "synthetic"), ("OPENAI_API_KEY", "synthetic"),
                   ("DATABASE_URL", "synthetic"), ("SUPABASE_KEY", "sb_secret_synthetic"),
                   ("MODEL_CONFIDENCE_THRESHOLD", "0.61"), ("MODEL_CONFIDENCE_THRESHOLD", "NaN"),
                   ("PRESCRIPTIVE_NEWS_ENABLED", "true"), ("STORAGE_PROVIDER", "postgresql")]
        for name, value in invalid:
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                PublicConfig.from_environment({**PUBLIC_ENV, name: value})
        with self.assertRaises(ConfigurationError):
            PublicConfig.from_environment({**PUBLIC_ENV, "openai_api_key": "synthetic"})
        self.assertNotIn(self.cfg.supabase_key, repr(self.cfg))

    def test_legacy_anon_allowed_but_service_role_rejected(self):
        for role in ("anon", "service_role", "authenticated"):
            claims = base64.urlsafe_b64encode(json.dumps({"role": role}).encode()).decode().rstrip("=")
            env = {**PUBLIC_ENV, "SUPABASE_KEY": "synthetic." + claims + ".synthetic"}
            if role == "anon":
                self.assertEqual(PublicConfig.from_environment(env).supabase_key, env["SUPABASE_KEY"])
            else:
                with self.assertRaises(ConfigurationError):
                    PublicConfig.from_environment(env)

    def test_three_public_tables_use_only_get_and_reviewed_columns(self):
        with self.reader() as reader:
            self.assertTrue(reader.candles())
            self.assertTrue(reader.predictions())
            self.assertTrue(reader.evaluations())
            sanitized = reader.rows("predictions", 20)
        self.assertNotIn("context_json", sanitized[0])
        self.assertEqual({r.method for r in self.requests}, {"GET"})
        self.assertEqual({r.url.path for r in self.requests}, {"/rest/v1/" + t for t in data.TABLES})
        for request in self.requests:
            self.assertEqual(request.url.host, "synthetic.invalid")
            self.assertNotEqual(request.url.params["select"], "*")

    def test_mutations_are_rejected_before_transport(self):
        with self.reader() as reader:
            for method in ("POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
                with self.subTest(method=method), self.assertRaises(data.ReadOnlyViolation):
                    reader._http.request(method, "/rest/v1/candles", json={"synthetic": True})
        self.assertEqual(self.requests, [])

    def test_private_paths_rpc_traversal_and_cross_origin_rejected(self):
        paths = ["/rest/v1/recommendations", "/rest/v1/recommendation_requests", "/rest/v1/pipeline_runs",
                 "/rest/v1/paper_accounts", "/rest/v1/paper_trades", "/rest/v1/rpc/paper_apply",
                 "/rest/v1/rpc/pending_live_evaluations", "/api/recommendation", "/api/paper", "/auth/v1/token",
                 "/rest/v1/%63andles", "/rest/v1/candles/../recommendations", "https://other.invalid/rest/v1/candles"]
        with self.reader() as reader:
            for path in paths:
                with self.subTest(path=path), self.assertRaises(data.ReadOnlyViolation):
                    reader._http.get(path)
        self.assertEqual(self.requests, [])

    def test_query_injection_duplicate_select_and_unbounded_reads_rejected(self):
        with self.reader() as reader:
            reader.predictions(1)
            original = self.requests.pop()
            params = list(original.url.params.multi_items())
            attacks = [params + [("select", "*")], params + [("or", "(true)")],
                       [(k, "*" if k == "select" else v) for k, v in params],
                       [(k, "1000000" if k == "limit" else v) for k, v in params],
                       [(k, "eq.xgboost_v3" if k == "model_version" else v) for k, v in params]]
            for attack in attacks:
                with self.assertRaises(data.ReadOnlyViolation):
                    reader._http.get("/rest/v1/predictions", params=attack)
            with self.assertRaises(data.ReadOnlyViolation):
                reader.rows("candles", 1001)
        self.assertEqual(self.requests, [])

    def test_redirects_never_forward_credentials(self):
        requests = []
        def redirect(request):
            requests.append(request)
            return httpx.Response(302, headers={"location": "https://other.invalid/rest/v1/paper_accounts"})
        with data.PublicReader(self.cfg, transport=httpx.MockTransport(redirect)) as reader:
            with self.assertRaises(data.ReadUnavailable):
                reader.predictions()
        self.assertEqual(len(requests), 1)

    def test_storage_error_suppresses_payload_and_credential(self):
        def failing(request):
            return httpx.Response(500, json={"message": self.cfg.supabase_key})
        with data.PublicReader(self.cfg, transport=httpx.MockTransport(failing)) as reader:
            with self.assertRaises(data.ReadUnavailable) as error:
                reader.predictions()
        self.assertNotIn(self.cfg.supabase_key, str(error.exception))

    def test_recent_metrics_are_bounded_reads_not_persistent_evaluation(self):
        with patch.object(data.LiveEvaluationService, "evaluate_pending", side_effect=AssertionError("PERSISTENCE_FORBIDDEN")), self.reader() as reader:
            result = reader.recent_metrics()
        self.assertEqual(result["evaluated"], 1)
        self.assertEqual({int(r.url.params["limit"]) for r in self.requests}, {1000, 500})
        self.assertEqual(len(self.requests), 2)

    def test_prescriptive_policy_is_pure_and_abstains_without_fresh_context(self):
        with self.reader() as reader:
            candle = reader.candles(1)[0]
            prediction = reader.predictions(1)[0]
        now = datetime.now(timezone.utc)
        self.assertEqual(data.prescriptive_scenarios(prediction, candle, now)[0]["accion"], "BUY")
        invalid = [prediction.model_copy(update={"valid_until": now - timedelta(seconds=1)}),
                   prediction.model_copy(update={"generated_at": now + timedelta(minutes=1)}),
                   prediction.model_copy(update={"timestamp": prediction.timestamp - timedelta(minutes=1)}),
                   prediction.model_copy(update={"prob_up": 0.59, "prob_down": 0.41})]
        for row in invalid:
            self.assertEqual(data.prescriptive_scenarios(row, candle, now)[0]["accion"], "WAIT")
        self.assertEqual([s["accion"] for s in data.prescriptive_scenarios(prediction, None, now)], ["WAIT", "HOLD"])


if __name__ == "__main__":
    unittest.main()
