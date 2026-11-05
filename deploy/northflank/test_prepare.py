"""Fast, offline launcher checks: never starts an app, socket, Docker or tunnel."""
from pathlib import Path
import unittest
from urllib.parse import urlunsplit

from deploy.northflank import entrypoint as entry


def configuration():
    return dict(STORAGE_PROVIDER="supabase", NORTHFLANK_PUBLIC_READ_ONLY="true", NORTHFLANK_PUBLIC_DATA_APPROVED="true",
                SUPABASE_URL="https://synthetic.invalid", SUPABASE_KEY="sb_publishable_synthetic_only",
                NF_HOSTS="frontend.example.com,alternative.example.com")


class PreparationChecks(unittest.TestCase):
    def test_northflank_hosts_keep_cors_and_xsrf_enabled(self):
        args = entry.command(configuration())
        self.assertIn("--server.enableCORS=true", args)
        self.assertIn("--server.enableXsrfProtection=true", args)
        self.assertIn("--server.corsAllowedOrigins=https://alternative.example.com", args)
        self.assertNotIn("synthetic-not-a-secret", " ".join(args))

    def test_write_keys_provider_secrets_and_changed_gate_are_rejected(self):
        for key, value in [("SUPABASE_KEY", "sb_secret_synthetic"), ("NVIDIA_API_KEY", "synthetic"),
                           ("DATABASE_URL", "synthetic"), ("MODEL_CONFIDENCE_THRESHOLD", "0.61"),
                           ("PRESCRIPTIVE_NEWS_ENABLED", "true")]:
            with self.subTest(key=key):
                env = configuration();env[key] = value
                with self.assertRaises(entry.ConfigurationError):entry.command(env)

    def test_malformed_host_and_project_url_are_rejected(self):
        for key,value in [("NF_HOSTS", "https://host.example/path"),
                          ("SUPABASE_URL", "http://synthetic.invalid"),
                          ("SUPABASE_URL", urlunsplit(("https", "synthetic-user:synthetic-password@synthetic.invalid", "/", "", "")))]:
            with self.subTest(key=key):
                env=configuration();env[key]=value
                with self.assertRaises(entry.ConfigurationError):entry.command(env)

    def test_existing_model_artifact_matches_its_checksum(self):
        import hashlib
        root=Path(entry.__file__).resolve().parents[2]
        model=root/"models/xgboost_v1.joblib"
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(),model.with_suffix(".sha256").read_text().strip())


if __name__=="__main__":unittest.main()
