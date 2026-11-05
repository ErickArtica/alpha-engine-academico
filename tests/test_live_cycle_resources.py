"""Offline regressions for process-owned LIVE infrastructure."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alpha_engine.config.settings import Settings
from alpha_engine.services.live_cycle import LiveCycleResources
from alpha_engine.services.prediction import PredictionService, load_model_bundle
from deploy.render.app import RenderApp
from tests.test_prediction_service import FakeMarket, FakePredictions, candles


class Query:
    data = []

    def __getattr__(self, name):
        if name == "execute":
            return lambda: self
        return lambda *args, **kwargs: self


class Client:
    def __init__(self):
        self.postgrest = self
        self.close_count = 0

    def table(self, name):
        return Query()

    def rpc(self, name, args):
        return Query()

    def aclose(self):
        self.close_count += 1


class Source:
    def __init__(self, *args):
        self.closed = False

    def fetch_page(self, *args, **kwargs):
        return []

    def close(self):
        self.closed = True


def settings():
    return Settings(_env_file=None, storage_provider="supabase",
                    supabase_url="https://example.supabase.co",
                    supabase_key="server-key")


def test_sixty_cycles_share_one_client_and_close_it_once(monkeypatch):
    import alpha_engine.repositories.supabase as supabase_adapter
    import alpha_engine.services.live_cycle as live_cycle

    clients = []
    sources = []
    monkeypatch.setattr(supabase_adapter, "_client_from_settings",
                        lambda _: clients.append(Client()) or clients[-1])
    monkeypatch.setattr(live_cycle, "BinanceKlineClient",
                        lambda *args: sources.append(Source()) or sources[-1])
    resources = LiveCycleResources(settings())
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: resources.run(settings(), now=now), range(60)))
    assert all(result.status == "NO_NEW_CANDLE" for result in results)
    assert len(clients) == 1
    assert len(sources) == 60 and all(source.closed for source in sources)
    assert clients[0].close_count == 0
    resources.close()
    resources.close()
    assert clients[0].close_count == 1


def test_shared_model_matches_uncached_prediction_and_loads_once(monkeypatch):
    import alpha_engine.repositories.supabase as supabase_adapter
    import alpha_engine.services.live_cycle as live_cycle

    monkeypatch.setattr(supabase_adapter, "_client_from_settings", lambda _: Client())
    model = Path(__file__).resolve().parents[1] / "models/xgboost_v1.joblib"
    load_count = 0
    original = load_model_bundle

    def counted(path):
        nonlocal load_count
        load_count += 1
        return original(path)

    monkeypatch.setattr(live_cycle, "load_model_bundle", counted)
    resources = LiveCycleResources(settings())
    with ThreadPoolExecutor(max_workers=4) as pool:
        bundles = list(pool.map(resources.model_bundle, [model] * 8))
    assert all(bundle is bundles[0] for bundle in bundles)
    rows = candles(300)
    old = PredictionService(FakeMarket(rows), FakePredictions(), model).predict_latest("BTCUSDT")
    shared = PredictionService(FakeMarket(rows), FakePredictions(), model,
                               loader=resources.model_bundle)
    new = shared.predict_latest("BTCUSDT")
    assert old.predicted_class == new.predicted_class
    assert new.prob_up == pytest.approx(old.prob_up, abs=1e-12)
    assert new.prob_down == pytest.approx(old.prob_down, abs=1e-12)
    assert new.prediction_id == old.prediction_id
    assert resources.model_bundle(model) is resources.model_bundle(model)
    assert load_count == 1
    resources.close()


def test_render_app_owns_one_container_until_lifespan_shutdown(monkeypatch):
    import deploy.render.app as render_app

    instances = []

    class Container:
        def __init__(self, selected):
            self.calls = 0
            self.closed = False
            instances.append(self)

        def run(self, selected):
            self.calls += 1
            return object()

        def close(self):
            self.closed = True

    monkeypatch.setattr(render_app, "LiveCycleResources", Container)
    app = RenderApp()
    selected = settings()
    assert app._run_cycle(selected) is not None
    assert app._run_cycle(selected) is not None
    assert len(instances) == 1 and instances[0].calls == 2

    async def shutdown():
        async def receive():
            return {"type": "lifespan.shutdown"}

        async def send(event):
            assert event == {"type": "lifespan.shutdown.complete"}

        await app({"type": "lifespan"}, receive, send)

    import asyncio
    asyncio.run(shutdown())
    assert instances[0].closed and app._live_resources is None
