"""Unit tests for ExperimentRegistry adapters and utilities."""

import httpx
from httpx import Response, Request
import pytest

from prompti.experiment import bucket, UnleashRegistry, GrowthBookRegistry
from prompti.engine import PromptEngine, Setting
from prompti.model_client import ModelClient, ModelConfig
from prompti import Message


def test_bucket_deterministic():
    split = {"A": 0.5, "B": 0.5}
    assert bucket("user1", split) == bucket("user1", split)
    assert bucket("user1", split) in {"A", "B"}


@pytest.mark.asyncio
async def test_unleash_registry():
    async def handler(request: Request):
        assert request.url.path == "/client/features/clarify"
        return Response(200, json={"name": "clarify", "variant": {"name": "A"}})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    reg = UnleashRegistry("http://unleash", client=client)
    split = await reg.get_split("clarify", "u1")
    assert split.experiment_id == "clarify"
    assert split.variant == "A"


@pytest.mark.asyncio
async def test_growthbook_registry():
    features = {"clarify": {"id": "clarify", "variants": {"A": 0.5, "B": 0.5}}}
    reg = GrowthBookRegistry(features)
    split = await reg.get_split("clarify", "user1")
    assert split.traffic_split == {"A": 0.5, "B": 0.5}
    assert split.variant is None


@pytest.mark.asyncio
async def test_engine_sdk_split(tmp_path):
    """Engine should pick variant using registry weights."""
    features = {"support_reply": {"id": "clarify", "variants": {"A": 1.0}}}
    reg = GrowthBookRegistry(features)
    settings = Setting(template_paths=["./prompts"])
    engine = PromptEngine.from_setting(settings)
    class Dummy(ModelClient):
        provider = "dummy"

        def __init__(self):
            super().__init__(client=httpx.AsyncClient(http2=False))

        async def _run(self, messages, model_cfg):
            yield Message(role="assistant", kind="text", content="ok")

    dummy = Dummy()
    cfg = ModelConfig(provider="dummy", model="x")
    msgs = engine.run(
        "support_reply",
        {"name": "Bob", "issue": "none"},
        None,
        model_cfg=cfg,
        client=dummy,
        registry=reg,
    )
    out = [m async for m in msgs]
    assert out[-1].content == "ok"
