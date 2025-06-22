"""Unit tests for ExperimentRegistry adapters and utilities."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from prompti.engine import PromptEngine, Setting
from prompti.experiment import GrowthBookRegistry, UnleashRegistry, bucket
from prompti.message import Message
from prompti.model_client import ModelClient, ModelConfig


def test_bucket_deterministic():
    split = {"A": 0.5, "B": 0.5}
    assert bucket("user1", split) == bucket("user1", split)
    assert bucket("user1", split) in {"A", "B"}


@pytest.mark.asyncio
async def test_unleash_registry():
    # Create a proper mock HTTP response
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "clarify", "variant": {"name": "A"}}
    mock_client.get.return_value = mock_response

    reg = UnleashRegistry("http://unleash", client=mock_client)
    split = await reg.get_split("clarify", "u1")
    assert split.experiment_id == "clarify"
    assert split.variant == "A"

    # Verify the correct URL was called
    mock_client.get.assert_called_once()


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

    # Create a mock client that properly inherits from ModelClient
    class MockClient(ModelClient):
        provider = "mock"

        def __init__(self, cfg: ModelConfig):
            super().__init__(cfg, client=httpx.AsyncClient(http2=False))

        async def _run(self, params):
            yield Message(role="assistant", kind="text", content="ok")

    cfg = ModelConfig(provider="mock", model="x")
    mock_client = MockClient(cfg)
    msgs = engine.run(
        "support_reply",
        {"name": "Bob", "issue": "none"},
        None,
        model_cfg=cfg,
        client=mock_client,
        registry=reg,
    )
    out = [m async for m in msgs]
    assert out[-1].content == "ok"
