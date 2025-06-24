"""Tests for logging and replaying model client responses."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from prompti.model_client import Message, ModelConfig, LiteLLMClient, RunParams
from prompti.replay import ModelClientRecorder, ReplayEngine


@pytest.mark.asyncio
async def test_record_and_replay(tmp_path):
    # Create a mock LiteLLM client that returns a fixed response
    mock_client = AsyncMock(spec=LiteLLMClient)
    mock_client.provider = "litellm"
    mock_client._client = MagicMock()  # Add the _client attribute that ModelClientRecorder expects

    async def mock_run(params):
        yield Message(role="assistant", kind="text", content="pong")

    mock_client.run = mock_run
    mock_client._run = mock_run

    mock_client.cfg = ModelConfig(provider="litellm", model="gpt-4o")
    recorder = ModelClientRecorder(mock_client, "sess", output_dir=tmp_path)
    params = RunParams(messages=[Message(role="user", kind="text", content="ping")])
    result = [m async for m in recorder.run(params)]
    assert result[0].content == "pong"

    log_file = next(tmp_path.iterdir())
    rows = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert rows[0]["direction"] == "req"
    assert rows[1]["direction"] == "res"

    def factory(provider: str):
        mock_factory_client = AsyncMock(spec=LiteLLMClient)
        mock_factory_client.provider = provider
        mock_factory_client._client = MagicMock()

        async def mock_factory_run(params):
            yield Message(role="assistant", kind="text", content="pong")

        mock_factory_client.run = mock_factory_run
        mock_factory_client._run = mock_factory_run
        return mock_factory_client

    engine = ReplayEngine(factory)
    out = [m async for m in engine.replay(rows)]
    assert out[0].content == "pong"
