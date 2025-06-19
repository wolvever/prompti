import json
import asyncio

import pytest
import httpx
from httpx import Response, Request

from prompti import (
    Message,
    ModelConfig,
    OpenAIClient,
    ModelClientRecorder,
    ReplayEngine,
)


@pytest.mark.asyncio
async def test_record_and_replay(tmp_path):
    async def handler(request: Request):
        return Response(200, json={"choices": [{"message": {"content": "pong"}}]})

    transport = httpx.MockTransport(handler)
    base = OpenAIClient(client=httpx.AsyncClient(transport=transport))
    recorder = ModelClientRecorder(base, "sess", output_dir=tmp_path)
    cfg = ModelConfig(provider="openai", model="gpt-4o")
    msgs = [Message(role="user", kind="text", content="ping")]
    result = [m async for m in recorder.run(msgs, cfg)]
    assert result[0].content == "pong"

    log_file = next(tmp_path.iterdir())
    rows = [json.loads(l) for l in log_file.read_text().splitlines()]
    assert rows[0]["direction"] == "req"
    assert rows[1]["direction"] == "res"

    def factory(provider: str):
        return OpenAIClient(client=httpx.AsyncClient(transport=transport))

    engine = ReplayEngine(factory)
    out = [m async for m in engine.replay(rows)]
    assert out[0].content == "pong"
