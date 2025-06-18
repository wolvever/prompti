import asyncio
import json

import pytest
import httpx
from httpx import Response, Request

from prompti.model_client import ModelClient, ModelConfig, Message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,url", [
        ("openai", "https://api.openai.com/v1/chat/completions"),
        ("openrouter", "https://openrouter.ai/api/v1/chat/completions"),
        ("litellm", "http://localhost:4000/v1/chat/completions"),
    ]
)
async def test_openai_like_providers(provider, url):
    async def handler(request: Request):
        assert str(request.url) == url
        return Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    mc = ModelClient(client=client)
    cfg = ModelConfig(provider=provider, model="gpt-4o")
    messages = [Message(role="user", kind="text", content="hi")]
    result = [m async for m in mc.run(messages, cfg)]
    assert result[0].content == "ok"

