"""Tests for the various model client implementations."""

import asyncio
import json

import pytest
import httpx
from httpx import Response, Request
import litellm

from prompti.model_client import (
    ModelConfig,
    Message,
    OpenAIClient,
    OpenRouterClient,
    LiteLLMClient,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,url",
    [
        ("openai", "https://api.openai.com/v1/chat/completions"),
        ("openrouter", "https://openrouter.ai/api/v1/chat/completions"),
    ],
)
async def test_openai_like_providers(provider: str, url: str):
    async def handler(request: Request):
        assert str(request.url) == url
        return Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    client_map = {
        "openai": OpenAIClient,
        "openrouter": OpenRouterClient,
    }
    mc = client_map[provider](client=client)
    cfg = ModelConfig(provider=provider, model="gpt-4o")
    messages = [Message(role="user", kind="text", content="hi")]
    result = [m async for m in mc.run(messages, cfg)]
    assert result[0].content == "ok"


@pytest.mark.asyncio
async def test_litellm_client(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured["kwargs"] = kwargs
        return litellm.ModelResponse(choices=[{"message": {"content": "ok"}}])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    mc = LiteLLMClient()
    cfg = ModelConfig(provider="litellm", model="gpt-3.5")
    messages = [Message(role="user", kind="text", content="hi")]
    result = [m async for m in mc.run(messages, cfg)]

    assert result[0].content == "ok"
    assert captured["kwargs"]["model"] == "gpt-3.5"
    assert captured["kwargs"]["messages"][0]["content"] == "hi"

