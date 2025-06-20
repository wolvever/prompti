"""Tests for the various model client implementations."""

import asyncio
import json

import pytest
import httpx
from httpx import Response, Request

from prompti.model_client import (
    ModelConfig,
    Message,
    OpenAIClient,
    OpenRouterClient,
    LiteLLMClient,
)


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
    client_map = {
        "openai": OpenAIClient,
        "openrouter": OpenRouterClient,
        "litellm": LiteLLMClient,
    }
    mc = client_map[provider](client=client)
    cfg = ModelConfig(provider=provider, model="gpt-4o")
    messages = [Message(role="user", kind="text", content="hi")]
    result = [m async for m in mc.run(messages, cfg)]
    assert result[0].content == "ok"


@pytest.mark.asyncio
async def test_model_client_tools():
    async def handler(request: Request):
        return Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": "ping", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    mc = OpenAIClient(client=httpx.AsyncClient(transport=transport))
    cfg = ModelConfig(provider="openai", model="gpt-4o")
    messages = [Message(role="user", kind="text", content="hi")]

    async def ping():
        return "pong"

    result = [m async for m in mc.run(messages, cfg, tools={"ping": ping})]
    assert result[-1].kind == "tool_result"
    assert result[-1].content == "pong"

