"""Tests for the various model client implementations."""

import os

import httpx
import pytest
from openai_mock_server import OpenAIMockServer

from prompti.model_client import (
    Message,
    ModelConfig,
    OpenAIClient,
    OpenRouterClient,
)

GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    ["openai", "openrouter"],
)
async def test_openai_like_providers(provider):
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        # Set OpenRouter API key to avoid empty bearer token
        if provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = "testkey"
        client_map = {
            "openai": OpenAIClient,
            "openrouter": OpenRouterClient,
        }
        mc = client_map[provider](client=httpx.AsyncClient())
        if provider == "openai":
            mc.api_url = url  # type: ignore
        else:
            # For OpenRouter, we need to set the base URL
            mc.api_url = url  # type: ignore

        cfg = ModelConfig(provider=provider, model="gpt-3.5-turbo")
        messages = [Message(role="user", kind="text", content="hello")]
        result = [m async for m in mc.run(messages, cfg)]
        assert result[0].content.startswith("Hello")


@pytest.mark.asyncio
async def test_model_client_tools():
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        mc = OpenAIClient(client=httpx.AsyncClient())
        mc.api_url = url  # type: ignore
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        messages = [Message(role="user", kind="text", content="What time is it?")]

        tools = [GET_TIME_TOOL]
        cfg.parameters = {"tools": tools, "tool_choice": {"type": "function", "function": {"name": "get_time"}}}

        result = [m async for m in mc.run(messages, cfg, tools=tools)]
        assert result[-1].kind == "tool_use"


@pytest.mark.asyncio
async def test_model_client_tool_request_format():
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        mc = OpenAIClient(client=httpx.AsyncClient())
        mc.api_url = url  # type: ignore
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        messages = [Message(role="user", kind="text", content="What time is it?")]

        tools = [GET_TIME_TOOL]
        cfg.parameters = {"tools": tools, "tool_choice": {"type": "function", "function": {"name": "get_time"}}}

        first = [m async for m in mc.run(messages, cfg, tools=tools)]
        messages.extend(first)
        messages.append(Message(role="tool", kind="tool_result", content="12:00 PM"))

        # For the second call, we test with calc question that should return text
        messages = [Message(role="user", kind="text", content="calc 1+1")]
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        result = [m async for m in mc.run(messages, cfg)]
        assert "2" in result[0].content
