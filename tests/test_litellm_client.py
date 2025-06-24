import httpx
from unittest.mock import patch
import json
import pytest

from mock_server import MockServer
from prompti.model_client import LiteLLMClient, ModelConfig, RunParams, ToolParams, ToolSpec, Message

GET_TIME_TOOL = ToolSpec(
    name="get_time",
    description="Get the current time",
    parameters={"type": "object", "properties": {}, "required": []},
)

@pytest.mark.asyncio
async def test_litellm_openai_basic_text():
    async def fake_acompletion(**kw):
        url = kw.pop("base_url")
        kw.pop("api_key", None)
        async with httpx.AsyncClient() as c:
            resp = await c.post(url, json=kw)
            return resp.json()

    with MockServer("tests/data/openai_record.jsonl") as url, patch("litellm.acompletion", fake_acompletion):
        cfg = ModelConfig(provider="litellm", model="gpt-3.5-turbo", api_key="k", api_url=url)
        client = LiteLLMClient(cfg)
        params = RunParams(messages=[Message(role="user", kind="text", content="hello")], stream=False)
        out = [m async for m in client.run(params)]
        assert out[0].content.startswith("Hello")
        await client.close()

@pytest.mark.asyncio
async def test_litellm_openai_tool_call():
    async def fake_acompletion(**kw):
        url = kw.pop("base_url")
        kw.pop("api_key", None)
        async with httpx.AsyncClient() as c:
            resp = await c.post(url, json=kw)
            return resp.json()

    with MockServer("tests/data/openai_record.jsonl") as url, patch("litellm.acompletion", fake_acompletion):
        cfg = ModelConfig(provider="litellm", model="gpt-3.5-turbo", api_key="k", api_url=url)
        client = LiteLLMClient(cfg)
        params = RunParams(
            messages=[Message(role="user", kind="text", content="What time is it?")],
            tool_params=ToolParams(tools=[GET_TIME_TOOL], choice={"type": "function", "function": {"name": "get_time"}}),
            stream=False,
        )
        out = [m async for m in client.run(params)]
        assert out[0].kind == "tool_use"
        data = json.loads(out[0].content)
        assert data["name"] == "get_time"
        await client.close()

@pytest.mark.asyncio
async def test_litellm_claude_basic_text():
    async def fake_acompletion(**kw):
        url = kw.pop("base_url")
        kw.pop("api_key", None)
        async with httpx.AsyncClient() as c:
            resp = await c.post(url, json=kw)
            return resp.json()

    with MockServer("tests/data/claude_record.jsonl") as url, patch("litellm.acompletion", fake_acompletion):
        cfg = ModelConfig(provider="litellm", model="claude-3-opus-20240229", api_key="k", api_url=url)
        client = LiteLLMClient(cfg)
        params = RunParams(messages=[Message(role="user", kind="text", content="hi")], stream=False)
        out = [m async for m in client.run(params)]
        assert out[0].content.startswith("Hello")
        await client.close()

@pytest.mark.asyncio
async def test_litellm_claude_tool_call():
    async def fake_acompletion(**kw):
        url = kw.pop("base_url")
        kw.pop("api_key", None)
        async with httpx.AsyncClient() as c:
            resp = await c.post(url, json=kw)
            return resp.json()

    with MockServer("tests/data/claude_record.jsonl") as url, patch("litellm.acompletion", fake_acompletion):
        cfg = ModelConfig(provider="litellm", model="claude-3-opus-20240229", api_key="k", api_url=url)
        client = LiteLLMClient(cfg)
        params = RunParams(
            messages=[Message(role="user", kind="text", content="What time is it?")],
            tool_params=ToolParams(tools=[GET_TIME_TOOL]),
            stream=False,
        )
        out = [m async for m in client.run(params)]
        assert out[0].kind in {"thinking", "tool_use"}
        await client.close()
