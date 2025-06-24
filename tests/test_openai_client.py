import json
import os

import httpx
import pytest
from mock_server import MockServer

from prompti.model_client import (
    Message,
    ModelConfig,
    OpenAIClient,
    RunParams,
    ToolParams,
    ToolSpec,
)

GET_TIME_TOOL = ToolSpec(
    name="get_time",
    description="Get the current time",
    parameters={"type": "object", "properties": {}, "required": []},
)


@pytest.mark.asyncio
async def test_openai_init_overrides():
    with MockServer("tests/data/openai_record.jsonl") as url:
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo", api_key="testkey", api_url=url)
        client = OpenAIClient(cfg, client=httpx.AsyncClient())
        params = RunParams(messages=[Message(role="user", kind="text", content="hello")], stream=False)
        out = [m async for m in client.run(params)]
        assert out[0].content.startswith("Hello")
        await client.close()


@pytest.mark.asyncio
async def test_openai_basic_text():
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo", api_url=url)
        client = OpenAIClient(cfg, client=httpx.AsyncClient())
        params = RunParams(messages=[Message(role="user", kind="text", content="hello")], stream=False)
        out = [m async for m in client.run(params)]
        assert out[0].content.startswith("Hello")
        params = RunParams(messages=[Message(role="user", kind="text", content="calc 1+1")], stream=False)
        out = [m async for m in client.run(params)]
        assert "2" in out[0].content
        await client.close()


@pytest.mark.asyncio
async def test_openai_multi_parts():
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo", api_url=url)
        client = OpenAIClient(cfg, client=httpx.AsyncClient())
        msgs = [
            Message(role="user", kind="text", content="hello"),
            Message(
                role="user",
                kind="file",
                content={"name": "README.md", "mimeType": "text/markdown", "bytes": "AA=="},
            ),
            Message(role="user", kind="data", content={"foo": "bar"}),
        ]
        params = RunParams(messages=msgs, stream=False)
        out = [m async for m in client.run(params)]
        assert out[0].content.startswith("Hello")
        await client.close()


@pytest.mark.asyncio
async def test_openai_text_with_tool():
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo", api_url=url)
        client = OpenAIClient(cfg, client=httpx.AsyncClient())
        params = RunParams(
            messages=[Message(role="user", kind="text", content="What time is it?")],
            tool_params=ToolParams(
                tools=[GET_TIME_TOOL],
                choice={"type": "function", "function": {"name": "get_time"}},
            ),
            stream=False,
        )
        out = [m async for m in client.run(params)]
        assert out[0].kind == "tool_use"
        data = json.loads(out[0].content)
        assert data["name"] == "get_time"
        await client.close()


@pytest.mark.asyncio
async def test_openai_file_with_tool():
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo", api_url=url)
        client = OpenAIClient(cfg, client=httpx.AsyncClient())
        tool_params = ToolParams(
            tools=[GET_TIME_TOOL],
            choice={"type": "function", "function": {"name": "get_time"}},
        )
        msgs = [
            Message(role="user", kind="file", content={"name": "input.txt", "mimeType": "text/plain", "bytes": "AA=="}),
            Message(role="user", kind="text", content="What time is it?"),
        ]
        params = RunParams(messages=msgs, tool_params=tool_params, stream=False)
        out = [m async for m in client.run(params)]
        assert out[0].kind == "tool_use"
        await client.close()
