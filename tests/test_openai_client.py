import json
import os

import httpx
import pytest
from mock_server import MockServer

from prompti.model_client import Message, ModelConfig, OpenAIClient

GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@pytest.mark.asyncio
async def test_openai_init_overrides():
    with MockServer("tests/data/openai_record.jsonl") as url:
        client = OpenAIClient(
            client=httpx.AsyncClient(),
            api_url=url,
            api_key="testkey",
            api_key_var="IGNORED",
        )
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        msgs = [Message(role="user", kind="text", content="hello")]
        out = [m async for m in client.run(msgs, cfg)]
        assert out[0].content.startswith("Hello")
        await client.close()


@pytest.mark.asyncio
async def test_openai_basic_text():
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        msgs = [Message(role="user", kind="text", content="hello")]
        out = [m async for m in client.run(msgs, cfg)]
        assert out[0].content.startswith("Hello")
        msgs = [Message(role="user", kind="text", content="calc 1+1")]
        out = [m async for m in client.run(msgs, cfg)]
        assert "2" in out[0].content
        await client.close()


@pytest.mark.asyncio
async def test_openai_multi_parts():
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        msgs = [
            Message(role="user", kind="text", content="hello"),
            Message(role="user", kind="file", content={"name": "README.md", "mimeType": "text/markdown", "bytes": "AA=="}),
            Message(role="user", kind="data", content={"foo": "bar"}),
        ]
        out = [m async for m in client.run(msgs, cfg)]
        assert out[0].content.startswith("Hello")
        await client.close()


@pytest.mark.asyncio
async def test_openai_text_with_tool():
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        cfg.parameters = {"tools": [GET_TIME_TOOL], "tool_choice": {"type": "function", "function": {"name": "get_time"}}}
        msgs = [Message(role="user", kind="text", content="What time is it?")]
        out = [m async for m in client.run(msgs, cfg)]
        assert out[0].kind == "tool_use"
        data = json.loads(out[0].content)
        assert data["name"] == "get_time"
        await client.close()


@pytest.mark.asyncio
async def test_openai_file_with_tool():
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        cfg.parameters = {"tools": [GET_TIME_TOOL], "tool_choice": {"type": "function", "function": {"name": "get_time"}}}
        msgs = [
            Message(role="user", kind="file", content={"name": "input.txt", "mimeType": "text/plain", "bytes": "AA=="}),
            Message(role="user", kind="text", content="What time is it?"),
        ]
        out = [m async for m in client.run(msgs, cfg)]
        assert out[0].kind == "tool_use"
        await client.close()

