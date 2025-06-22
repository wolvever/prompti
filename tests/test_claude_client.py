import os

import httpx
import pytest
from mock_server import MockServer

from prompti import Message
from prompti.model_client import (
    ClaudeClient,
    ModelConfig,
    RunParams,
    ToolParams,
    ToolSpec,
)


@pytest.mark.asyncio
async def test_claude_client():
    with MockServer("tests/data/claude_record.jsonl") as url:
        os.environ["ANTHROPIC_API_KEY"] = "testkey"
        cfg = ModelConfig(provider="claude", model="claude-3-opus-20240229")
        client = ClaudeClient(cfg, client=httpx.AsyncClient(), api_url=url)
        params = RunParams(messages=[Message(role="user", kind="text", content="hi")])
        out = [m async for m in client.run(params)]
        assert out[0].content.startswith("Hello")

        messages = [Message(role="user", kind="text", content="What time is it?")]
        tool = ToolSpec(
            name="get_time",
            description="Get the current time",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        params = RunParams(messages=messages, tool_params=ToolParams(tools=[tool]))
        out = [m async for m in client.run(params)]
        assert out[0].kind == "thinking"
        assert out[1].kind == "tool_use"
        call_id = out[1].content["call_id"]

        messages = [
            Message(role="user", kind="text", content="What time is it?"),
            Message(
                role="assistant",
                kind="tool_use",
                content={"name": "get_time", "arguments": {}, "call_id": call_id},
            ),
            Message(
                role="user",
                kind="tool_result",
                content="2024-06-20T12:00:00Z",
            ),
        ]
        params = RunParams(messages=messages)
        out = [m async for m in client.run(params)]
        assert "12:00" in out[0].content

        messages = [Message(role="user", kind="text", content="Think step by step: what is 2+2?")]
        params = RunParams(messages=messages)
        out = [m async for m in client.run(params)]
        assert "4" in out[0].content

        messages = [
            Message(
                role="user",
                kind="image",
                content="https://example.com/photo.png",
            )
        ]
        params = RunParams(messages=messages)
        out = [m async for m in client.run(params)]
        assert "sunset" in out[0].content.lower()

        await client.close()


@pytest.mark.asyncio
async def test_claude_client_init_overrides():
    with MockServer("tests/data/claude_record.jsonl") as url:
        cfg = ModelConfig(provider="claude", model="claude-3-opus-20240229", api_key="override")
        client = ClaudeClient(cfg, client=httpx.AsyncClient(), api_url=url)
        params = RunParams(messages=[Message(role="user", kind="text", content="hi")])
        out = [m async for m in client.run(params)]
        assert out[0].content.startswith("Hello")
        await client.close()
