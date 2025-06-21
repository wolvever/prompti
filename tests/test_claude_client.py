import os

import httpx
import pytest
from claude_mock_server import ClaudeMockServer

from prompti import Message
from prompti.model_client import ClaudeClient, ModelConfig


@pytest.mark.asyncio
async def test_claude_client():
    with ClaudeMockServer("tests/data/claude_record.jsonl") as url:
        os.environ["ANTHROPIC_API_KEY"] = "testkey"
        client = ClaudeClient(client=httpx.AsyncClient())
        client.api_url = url  # type: ignore

        cfg = ModelConfig(provider="claude", model="claude-3-opus-20240229")
        messages = [Message(role="user", kind="text", content="hi")]
        out = [m async for m in client.run(messages, cfg)]
        assert out[0].content.startswith("Hello")

        messages = [Message(role="user", kind="text", content="What time is it?")]
        cfg.parameters = {
            "tools": [
                {
                    "name": "get_time",
                    "description": "Get the current time",
                    "input_schema": {"type": "object", "properties": {}, "required": []},
                }
            ]
        }
        out = [m async for m in client.run(messages, cfg)]
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
        cfg.parameters = {}
        out = [m async for m in client.run(messages, cfg)]
        assert "12:00" in out[0].content

        messages = [Message(role="user", kind="text", content="Think step by step: what is 2+2?")]
        out = [m async for m in client.run(messages, cfg)]
        assert "4" in out[0].content

        messages = [
            Message(
                role="user",
                kind="image",
                content="https://example.com/photo.png",
            )
        ]
        out = [m async for m in client.run(messages, cfg)]
        assert "sunset" in out[0].content.lower()

        await client.close()
