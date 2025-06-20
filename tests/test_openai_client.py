import json
import pytest
import httpx
import os

from prompti.model_client import ModelConfig, OpenAIClient
from prompti import Message
from openai_mock_server import OpenAIMockServer

@pytest.mark.asyncio
async def test_openai_client():
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        client = OpenAIClient(client=httpx.AsyncClient())
        client.api_url = url  # type: ignore

        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        messages = [Message(role="user", kind="text", content="hello")]
        out = [m async for m in client.run(messages, cfg)]
        assert out[0].content.startswith("Hello")

        messages = [Message(role="user", kind="text", content="calc 1+1")]
        out = [m async for m in client.run(messages, cfg)]
        assert "2" in out[0].content

        messages = [Message(role="user", kind="text", content="What time is it?")]
        cfg.parameters = {"tools": [{"type": "function", "function": {"name": "get_time", "description": "Get the current time", "parameters": {"type": "object", "properties": {}, "required": []}}}], "tool_choice": {"type": "function", "function": {"name": "get_time"}}}
        out = [m async for m in client.run(messages, cfg)]
        assert out[0].kind == "tool_use"
        data = json.loads(out[0].content)
        assert data["name"] == "get_time"

        cfg = ModelConfig(provider="openai", model="gpt-4o")
        messages = [Message(role="user", kind="text", content="Think step by step: what is 2+2?")]
        out = [m async for m in client.run(messages, cfg)]
        assert "4" in out[0].content
        assert "step" in out[0].content.lower()

        messages = [
            Message(
                role="user",
                kind="image_url",
                content="https://example.com/photo.png",
            )
        ]
        out = [m async for m in client.run(messages, cfg)]
        assert "sunset" in out[0].content.lower()

        await client.close()
