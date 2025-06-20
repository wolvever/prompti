import json
import pytest
import httpx
import os

from prompti.model_client import ModelConfig, OpenAIClient
from prompti import Message
from openai_mock_server import OpenAIMockServer

@pytest.mark.asyncio
async def test_openai_client_with_mock_server():
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

        await client.close()
