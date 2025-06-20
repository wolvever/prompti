import json
import pytest
import httpx
import os

from prompti.model_client import ModelConfig, ClaudeClient
from prompti import Message
from openai_mock_server import OpenAIMockServer

@pytest.mark.asyncio
async def test_anthropic_client():
    with OpenAIMockServer("tests/data/anthropic_record.jsonl") as url:
        os.environ["ANTHROPIC_API_KEY"] = "testkey"
        client = ClaudeClient(client=httpx.AsyncClient())
        client.api_url = url  # type: ignore

        cfg = ModelConfig(provider="claude", model="claude-3-7-sonnet-20250219")
        messages = [Message(role="user", kind="text", content="你好")]
        out = [m async for m in client.run(messages, cfg)]
        print(out)
        assert out[0].content.startswith("你好！有什么我可以帮助你的吗？")

        messages = [Message(role="user", kind="text", content="What is 15 + 27?")]
        out = [m async for m in client.run(messages, cfg)]
        assert "42" in out[0].content

        messages = [Message(role="user", kind="text", content="Write a short poem about spring")]
        out = [m async for m in client.run(messages, cfg)]
        assert "spring" in out[0].content.lower()
        assert "awakening" in out[0].content.lower()

        messages = [Message(role="user", kind="text", content="How do I read a file in Python?")]
        out = [m async for m in client.run(messages, cfg)]
        assert "open" in out[0].content.lower()
        assert "with" in out[0].content.lower()

        messages = [Message(role="user", kind="text", content="Translate 'Hello World' to Spanish")]
        out = [m async for m in client.run(messages, cfg)]
        assert "hola" in out[0].content.lower() or "mundo" in out[0].content.lower()

        messages = [Message(role="user", kind="text", content="Tell me a short story about a cat")]
        out = [m async for m in client.run(messages, cfg)]
        assert "cat" in out[0].content.lower() or "visitor" in out[0].content.lower()

        messages = [Message(role="user", kind="text", content="How to maintain a healthy sleep schedule?")]
        out = [m async for m in client.run(messages, cfg)]
        assert "sleep" in out[0].content.lower()
        assert "schedule" in out[0].content.lower() or "routine" in out[0].content.lower()

        messages = [Message(role="user", kind="text", content="Recommend a good place for spring travel")]
        out = [m async for m in client.run(messages, cfg)]
        assert "spring" in out[0].content.lower()
        assert "travel" in out[0].content.lower() or "kyoto" in out[0].content.lower()

        messages = [Message(role="user", kind="text", content="How to improve learning efficiency?")]
        out = [m async for m in client.run(messages, cfg)]
        assert "learning" in out[0].content.lower()
        assert "efficiency" in out[0].content.lower()

        messages = [Message(role="user", kind="text", content="How to manage work stress effectively?")]
        out = [m async for m in client.run(messages, cfg)]
        assert "stress" in out[0].content.lower()
        assert "work" in out[0].content.lower()

        await client.close() 