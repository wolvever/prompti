"""Tests for the Rust model client."""

import os
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mock_server import MockServer

from prompti.message import Message
from prompti.model_client import OpenAIClient, RunParams, RustModelClient
from prompti.model_client.base import ModelConfig


async def async_iterator(items):
    """Helper to create an async iterator from a list of items."""
    for item in items:
        yield item


@pytest.fixture
def rust_client():
    """Create a Rust model client for testing using the native wrapper."""
    with patch("prompti.model_client.rust.rs_client.ModelClient") as mock_cls:
        mock_cls.return_value = AsyncMock()
        return RustModelClient(ModelConfig(provider="openai", model="gpt-3.5-turbo"))


def test_rust_client_initialization():
    """Test that the Rust client can be initialized."""
    with patch("prompti.model_client.rust.rs_client.ModelClient") as mock_cls:
        mock_cls.return_value = AsyncMock()
        client = RustModelClient(ModelConfig(provider="openai", model="gpt-3.5-turbo"))
        assert client.provider == "rust"
        assert hasattr(client, "_rs_client")


@pytest.mark.asyncio
async def test_rust_client_with_openai_fallback():
    """Test the Rust client with OpenAI mock server as fallback."""
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"

        # Test the actual OpenAI client that the Rust client might fall back to
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo", api_url=url)
        openai_client = OpenAIClient(cfg, client=httpx.AsyncClient())

        messages = [Message(role="user", content="hello", kind="text")]
        openai_client.cfg = ModelConfig(api_key="test-key", provider="openai", model="gpt-3.5-turbo")

        results = []
        async for msg in openai_client._run(RunParams(messages=messages, stream=False)):
            results.append(msg)

        assert len(results) >= 1
        assert results[0].content.startswith("Hello")


@pytest.mark.asyncio
async def test_rust_client_run():
    """Test the Rust client run method with mocked Rust wrapper."""

    class DummyClient:
        async def chat_stream(self, _):
            for chunk in [{"content": "Hello"}, {"content": " world"}]:
                yield chunk

    mock_instance = DummyClient()
    with patch("prompti.model_client.rust.rs_client.ModelClient", return_value=mock_instance):
        client = RustModelClient(ModelConfig(provider="openai", model="gpt-3.5-turbo"))

        messages = [Message(role="user", content="Hello", kind="text")]
        client.cfg = ModelConfig(api_key="test-key", provider="openai", model="gpt-3.5-turbo")

        results = []
        async for msg in client._run(RunParams(messages=messages)):
            results.append(msg)

        assert len(results) == 2
        assert results[0].content == "Hello"
        assert results[1].content == " world"


@pytest.mark.asyncio
async def test_rust_client_run_error():
    """Test error handling in the Rust client."""

    class DummyClient:
        async def chat_stream(self, _):
            raise RuntimeError("Rust client failed")
            yield  # pragma: no cover

    mock_instance = DummyClient()
    with patch("prompti.model_client.rust.rs_client.ModelClient", return_value=mock_instance):
        client = RustModelClient(ModelConfig(provider="openai", model="gpt-3.5-turbo"))
        messages = [Message(role="user", content="Hello", kind="text")]
        client.cfg = ModelConfig(api_key="test-key", provider="openai", model="gpt-3.5-turbo")
        with pytest.raises(RuntimeError, match="Rust client failed"):
            async for _ in client._run(RunParams(messages=messages)):
                pass


@pytest.mark.asyncio
async def test_rust_client_tools_support():
    """Ensure tool definitions are forwarded to the Rust client."""

    captured: dict[str, Any] = {}

    class DummyClient:
        async def chat_stream(self, request):
            captured.update(request)
            yield {"content": "ok"}

    tool = {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    with patch("prompti.model_client.rust.rs_client.ModelClient", return_value=DummyClient()):
        client = RustModelClient(ModelConfig(provider="openai", model="gpt-3.5-turbo"))

        messages = [Message(role="user", content="What time?", kind="text")]
        client.cfg = ModelConfig(api_key="k", provider="openai", model="gpt-3.5-turbo")
        params = RunParams(messages=messages, tool_params=[tool], extra_params={"tool_choice": "auto"})
        results = [m async for m in client._run(params)]

        assert captured.get("tools") == [tool]
        assert captured.get("tool_choice") == "auto"
        assert results[0].content == "ok"
