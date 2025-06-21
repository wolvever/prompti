"""Tests for the Rust model client."""

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mock_server import MockServer

from prompti.message import Message
from prompti.model_client import OpenAIClient, RustModelClient
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
        return RustModelClient()


def test_rust_client_initialization():
    """Test that the Rust client can be initialized."""
    with patch("prompti.model_client.rust.rs_client.ModelClient") as mock_cls:
        mock_cls.return_value = AsyncMock()
        client = RustModelClient()
        assert client.provider == "rust"
        assert hasattr(client, "_rs_client")




@pytest.mark.asyncio
async def test_rust_client_with_openai_fallback():
    """Test the Rust client with OpenAI mock server as fallback."""
    with MockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"

        # Test the actual OpenAI client that the Rust client might fall back to
        openai_client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)

        messages = [Message(role="user", content="hello", kind="text")]
        model_cfg = ModelConfig(api_key="test-key", provider="openai", model="gpt-3.5-turbo")

        results = []
        async for msg in openai_client._run(messages, model_cfg):
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
        client = RustModelClient()

        messages = [Message(role="user", content="Hello", kind="text")]
        model_cfg = ModelConfig(api_key="test-key", provider="openai", model="gpt-3.5-turbo")

        results = []
        async for msg in client._run(messages, model_cfg):
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
        client = RustModelClient()
        messages = [Message(role="user", content="Hello", kind="text")]
        model_cfg = ModelConfig(api_key="test-key", provider="openai", model="gpt-3.5-turbo")
        with pytest.raises(RuntimeError, match="Rust client failed"):
            async for _ in client._run(messages, model_cfg):
                pass
