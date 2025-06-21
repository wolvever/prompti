"""Tests for the Rust model client."""

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai_mock_server import OpenAIMockServer

from prompti.message import Message
from prompti.model_client import OpenAIClient, RustModelClient
from prompti.model_client.base import ModelConfig


async def async_iterator(items):
    """Helper to create an async iterator from a list of items."""
    for item in items:
        yield item


@pytest.fixture
def rust_client():
    """Create a Rust model client for testing."""
    with patch.object(RustModelClient, '_find_rust_binary', return_value='/fake/path/to/rust-binary'):
        return RustModelClient()


def test_rust_client_initialization():
    """Test that the Rust client can be initialized."""
    with patch.object(RustModelClient, '_find_rust_binary', return_value='/fake/path/to/rust-binary'):
        client = RustModelClient()
        assert client.provider == "rust"
        assert hasattr(client, 'rust_binary_path')


def test_find_rust_binary_not_found():
    """Test that an error is raised when the Rust binary is not found."""
    with patch('pathlib.Path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            RustModelClient()._find_rust_binary()


@pytest.mark.asyncio
async def test_rust_client_with_openai_fallback():
    """Test the Rust client with OpenAI mock server as fallback."""
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"

        # Test the actual OpenAI client that the Rust client might fall back to
        openai_client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)

        messages = [Message(role="user", content="hello", kind="text")]
        model_cfg = ModelConfig(
            api_key="test-key",
            provider="openai",
            model="gpt-3.5-turbo"
        )

        results = []
        async for msg in openai_client._run(messages, model_cfg):
            results.append(msg)

        assert len(results) >= 1
        assert results[0].content.startswith("Hello")


@pytest.mark.asyncio
async def test_rust_client_run():
    """Test the Rust client run method with mock subprocess."""
    with patch.object(RustModelClient, '_find_rust_binary', return_value='/fake/path/to/rust-binary'):
        client = RustModelClient()

        # Mock the subprocess execution
        mock_process = AsyncMock()
        mock_process.stdout = async_iterator([
            b'{"content": "Hello", "role": "assistant"}\n',
            b'{"content": " world", "role": "assistant"}\n'
        ])
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.returncode = 0  # Ensure the return code is 0
        mock_process.stderr = AsyncMock()
        mock_process.stderr.read = AsyncMock(return_value=b'')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            messages = [Message(role="user", content="Hello", kind="text")]
            model_cfg = ModelConfig(
                api_key="test-key",
                provider="openai",
                model="gpt-3.5-turbo"
            )

            results = []
            async for msg in client._run(messages, model_cfg):
                results.append(msg)

            assert len(results) == 2
            assert results[0].content == "Hello"
            assert results[1].content == " world"


@pytest.mark.asyncio
async def test_rust_client_run_error():
    """Test error handling in the Rust client."""
    with patch.object(RustModelClient, '_find_rust_binary', return_value='/fake/path/to/rust-binary'):
        client = RustModelClient()

        # Mock the subprocess execution with an error
        mock_process = AsyncMock()
        mock_process.stdout = async_iterator([])
        mock_process.wait = AsyncMock(return_value=1)
        mock_process.returncode = 1  # Ensure the return code is 1 for error case
        mock_process.stderr = AsyncMock()
        mock_process.stderr.read = AsyncMock(return_value=b'Error: API key invalid')

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            messages = [Message(role="user", content="Hello", kind="text")]
            model_cfg = ModelConfig(
                api_key="test-key",
                provider="openai",
                model="gpt-3.5-turbo"
            )

            with pytest.raises(RuntimeError, match="Rust client failed"):
                async for _ in client._run(messages, model_cfg):
                    pass
