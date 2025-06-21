import json
import os

import httpx
import pytest
from openai_mock_server import OpenAIMockServer

from prompti.model_client import Message, ModelConfig, OpenAIClient

GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current time",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

# Example FilePart used in docs for inline binary upload
IMAGE_FILE_PART = {
    "name": "input_image.png",
    "mimeType": "image/png",
    "bytes": "iVBORw0KGgoAAA...",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["default", "api_key", "key_var"])
async def test_openai_client_init_variants(variant):
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        if variant == "api_key":
            client = OpenAIClient(client=httpx.AsyncClient(), api_url=url, api_key="override")
        elif variant == "key_var":
            os.environ["OTHER_KEY"] = "testkey"
            client = OpenAIClient(client=httpx.AsyncClient(), api_url=url, api_key_var="OTHER_KEY")
        else:
            os.environ["OPENAI_API_KEY"] = "testkey"
            client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)

        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        messages = [Message(role="user", kind="text", content="hello")]
        out = [m async for m in client.run(messages, cfg)]
        assert out[0].content.startswith("Hello")
        await client.close()


@pytest.mark.asyncio
async def test_openai_basic_text():
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")

        messages = [Message(role="user", kind="text", content="calc 1+1")]
        out = [m async for m in client.run(messages, cfg)]
        assert "2" in out[0].content

        await client.close()


@pytest.mark.asyncio
async def test_openai_multi_message():
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        messages = [
            Message(role="user", kind="text", content="hello"),
            Message(role="user", kind="file", content=IMAGE_FILE_PART),
            Message(role="user", kind="data", content={"k": "v"}),
        ]
        out = [m async for m in client.run(messages, cfg)]
        assert out[0].content.startswith("Hello")
        await client.close()


@pytest.mark.asyncio
async def test_openai_text_with_tools():
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        cfg.parameters = {
            "tools": [GET_TIME_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_time"}},
        }
        messages = [Message(role="user", kind="text", content="What time is it?")]
        out = [m async for m in client.run(messages, cfg)]
        assert out[0].kind == "tool_use"
        data = json.loads(out[0].content)
        assert data["name"] == "get_time"
        await client.close()


@pytest.mark.asyncio
async def test_openai_file_with_tool():
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")
        cfg.parameters = {
            "tools": [GET_TIME_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_time"}},
        }
        messages = [
            Message(role="user", kind="file", content=IMAGE_FILE_PART),
            Message(role="user", kind="text", content="What time is it?"),
        ]
        out = [m async for m in client.run(messages, cfg)]
        assert out[0].kind == "tool_use"
        await client.close()
