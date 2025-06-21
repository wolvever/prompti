import os
from pathlib import Path

import httpx
import pytest
from openai_mock_server import OpenAIMockServer

from prompti.engine import PromptEngine, Setting
from prompti.model_client import ModelConfig, OpenAIClient


@pytest.mark.asyncio
async def test_engine_with_tools():
    with OpenAIMockServer("tests/data/openai_record.jsonl") as url:
        os.environ["OPENAI_API_KEY"] = "testkey"
        engine = PromptEngine.from_setting(Setting(template_paths=[Path("./prompts")]))
        client = OpenAIClient(client=httpx.AsyncClient(), api_url=url)
        cfg = ModelConfig(provider="openai", model="gpt-3.5-turbo")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
        out = [
            m
            async for m in engine.run(
                "support_reply",
                {"name": "Bob", "issue": "none"},
                None,
                model_cfg=cfg,
                client=client,
                tools=tools,
            )
        ]
        # The mock server should return a response from the recorded data
        assert len(out) > 0
        assert out[-1].content  # Should have content
