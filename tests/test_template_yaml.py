"""Tests for YAML-based template loading."""

import pytest
from prompti.engine import PromptEngine, Setting

@pytest.mark.asyncio
async def test_format_yaml():
    settings = Setting(template_paths=["./prompts"])
    engine = PromptEngine.from_setting(settings)
    messages = await engine.format("multi/summary", {"summary": "Hello"})
    assert messages[-1].content == "Hello"
