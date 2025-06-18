import asyncio
import pytest
from prompti.engine import PromptEngine, Setting


@pytest.mark.asyncio
async def test_format_basic():
    settings = Setting(template_paths=["./prompts"]) 
    engine = PromptEngine.from_setting(settings)
    messages = await engine.format("support_reply", {"name": "Ada", "issue": "login"})
    assert messages[0].content.startswith("Hello Ada")


