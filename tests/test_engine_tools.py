import pytest
import httpx

from prompti.engine import PromptEngine, Setting
from prompti.model_client import ModelClient, ModelConfig, Message

class Dummy(ModelClient):
    provider = "dummy"

    def __init__(self):
        super().__init__(client=httpx.AsyncClient(http2=False))

    async def _run(self, messages, model_cfg):
        if not any(m.kind == "tool_result" for m in messages):
            yield Message(role="assistant", kind="tool_use", content={"name": "ping", "arguments": {}})
        else:
            yield Message(role="assistant", kind="text", content="done")


@pytest.mark.asyncio
async def test_engine_with_tools():
    engine = PromptEngine.from_setting(Setting(template_paths=["./prompts"]))
    dummy = Dummy()
    cfg = ModelConfig(provider="dummy", model="x")

    async def ping():
        return "pong"

    out = [m async for m in engine.run(
        "support_reply",
        {"name": "Bob", "issue": "none"},
        None,
        model_cfg=cfg,
        client=dummy,
        tools={"ping": ping},
    )]
    assert out[-1].content == "done"
    assert any(m.kind == "tool_result" for m in out)
