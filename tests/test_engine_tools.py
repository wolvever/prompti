import pytest
import httpx

from prompti.engine import PromptEngine, Setting
from prompti.model_client import ModelClient, ModelConfig, Message

class Dummy(ModelClient):
    provider = "dummy"

    def __init__(self):
        super().__init__(client=httpx.AsyncClient(http2=False))

    async def _run(self, messages, model_cfg, tools=None):
        self.received_tools = tools
        yield Message(role="assistant", kind="text", content="ok")


@pytest.mark.asyncio
async def test_engine_with_tools():
    engine = PromptEngine.from_setting(Setting(template_paths=["./prompts"]))
    dummy = Dummy()
    cfg = ModelConfig(provider="dummy", model="x")

    tools = [{"type": "function", "function": {"name": "ping"}}]
    out = [
        m
        async for m in engine.run(
            "support_reply",
            {"name": "Bob", "issue": "none"},
            None,
            model_cfg=cfg,
            client=dummy,
            tools=tools,
        )
    ]
    assert dummy.received_tools == tools
    assert out[-1].content == "ok"
