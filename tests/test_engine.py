import pytest

from prompti.engine import PromptEngine
from prompti.loader import MemoryLoader
from prompti.model_client import ModelClient, ModelConfig, RunParams
from prompti.message import Message


class DummyClient(ModelClient):
    provider = "dummy"

    async def _run(self, params: RunParams):
        yield Message(role="assistant", kind="text", content="ok")


@pytest.mark.asyncio
async def test_engine_run_uses_model_cfg():
    yaml_text = """
name: x
version: '1'
variants:
  base:
    contains: []
    model_config:
      provider: dummy
      model: x
    messages:
      - role: user
        parts: []
"""
    engine = PromptEngine([MemoryLoader({"x": {"yaml": yaml_text}})])
    client = DummyClient(ModelConfig(provider="dummy", model="y"))

    out = [
        m async for m in engine.run("x", {}, client=client, variant="base", stream=False)
    ]
    assert out[0].content == "ok"
    assert client.cfg == ModelConfig(provider="dummy", model="x")
