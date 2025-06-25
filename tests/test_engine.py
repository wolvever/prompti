import pytest

from pathlib import Path

from prompti.engine import PromptEngine
from prompti.loader import FileSystemLoader, MemoryLoader, TemplateLoader
from prompti.template import PromptTemplate, Variant
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
    selector: []
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



@pytest.mark.asyncio
async def test_load_returns_template():
    engine = PromptEngine([FileSystemLoader(Path("./prompts"))])
    tmpl = await engine.load("summary")
    assert isinstance(tmpl, PromptTemplate)
    assert tmpl.name == "summary"
    assert tmpl.version == "1.0"


@pytest.mark.asyncio
async def test_load_caches_result():
    class CountingLoader(TemplateLoader):
        def __init__(self):
            self.calls = 0

        async def load(self, name: str, tags: str | None):
            self.calls += 1
            return "1", PromptTemplate(
                name=name,
                description="",
                version="1",
                variants={
                    "base": Variant(
                        selector=[],
                        model_config=ModelConfig(provider="dummy", model="x"),
                        messages=[],
                    )
                },
            )

    loader = CountingLoader()
    engine = PromptEngine([loader])
    tmpl1 = await engine.load("demo")
    tmpl2 = await engine.load("demo")
    assert tmpl1 is tmpl2
    assert loader.calls == 1


@pytest.mark.asyncio
async def test_load_missing_raises():
    engine = PromptEngine([FileSystemLoader(Path("./prompts"))])
    with pytest.raises(FileNotFoundError):
        await engine.load("nonexistent")

