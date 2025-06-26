from pathlib import Path

import pytest

from prompti.engine import PromptEngine, Setting
from prompti.model_config_loader import ModelConfigLoader
from prompti.loader import (
    FileSystemLoader,
    MemoryLoader,
    TemplateLoader,
    TemplateNotFoundError,
)
from prompti.message import Message
from prompti.model_client import ModelClient, ModelConfig, RunParams
from prompti.template import PromptTemplate, Variant


class DummyClient(ModelClient):
    provider = "dummy"

    async def _run(self, params: RunParams):
        yield Message(role="assistant", kind="text", content="ok")


class DummyConfigLoader(ModelConfigLoader):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def load(self) -> ModelConfig:
        return self.cfg


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

    out = [m async for m in engine.run("x", {}, client=client, variant="base", stream=False)]
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
    from prompti.loader.base import VersionEntry

    class CountingLoader(TemplateLoader):
        def __init__(self):
            self.calls = 0

        async def list_versions(self, name: str) -> list[VersionEntry]:
            return [VersionEntry(id="1", tags=[])]

        async def get_template(self, name: str, version: str) -> PromptTemplate:
            self.calls += 1
            return PromptTemplate(
                id=name,
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
                yaml="",
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
    with pytest.raises(TemplateNotFoundError):
        await engine.load("nonexistent")


@pytest.mark.asyncio
async def test_global_model_config_used_when_variant_missing():
    yaml_text = """
name: x
version: '1'
variants:
  base:
    selector: []
    messages:
      - role: user
        parts: []
"""
    setting = Setting(
        memory_templates={"x": {"yaml": yaml_text}},
        global_config_loader=DummyConfigLoader(ModelConfig(provider="dummy", model="z")),
    )
    engine = PromptEngine.from_setting(setting)
    client = DummyClient(ModelConfig(provider="dummy", model="y"))
    out = [m async for m in engine.run("x", {}, client=client, variant="base", stream=False)]
    assert out[0].content == "ok"
    assert client.cfg == ModelConfig(provider="dummy", model="z")


@pytest.mark.asyncio
async def test_variant_overrides_global_model_config():
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
    setting = Setting(
        memory_templates={"x": {"yaml": yaml_text}},
        global_config_loader=DummyConfigLoader(ModelConfig(provider="dummy", model="z")),
    )
    engine = PromptEngine.from_setting(setting)
    client = DummyClient(ModelConfig(provider="dummy", model="y"))
    out = [m async for m in engine.run("x", {}, client=client, variant="base", stream=False)]
    assert client.cfg == ModelConfig(provider="dummy", model="x")


@pytest.mark.asyncio
async def test_error_when_no_model_config_available():
    yaml_text = """
name: x
version: '1'
variants:
  base:
    selector: []
    messages:
      - role: user
        parts: []
"""
    engine = PromptEngine([MemoryLoader({"x": {"yaml": yaml_text}})])
    client = DummyClient(ModelConfig(provider="dummy", model="y"))
    with pytest.raises(ValueError):
        [m async for m in engine.run("x", {}, client=client, variant="base", stream=False)]


@pytest.mark.asyncio
async def test_engine_format_openai():
    yaml_text = """
name: greet
version: '1'
variants:
  base:
    selector: []
    messages:
      - role: user
        parts:
          - type: text
            text: "Hello {{ name }}"
"""

    engine = PromptEngine([MemoryLoader({"greet": {"yaml": yaml_text}})])
    msgs = await engine.format("greet", {"name": "Ada"}, variant="base", format="openai")
    assert msgs == [{"role": "user", "content": "Hello Ada"}]
