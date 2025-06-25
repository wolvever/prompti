import pytest
from pathlib import Path

from prompti.engine import PromptEngine
from prompti.loader import FileSystemLoader, TemplateLoader
from prompti.template import PromptTemplate, Variant
from prompti.model_client import ModelConfig


@pytest.mark.asyncio
async def test_load_returns_template():
    engine = PromptEngine([FileSystemLoader(Path("./prompts"))])
    tmpl = await engine.load("summary")
    assert tmpl.name == "summary"
    assert tmpl.version == "1.0"


@pytest.mark.asyncio
async def test_load_caches_result():
    class CountingLoader(TemplateLoader):
        def __init__(self):
            self.calls = 0

        async def __call__(self, name: str, label: str | None):
            self.calls += 1
            return "1", PromptTemplate(
                name=name,
                description="",
                version="1",
                variants={
                    "base": Variant(
                        contains=[],
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
