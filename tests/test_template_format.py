import pytest
from pathlib import Path

from prompti.loader import FileSystemLoader
from prompti.template import PromptTemplate, Variant
from prompti.model_client import ModelClient, ModelConfig, RunParams
from prompti.message import Message


@pytest.mark.asyncio
async def test_load_from_file_has_expected_fields():
    loader = FileSystemLoader(Path("./prompts"))
    version, tmpl = await loader("summary", None)
    assert version == "1.0"
    assert tmpl.name == "summary"
    assert tmpl.version == "1.0"
    assert tmpl.tags == ["summarization", "text-processing"]
    assert "default" in tmpl.variants


def test_single_message_format():
    template = PromptTemplate(
        name="hello",
        description="",
        version="1.0",
        variants={
            "base": Variant(
                contains=[],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {
                        "role": "user",
                        "parts": [{"type": "text", "text": "Hello {{ name }}!"}],
                    }
                ],
            )
        },
    )
    msgs, _ = template.format({"name": "World"}, variant="base")
    assert len(msgs) == 1
    assert msgs[0].content == "Hello World!"


def test_multi_message_different_kinds(tmp_path: Path):
    file_path = tmp_path / "document.pdf"
    template = PromptTemplate(
        name="test",
        description="",
        version="1.0",
        variants={
            "base": Variant(
                contains=[],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {"role": "system", "parts": [{"type": "text", "text": "Analyze file"}]},
                    {"role": "user", "parts": [{"type": "file", "file": str(file_path)}]},
                ],
            )
        },
    )
    msgs, _ = template.format({"file_path": str(file_path)}, variant="base")
    assert msgs[0].content == "Analyze file"
    assert msgs[1].kind == "file"


def test_complex_jinja_multi_message():
    template = PromptTemplate(
        name="task",
        description="",
        version="1.0",
        variants={
            "base": Variant(
                contains=[],
                model_config=ModelConfig(provider="dummy", model="x"),
                messages=[
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": (
                                    "Task Report:\n"
                                    "{% for t in tasks %}"\
                                    "{{ '- ' + t.name }} ({{ t.priority }})\n"\
                                    "{% endfor %}"
                                ),
                            }
                        ],
                    }
                ],
            )
        },
    )
    tasks = [{"name": "Fix", "priority": 1}, {"name": "Doc", "priority": 2}]
    msgs, _ = template.format({"tasks": tasks}, variant="base")
    content = msgs[0].content
    assert "Fix" in content and "Doc" in content


@pytest.mark.asyncio
async def test_template_run_uses_model_cfg():
    class DummyClient(ModelClient):
        provider = "dummy"

        async def _run(self, params: RunParams):
            yield Message(role="assistant", kind="text", content="ok")

    tmpl_cfg = ModelConfig(provider="dummy", model="x")
    client = DummyClient(ModelConfig(provider="dummy", model="y"))
    template = PromptTemplate(
        name="x",
        description="",
        version="1",
        variants={
            "base": Variant(
                contains=[],
                model_config=tmpl_cfg,
                messages=[{"role": "user", "parts": []}],
            )
        },
    )

    out = [
        m
        async for m in template.run({}, client=client, variant="base", stream=False)
    ]
    assert out[0].content == "ok"
    assert client.cfg == tmpl_cfg

