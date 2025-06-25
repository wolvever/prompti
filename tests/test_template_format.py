import pytest
from pathlib import Path

from prompti.loader import FileSystemLoader
from prompti.template import PromptTemplate, Variant
from prompti.model_client import ModelConfig


@pytest.mark.asyncio
async def test_load_from_file_has_expected_fields():
    loader = FileSystemLoader(Path("./prompts"))
    version, tmpl = await loader.load("summary", None)
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
                selector=[],
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
                selector=[],
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
                selector=[],
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

