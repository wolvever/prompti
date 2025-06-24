from pathlib import Path

import pytest

from prompti.loader import FileSystemLoader
from prompti.template import PromptTemplate
from prompti.model_client import ModelClient, ModelConfig, RunParams
from prompti.message import Message


@pytest.mark.asyncio
async def test_load_from_file_has_expected_fields():
    loader = FileSystemLoader(Path("./prompts"))
    version, tmpl = await loader("summary", None)
    assert version == "1.0"
    assert tmpl.id == "summary"
    assert tmpl.name == "summary"
    assert tmpl.version == "1.0"
    assert tmpl.labels == ["summarization", "text-processing"]
    assert tmpl.required_variables == ["summary"]


def test_single_message_format():
    template = PromptTemplate(
        id="test",
        name="test",
        version="1.0",
        required_variables=["name"],
        yaml="""
messages:
  - role: user
    parts:
      - type: text
        text: "Hello {{ name }}!"
""",
    )
    messages = template.format({"name": "World"})
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].kind == "text"
    assert messages[0].content == "Hello World!"


def test_multi_message_different_kinds(tmp_path: Path):
    file_path = tmp_path / "document.pdf"
    template = PromptTemplate(
        id="test",
        name="test",
        version="1.0",
        required_variables=["file_path"],
        yaml=f"""
messages:
  - role: system
    parts:
      - type: text
        text: "Analyze file"
  - role: user
    parts:
      - type: file
        file: "{file_path}"
""",
    )
    messages = template.format({"file_path": str(file_path)})
    assert len(messages) == 2
    assert messages[0].content == "Analyze file"
    assert messages[1].kind == "file"
    assert messages[1].content == str(file_path)


def test_complex_jinja_multi_message():
    template = PromptTemplate(
        id="test",
        name="test",
        version="1.0",
        required_variables=["tasks", "priority_threshold"],
        yaml="""
messages:
  - role: user
    parts:
      - type: text
        text: |
          Task Report:
          {% for task in tasks -%}
          {% if task.priority >= priority_threshold -%}
          🔥 HIGH: {{ task.name }} (Priority: {{ task.priority }})
          {% else -%}
          📝 NORMAL: {{ task.name }} (Priority: {{ task.priority }})
          {% endif -%}
          {% endfor -%}

          {% set high_priority_count = tasks | selectattr('priority', '>=', priority_threshold) | list | length -%}
          Total high-priority tasks: {{ high_priority_count }}
""",
    )
    tasks = [
        {"name": "Fix bug", "priority": 9},
        {"name": "Update docs", "priority": 3},
        {"name": "Security patch", "priority": 10},
        {"name": "Refactor code", "priority": 5},
    ]
    messages = template.format({"tasks": tasks, "priority_threshold": 8})
    content = messages[0].content
    assert "🔥 HIGH: Fix bug (Priority: 9)" in content
    assert "📝 NORMAL: Update docs (Priority: 3)" in content
    assert "🔥 HIGH: Security patch (Priority: 10)" in content
    assert "📝 NORMAL: Refactor code (Priority: 5)" in content
    assert "Total high-priority tasks: 2" in content


@pytest.mark.asyncio
async def test_template_run_uses_model_cfg():
    class DummyClient(ModelClient):
        provider = "dummy"

        async def _run(self, params: RunParams):
            yield Message(role="assistant", kind="text", content="ok")

    tmpl_cfg = ModelConfig(provider="dummy", model="x")
    client = DummyClient(ModelConfig(provider="dummy", model="y"))
    template = PromptTemplate(id="x", name="x", version="1", model_cfg=tmpl_cfg)

    out = [
        m
        async for m in template.run(
            {},
            None,
            model_cfg=None,
            client=client,
            stream=False,
        )
    ]

    assert out[0].content == "ok"
    assert client.cfg == tmpl_cfg
