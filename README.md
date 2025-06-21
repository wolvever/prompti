# Prompti

Prompti is a provider‑agnostic asynchronous prompt engine built around the
Agent‑to‑Agent (A2A) message format. Prompts are stored as YAML templates with Jinja-formatted text and
may be loaded from disk, memory or a remote registry. All LLM calls are routed
through a pluggable `ModelClient` so that calling code never deals with
provider‑specific protocols.

## Get started

1. **Install dependencies** (requires Python 3.10+ and `uv`):

   ```bash
   uv pip install --system -e .[test]
   ```

2. **Run the tests** to verify your environment:

   ```bash
   pytest -q
   ```

3. **Execute an example** using the bundled prompt template:

   ```bash
   python examples/basic.py
   ```

   This will render `prompts/support_reply.yaml` and invoke the
   `OpenAIClient`, printing messages to the console.

Supported providers include **OpenAI**, **Claude (Anthropic)**, **OpenRouter**,
**LiteLLM**, and a **Rust**-based client for custom integrations.  Each provider
has its own `ModelClient` subclass (e.g. `OpenAIClient`).  Set the relevant API
key environment variables such as `OPENAI_API_KEY` before running examples.  The
Rust client is compiled as a Python extension and accessed through a thin
wrapper rather than spawning a subprocess.

### Built-in model clients

| Client            | Environment variables                        | Notes                              |
| ----------------- | -------------------------------------------- | ---------------------------------- |
| `OpenAIClient`    | `OPENAI_API_KEY`, optional `OPENAI_API_BASE` | Uses OpenAI chat completions.       |
| `ClaudeClient`    | `ANTHROPIC_API_KEY`                          | Supports thinking, tool use, image. Accepts custom `api_url`, `api_key_var`, `api_key` |
| `LiteLLMClient`   | `LITELLM_API_KEY`, `LITELLM_ENDPOINT`        | Routes through `litellm.acompletion` |
| `RustModelClient` | n/a (reads from `ModelConfig.api_key`)       | Uses the `model_client_rs` library directly via a Python wrapper. |

Prompti also supports SDK-level A/B experiments via the `ExperimentRegistry`
interface with built-in **Unleash** and **GrowthBook** adapters.

See `DESIGN.md` for a more detailed description of the architecture.

## Template examples

### Single message

```python
from prompti.template import PromptTemplate

template = PromptTemplate(
    id="hello",
    name="hello",
    version="1.0",
    yaml="""
messages:
  - role: user
    parts:
      - type: text
        text: "Hello {{ name }}!"
""",
)

print(template.format({"name": "World"})[0].content)
```

### Multi-message

```python
multi = PromptTemplate(
    id="analyze",
    name="analyze",
    version="1.0",
    required_variables=["file_path"],
    yaml="""
messages:
  - role: system
    parts:
      - type: text
        text: "Analyze file"
  - role: user
    parts:
      - type: file
        file: "{{ file_path }}"
""",
)

msgs = multi.format({"file_path": "/tmp/document.pdf"})
for m in msgs:
    print(m.role, m.kind, m.content)
```

### Jinja template

```python
tasks = [
    {"name": "Fix bug", "priority": 9},
    {"name": "Update docs", "priority": 3},
]

tmpl = PromptTemplate(
    id="report",
    name="report",
    version="1.0",
    yaml="""
messages:
  - role: user
    parts:
      - type: text
        text: |
          Task Report:
          {% for t in tasks %}
          - {{ t.name }} ({{ t.priority }})
          {% endfor %}
""",
)

print(tmpl.format({"tasks": tasks})[0].content)
```

### A/B test with GrowthBook

```python
import asyncio
from prompti.engine import PromptEngine, Setting
from prompti.experiment import GrowthBookRegistry
from prompti.model_client import ModelConfig, OpenAIClient

features = {"support_reply": {"id": "clarify", "variants": {"A": 1.0}}}
reg = GrowthBookRegistry(features)

async def main():
    engine = PromptEngine.from_setting(Setting(template_paths=["./prompts"]))
    async for msg in engine.run(
        "support_reply",
        {"name": "Ada", "issue": "login failed"},
        None,
        model_cfg=ModelConfig(provider="openai", model="gpt-4o"),
        client=OpenAIClient(),
        registry=reg,
    ):
        print(msg.content)

asyncio.run(main())
```
