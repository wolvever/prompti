# 🚀 Prompti

**Provider-agnostic asynchronous prompt engine with A2A message format and Jinja2 templating**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Prompti is a provider‑agnostic asynchronous prompt engine built around the
Agent‑to‑Agent (A2A) message format. Prompts are stored as YAML templates with Jinja-formatted text and
may be loaded from disk, memory or a remote registry. All LLM calls are routed
through a pluggable `ModelClient` so that calling code never deals with
provider‑specific protocols.

## 🌟 Key Features

- **🔄 Provider Agnostic**: Unified interface for LiteLLM and custom providers
- **📝 Jinja2 Templates**: First-class support for dynamic prompt templating with loops, filters, and safe sandboxing
- **⚡ Async First**: Full asynchronous workflow with cloud-native observability (OpenTelemetry, Prometheus)
- **🎯 A2A Message Format**: Standardized Agent-to-Agent communication with support for text, files, data, and tool interactions
- **📂 Multi-Source Templates**: Load prompts from local files, remote registries, or in-memory storage

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
   model via LiteLLM using `create_client`, printing messages to the console.

The included client implementation uses **LiteLLM**. Set the
appropriate environment variables such as `LITELLM_API_KEY` before running the
examples.

4. **Send an ad-hoc query via the CLI**:

   ```bash
   python examples/chat_cli.py -q "Hello" \
       --api-key YOUR_KEY --model gpt-4o \
       -f README.md --time-tool --reasoning
   ```

   Use `-f` to attach files, `--time-tool` to enable a sample `get_time` tool,
   and `--reasoning` to request thinking messages when supported.
   Add `--no-stream` to disable streaming.
   Metrics are available at `http://localhost:8000/metrics`.
   Logs and OpenTelemetry spans (including the full request and each response chunk) are printed to the console.


## 🛠️ Supported Providers

| Provider | Environment Variables | Notes |
|----------|----------------------|-------|
| **LiteLLM** | `LITELLM_API_KEY`, `LITELLM_ENDPOINT` | Universal LLM gateway |


See `DESIGN.md` for a more detailed description of the architecture.

## 🔌 Template Loaders

Prompti can read prompt templates from multiple back-ends. In addition to local
files, you can load templates from PromptLayer, Langfuse, Pezzo, Agenta, GitHub
repositories, or a local Git repo. Each loader exposes the same async call
contract:

```python
version, tmpl = await loader.load("my_prompt", tags="prod")
```

### 🎯 Version Selection

Template loaders support flexible version selection syntax for precise version matching:

| Format | Description | Example |
|--------|-------------|---------|
| `name@1.x` | Major version wildcard | Selects latest 1.x.x version |
| `name@1.x#prod` | Major version + tag | Latest 1.x.x with 'prod' tag |
| `name@1.x#prod+exp_a` | Major version + multiple tags | Version with both 'prod' and 'exp_a' tags |
| `name@1.2.x` | Minor version wildcard | Latest 1.2.x version |
| `name@1.2.x#beta` | Minor version + tag | 1.2.x version with 'beta' tag |
| `name@>=1.2.0 <1.5.0` | Version range | Latest version in specified range |

**Usage Examples:**

```python
# Load production version of latest 1.x
template = await loader.load("user-greeting", "1.x#prod")

# Load experimental version with multiple tags
template = await loader.load("user-greeting", "1.2.x#prod+exp_a")

# Load specific version range
template = await loader.load("user-greeting", ">=1.2.0 <1.5.0")

# Load exact version
template = await loader.load("user-greeting", "1.3.2")
```

To wire them up:

```python
from prompti.loader import (
    PromptLayerLoader,
    LangfuseLoader,
    PezzoLoader,
    AgentaLoader,
    GitHubRepoLoader,
    LocalGitRepoLoader,
)

loaders = {
    "promptlayer": PromptLayerLoader(api_key="pl-key"),
    "langfuse": LangfuseLoader(public_key="pk", secret_key="sk"),
    "pezzo": PezzoLoader(project="my-project"),
    "agenta": AgentaLoader(app_slug="my-app"),
    "github": GitHubRepoLoader(repo="org/repo"),
    "git": LocalGitRepoLoader(Path("/opt/prompts")),
}
```

## 💬 A2A Message Format

Messages consist of an array of parts. The three common part shapes are:

```json
{ "kind": "text", "text": "tell me a joke" }
{ "kind": "file", "file": { "name": "README.md", "mimeType": "text/markdown", "bytes": "IyBTYW1wbGUgTWFya2Rvd24gZmlsZQoK…" } }
{ "kind": "data", "data": { "action": "create-issue", "fields": { "project": "MLInfra", "severity": "high", "title": "GPU node failure" } } }
```

## 📝 Template Examples

### Single message

```python
from prompti.template import PromptTemplate, Variant
from prompti.model_client import ModelConfig

template = PromptTemplate(
    name="hello",
    description="demo",
    version="1.0",
    variants={
        "default": Variant(
            selector=[],
            model_config=ModelConfig(provider="litellm", model="gpt-4o"),
            messages=[{"role": "user", "parts": [{"type": "text", "text": "Hello {{ name }}!"}]}],
        )
    },
)

# ``format`` defaults to ``"openai"``. Use ``format="a2a"`` for raw A2A messages
# or ``format="claude"``/``format="litellm"`` for provider‑specific structures.
msgs, _ = template.format({"name": "World"}, format="a2a")
print(msgs[0].content)
```

### Multi-message

```python
multi = PromptTemplate(
    name="analyze",
    description="",
    version="1.0",
    variants={
        "default": Variant(
            selector=[],
            model_config=ModelConfig(provider="litellm", model="gpt-3.5-turbo"),
            messages=[
                {"role": "system", "parts": [{"type": "text", "text": "Analyze file"}]},
                {"role": "user", "parts": [{"type": "file", "file": "{{ file_path }}"}]},
            ],
        )
    },
)

msgs, _ = multi.format({"file_path": "/tmp/document.pdf"}, format="a2a")
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
    name="report",
    description="",
    version="1.0",
    variants={
        "default": Variant(
            selector=[],
            model_config=ModelConfig(provider="litellm", model="gpt-3.5-turbo"),
            messages=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "|\n  Task Report:\n  {% for t in tasks %}\n  - {{ t.name }} ({{ t.priority }})\n  {% endfor %}",
                        }
                    ],
                }
            ],
        )
    },
)

msgs, _ = tmpl.format({"tasks": tasks}, format="a2a")
print(msgs[0].content)
```


## 🧪 Use Cases

- **Multi-provider LLM Applications**: Build applications that can switch between different LLM providers seamlessly
- **Dynamic Prompt Management**: Use Jinja2 templates for complex, data-driven prompt generation
- **Agent Workflows**: Implement complex agent-to-agent communication patterns
- **Production LLM Systems**: Deploy robust, observable LLM applications with proper error handling and monitoring

## 📖 Documentation

- See `DESIGN.md` for detailed architecture and design decisions
- Check `examples/` directory for usage patterns
- Review `prompts/` for template examples

## 📋 Requirements

- Python 3.10+
- uv (for dependency management)

---

*Built for production LLM applications that need flexibility, performance, and reliability.*
```
