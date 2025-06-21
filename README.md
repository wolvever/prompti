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

   This will render `prompts/support_reply.jinja` and invoke the
   `OpenAIClient`, printing messages to the console.

Supported providers include **OpenAI**, **Claude (Anthropic)**, **OpenRouter**,
**LiteLLM**, and a **Rust**-based client for custom integrations.  Each provider
has its own `ModelClient` subclass (e.g. `OpenAIClient`).  Set the relevant API
key environment variables such as `OPENAI_API_KEY` before running examples.

### Built-in model clients

| Client            | Environment variables                        | Notes                              |
| ----------------- | -------------------------------------------- | ---------------------------------- |
| `OpenAIClient`    | `OPENAI_API_KEY`, optional `OPENAI_API_BASE` | Uses OpenAI chat completions.       |
| `ClaudeClient`    | `ANTHROPIC_API_KEY`                          | Supports thinking, tool use, image |
| `LiteLLMClient`   | `LITELLM_API_KEY`, `LITELLM_ENDPOINT`        | Routes through `litellm.acompletion` |
| `RustModelClient` | n/a (reads from `ModelConfig.api_key`)       | Calls an external Rust binary.     |

Prompti also supports SDK-level A/B experiments via the `ExperimentRegistry`
interface with built-in **Unleash** and **GrowthBook** adapters.

See `DESIGN.md` for a more detailed description of the architecture.
