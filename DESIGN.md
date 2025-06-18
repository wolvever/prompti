# PromptI Design Overview

This document summarizes the architecture of PromptI.  Each prompt is stored as a
versioned Jinja template and rendered into the A2A message format before it is
sent to a language model via a pluggable client.

## 0 · High-level summary

The library contains three main layers:

| Layer              | Purpose                                                            | Contract                                      |
| ------------------ | ------------------------------------------------------------------ | --------------------------------------------- |
| **PromptTemplate** | Render a Jinja revision to `Message[]` or stream results from an LLM | Works entirely in A2A messages                |
| **PromptEngine**   | Locate templates from files, HTTP, or memory and cache them         | Provides a simple `run(name, ...)` API        |
| **ModelClient**    | Convert between A2A messages and provider protocols                 | Returns an async generator of A2A messages    |

All APIs are async and independent from specific LLM providers.

## 1 · A2A message schema

Every exchange uses the Google A2A format with the fields:

- `role` (`assistant`, `user`, `tool`, ...)
- `kind` (`text`, `tool_use`, `tool_result`, `error`, `done`)
- `content` (string or JSON object)

## 2 · Core classes

### PromptTemplate

- Parses Jinja templates in a sandboxed environment
- Provides `format()` for local rendering and `run()` to invoke a `ModelClient`

### PromptEngine

- Holds multiple loaders (`FileSystemLoader`, `HTTPLoader`, `MemoryLoader`)
- Resolves mutable labels like `prod` to immutable revisions and caches them

### ModelClient

- Adapts A2A messages to provider wire formats (OpenAI, Anthropic, ...)
- Integrates tracing via OpenTelemetry and metrics via Prometheus

## 3 · Template discovery

Templates can come from the local filesystem, an HTTP registry or in-memory
mappings. Labels such as `prod` map to specific template revisions.

## 4 · Error handling and observability

- Missing templates fall back to cached versions if available
- LLM calls are retried with exponential backoff (Tenacity)
- Metrics capture latency and token counts

See `README.md` for usage instructions.
