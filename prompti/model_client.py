from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator

from pydantic import BaseModel
from opentelemetry import trace
import httpx
from prometheus_client import Counter, Histogram
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

from .message import Message


class ModelConfig(BaseModel):
    provider: str
    model: str
    parameters: dict[str, Any] = {}


class ModelClient:
    """Adapter between A2A messages and provider wire protocols."""

    _counter = Counter(
        "llm_tokens_total", "Tokens in/out", labelnames=["direction"]
    )
    _histogram = Histogram(
        "llm_request_latency_seconds", "LLM latency", labelnames=["provider"]
    )

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._tracer = trace.get_tracer(__name__)
        self._client = client or httpx.AsyncClient(http2=True)

    @retry(wait=wait_exponential_jitter(), stop=stop_after_attempt(3))
    async def run(
        self,
        messages: list[Message],
        model_cfg: ModelConfig,
    ) -> AsyncGenerator[Message, None]:
        """Call the underlying LLM provider and yield A2A messages."""

        provider = model_cfg.provider
        model = model_cfg.model
        with self._tracer.start_as_current_span(
            "llm.call", attributes={"provider": provider, "model": model}
        ):
            with self._histogram.labels(provider).time():
                if provider in {"openai", "openrouter", "litellm"}:
                    async for m in self._run_openai_like(messages, model_cfg, provider):
                        yield m
                elif provider in {"anthropic", "claude"}:
                    async for m in self._run_claude(messages, model_cfg):
                        yield m
                else:
                    # Unknown provider, echo last message
                    if messages:
                        yield Message(role="assistant", kind="text", content="ack")

    async def close(self) -> None:
        await self._client.aclose()

    async def _run_openai_like(
        self,
        messages: list[Message],
        model_cfg: ModelConfig,
        provider: str,
    ) -> AsyncGenerator[Message, None]:
        url_map = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "litellm": os.environ.get("LITELLM_ENDPOINT", "http://localhost:4000/v1/chat/completions"),
        }
        url = url_map[provider]
        headers = {
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "litellm": "LITELLM_API_KEY",
        }
        api_key_var = headers[provider]
        api_key = os.environ.get(api_key_var, "")
        request_headers = {"Authorization": f"Bearer {api_key}"}

        oa_messages = []
        for m in messages:
            if m.kind == "text":
                oa_messages.append({"role": m.role, "content": m.content})
            elif m.kind == "tool_use":
                data = json.loads(m.content)
                oa_messages.append({
                    "role": m.role,
                    "content": None,
                    "tool_calls": [{"type": "function", "function": {"name": data["name"], "arguments": json.dumps(data.get("arguments", {}))}}],
                })

        payload = {"model": model_cfg.model, "messages": oa_messages}
        payload.update(model_cfg.parameters)
        resp = await self._client.post(url, json=payload, headers=request_headers)
        if resp.status_code != 200:
            yield Message(role="assistant", kind="error", content=str(resp.text))
            return
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                func = call["function"]
                yield Message(role="assistant", kind="tool_use", content=json.dumps({"name": func["name"], "arguments": json.loads(func.get("arguments", "{}"))}))
        elif msg.get("content"):
            yield Message(role="assistant", kind="text", content=msg["content"])

    async def _run_claude(
        self,
        messages: list[Message],
        model_cfg: ModelConfig,
    ) -> AsyncGenerator[Message, None]:
        url = "https://api.anthropic.com/v1/messages"
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        prompt = []
        for m in messages:
            if m.kind == "text":
                prompt.append({"role": m.role, "content": m.content})
        payload = {"model": model_cfg.model, "messages": prompt, "max_tokens": model_cfg.parameters.get("max_tokens", 16)}
        resp = await self._client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            yield Message(role="assistant", kind="error", content=str(resp.text))
            return
        data = resp.json()
        content = data.get("content", "") if isinstance(data, dict) else ""
        yield Message(role="assistant", kind="text", content=content)
