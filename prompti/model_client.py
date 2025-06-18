from __future__ import annotations

"""Model clients for various providers."""

import json
import os
from typing import Any, AsyncGenerator

import httpx
from opentelemetry import trace
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

from .message import Message


class ModelConfig(BaseModel):
    provider: str
    model: str
    parameters: dict[str, Any] = {}


class ModelClient:
    """Base class for model clients."""

    provider: str = "generic"

    _counter = Counter("llm_tokens_total", "Tokens in/out", labelnames=["direction"])
    _histogram = Histogram(
        "llm_request_latency_seconds", "LLM latency", labelnames=["provider"]
    )

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(http2=True)
        self._tracer = trace.get_tracer(__name__)

    @retry(wait=wait_exponential_jitter(), stop=stop_after_attempt(3))
    async def run(
        self, messages: list[Message], model_cfg: ModelConfig
    ) -> AsyncGenerator[Message, None]:
        """Execute the LLM call."""

        with self._tracer.start_as_current_span(
            "llm.call", attributes={"provider": self.provider, "model": model_cfg.model}
        ):
            with self._histogram.labels(self.provider).time():
                async for msg in self._run(messages, model_cfg):
                    yield msg

    async def _run(
        self, messages: list[Message], model_cfg: ModelConfig
    ) -> AsyncGenerator[Message, None]:
        raise NotImplementedError

    async def close(self) -> None:
        await self._client.aclose()


class _OpenAICore(ModelClient):
    """Shared logic for OpenAI-like providers."""

    api_url: str
    api_key_var: str

    async def _run(
        self, messages: list[Message], model_cfg: ModelConfig
    ) -> AsyncGenerator[Message, None]:
        oa_messages = []
        for m in messages:
            if m.kind == "text":
                oa_messages.append({"role": m.role, "content": m.content})
            elif m.kind == "tool_use":
                data = json.loads(m.content)
                oa_messages.append(
                    {
                        "role": m.role,
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": data["name"],
                                    "arguments": json.dumps(data.get("arguments", {})),
                                },
                            }
                        ],
                    }
                )

        payload = {"model": model_cfg.model, "messages": oa_messages}
        payload.update(model_cfg.parameters)
        api_key = os.environ.get(self.api_key_var, "")
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = await self._client.post(self.api_url, json=payload, headers=headers)
        if resp.status_code != 200:
            yield Message(role="assistant", kind="error", content=resp.text)
            return
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                func = call["function"]
                yield Message(
                    role="assistant",
                    kind="tool_use",
                    content=json.dumps(
                        {
                            "name": func["name"],
                            "arguments": json.loads(func.get("arguments", "{}")),
                        }
                    ),
                )
        elif msg.get("content"):
            yield Message(role="assistant", kind="text", content=msg["content"])


class OpenAIClient(_OpenAICore):
    provider = "openai"
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key_var = "OPENAI_API_KEY"


class OpenRouterClient(_OpenAICore):
    provider = "openrouter"
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    api_key_var = "OPENROUTER_API_KEY"


class LiteLLMClient(_OpenAICore):
    provider = "litellm"
    api_url = os.environ.get("LITELLM_ENDPOINT", "http://localhost:4000/v1/chat/completions")
    api_key_var = "LITELLM_API_KEY"


class ClaudeClient(ModelClient):
    provider = "claude"

    async def _run(
        self, messages: list[Message], model_cfg: ModelConfig
    ) -> AsyncGenerator[Message, None]:
        url = "https://api.anthropic.com/v1/messages"
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

        prompt = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.kind == "text"
        ]
        payload = {
            "model": model_cfg.model,
            "messages": prompt,
            "max_tokens": model_cfg.parameters.get("max_tokens", 16),
        }
        resp = await self._client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            yield Message(role="assistant", kind="error", content=resp.text)
            return
        data = resp.json()
        content = data.get("content", "") if isinstance(data, dict) else ""
        yield Message(role="assistant", kind="text", content=content)

