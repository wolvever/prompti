from __future__ import annotations

"""Base classes for model clients."""

from typing import Any, AsyncGenerator

import httpx
from opentelemetry import trace
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

from ..message import Message


class ModelConfig(BaseModel):
    """Configuration for a model invocation."""

    provider: str
    model: str
    parameters: dict[str, Any] = {}
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = True


class ModelClient:
    """Base class for model clients."""

    provider: str = "generic"

    _counter = Counter("llm_tokens_total", "Tokens in/out", labelnames=["direction"])
    _histogram = Histogram(
        "llm_request_latency_seconds", "LLM latency", labelnames=["provider"]
    )

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Create the client with an optional :class:`httpx.AsyncClient`."""
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
        yield  # This line will never execute but makes this an async generator

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
