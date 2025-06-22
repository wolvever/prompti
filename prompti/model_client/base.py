"""Base classes and data models for model clients."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

import httpx
from opentelemetry import trace
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from ..message import Message


class ModelConfig(BaseModel):
    """Static connection details for a model provider."""

    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None


class ToolSpec(BaseModel):
    """Specification for a single tool."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolChoice(str, Enum):
    """Allowed tool invocation policies."""

    AUTO = "auto"
    BLOCK = "none"
    REQUIRED = "required"
    FORCE = "force"


class ToolParams(BaseModel):
    """Tool catalogue and invocation configuration."""

    tools: list[ToolSpec] | list[dict]
    choice: ToolChoice | dict[str, Any] = ToolChoice.AUTO
    force_tool: str | None = None
    parallel_allowed: bool = True
    max_calls: int | None = None


class RunParams(BaseModel):
    """Per-call parameters for :class:`ModelClient.run`."""

    messages: list[Message]
    tool_params: ToolParams | list[ToolSpec] | list[dict] | None = None

    # sampling & length
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None

    # control & reproducibility
    stream: bool = True
    n: int | None = None
    seed: int | None = None
    logit_bias: dict[int, float] | None = None
    response_format: str | None = None

    # misc
    user_id: str | None = None
    extra_params: dict[str, Any] = {}


class ModelClient:
    """Base class for model clients."""

    provider: str = "generic"

    _counter = Counter("llm_tokens_total", "Tokens in/out", labelnames=["direction"])
    _histogram = Histogram("llm_request_latency_seconds", "LLM latency", labelnames=["provider"])

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None, **_: Any) -> None:
        """Create the client with static :class:`ModelConfig` and optional HTTP client."""

        self.cfg = cfg
        self._client = client or httpx.AsyncClient(http2=True)
        self._tracer = trace.get_tracer(__name__)

    @retry(wait=wait_exponential_jitter(), stop=stop_after_attempt(3))
    async def run(self, params: RunParams) -> AsyncGenerator[Message, None]:
        """Execute the LLM call with dynamic ``params``."""

        with self._tracer.start_as_current_span(
            "llm.call",
            attributes={"provider": self.cfg.provider, "model": self.cfg.model},
        ), self._histogram.labels(self.cfg.provider).time():
            async for msg in self._run(params):
                yield msg

    async def _run(self, params: RunParams) -> AsyncGenerator[Message, None]:
        raise NotImplementedError
        yield  # pragma: no cover - satisfies generator type

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
