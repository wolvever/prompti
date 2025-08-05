"""Base classes and data models for model clients."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from typing import Any, Union

import httpx
from opentelemetry import trace
from opentelemetry.baggage import set_baggage
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from collections.abc import Generator

from ..message import Message, ModelResponse, StreamingModelResponse
from typing import Optional


class ModelConfig(BaseModel):
    """Static connection and default generation parameters."""

    provider: Optional[str] | None = None
    model: Optional[str] | None = None
    api_key: Optional[str] | None = None
    api_url: Optional[str] | None = None

    # generation defaults (may be overridden per call)
    temperature: Optional[float] = None
    top_p: Optional[float] | None = None
    max_tokens: Optional[int] | None = None


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

    tools: list[ToolSpec]
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
    request_id: str | None = None
    session_id: str | None = None
    span_id : str | None = None
    parent_span_id : str | None = None
    source: str | None = None
    extra_params: dict[str, Any] = {}

    
    # trace data capture - used to pass data between engine and model client
    trace_context: dict[str, Any] = {}


class ModelClient:
    """Base class for model clients."""

    provider: str = "generic"

    _counter = Counter("llm_tokens_total", "Tokens in/out", labelnames=["direction"])
    _histogram = Histogram("llm_request_latency_seconds", "LLM latency", labelnames=["provider"])
    _inflight = Gauge(
        "llm_inflight_requests",
        "Inflight LLM requests",
        labelnames=["provider", "is_error"],
    )
    _request_counter = Counter(
        "llm_requests_total",
        "LLM request results",
        labelnames=["provider", "result", "is_error"],
    )
    _first_token = Histogram(
        "llm_first_token_latency_seconds",
        "Time to first token",
        labelnames=["provider", "model"],
        buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10),
    )
    _token_gap = Histogram(
        "llm_stream_intertoken_gap_seconds",
        "Gap between streamed tokens",
        labelnames=["provider", "model"],
    )
    _prompt_tokens = Counter(
        "llm_prompt_tokens_total",
        "Prompt tokens sent to the provider",
        labelnames=["provider", "model"],
    )
    _completion_tokens = Counter(
        "llm_completion_tokens_total",
        "Completion tokens received from the provider",
        labelnames=["provider", "model"],
    )

    def __init__(
        self, cfg: ModelConfig, client: httpx.AsyncClient | None = None, is_debug: bool = False, **_: Any
    ) -> None:
        """Create the client with static :class:`ModelConfig` and optional HTTP client."""
        self.cfg = cfg
        self._client = client or httpx.AsyncClient(http2=True, timeout=httpx.Timeout(600))
        self._tracer = trace.get_tracer(__name__)
        self._logger = logging.getLogger("model_client")
        self._is_debug = is_debug

        if self._is_debug:
            self._client.event_hooks.setdefault("request", []).append(self._log_request)
            self._client.event_hooks.setdefault("response", []).append(self._log_response)
        else:
            self._client.event_hooks.setdefault("request", []).append(self._log_request_jsonl)
            self._client.event_hooks.setdefault("response", []).append(self._log_response_jsonl)

    async def _log_request(self, request: httpx.Request) -> None:
        """Log outgoing HTTP request details as a cURL command."""
        import shlex

        command = f"curl -X {request.method} '{request.url}'"
        for k, v in request.headers.items():
            command += f" \\\n  -H '{k}: {v}'"

        body_bytes = request.content
        if body_bytes:
            body_str = ""
            try:
                body_str = body_bytes.decode()
            except UnicodeDecodeError:
                body_str = "<...binary data...>"

            command += f" \\\n  -d {shlex.quote(body_str)}"

        self._logger.info(f"http request as curl:\n{command}")

    async def _log_response(self, response: httpx.Response) -> None:
        """Log incoming HTTP response details."""
        # Log response in structured format similar to cURL output
        log_lines = [
            f"http response: {response.status_code} {response.reason_phrase}",
            f"  url: {response.url}"
        ]

        # Add headers
        for k, v in response.headers.items():
            log_lines.append(f"  header '{k}: {v}'")

        # Only read content for non-streaming responses
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            content = await response.aread()
            text = content.decode(response.encoding or "utf-8", "replace")
            if text:
                log_lines.append(f"  body: {text}")
            response._content = content

        self._logger.info("\n".join(log_lines))

    def _sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Remove sensitive information from headers."""
        sensitive_keys = {"authorization", "x-api-key", "api-key", "bearer"}
        sanitized = {}
        for k, v in headers.items():
            if k.lower() in sensitive_keys:
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = v
        return sanitized

    def _sanitize_body(self, body: str) -> dict[str, Any] | str:
        """Remove sensitive information from request/response body."""
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                # Remove or mask sensitive fields
                sanitized = data.copy()
                # Common sensitive fields to redact
                sensitive_fields = {"api_key", "authorization", "token", "secret", "password"}
                for field in sensitive_fields:
                    if field in sanitized:
                        sanitized[field] = "[REDACTED]"
                return sanitized
            return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If not JSON or can't decode, return truncated string
            return body[:1000] + "..." if len(body) > 1000 else body

    async def _log_request_jsonl(self, request: httpx.Request) -> None:
        """Log outgoing HTTP request in JSONL format for production use."""
        body_str = ""
        if request.content:
            try:
                body_str = request.content.decode()
            except UnicodeDecodeError:
                body_str = "<binary data>"

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "http_request",
            "method": request.method,
            "url": str(request.url),
            "headers": self._sanitize_headers(dict(request.headers)),
            "body": self._sanitize_body(body_str) if body_str else None,
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        self._logger.info(json.dumps(log_data, separators=(",", ":")))

    async def _log_response_jsonl(self, response: httpx.Response) -> None:
        """Log incoming HTTP response in JSONL format for production use."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "http_response",
            "status_code": response.status_code,
            "reason_phrase": response.reason_phrase,
            "url": str(response.url),
            "headers": self._sanitize_headers(dict(response.headers)),
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        # Only read content for non-streaming responses
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            content = await response.aread()
            text = content.decode(response.encoding or "utf-8", "replace")
            log_data["body"] = self._sanitize_body(text) if text else None
            response._content = content
        else:
            log_data["body"] = "<streaming response>"

        self._logger.info(json.dumps(log_data, separators=(",", ":")))

    @retry(wait=wait_exponential_jitter(), stop=stop_after_attempt(3))
    async def arun(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Execute the LLM call with dynamic ``params``.
        
        Returns:
            AsyncGenerator yielding ModelResponse for non-streaming calls or 
            StreamingResponse for streaming calls.
        """
        is_error = False
        self._inflight.labels(self.cfg.provider, "false").inc()
        result = "success"
        start = perf_counter()
        first = True
        last = start
        attrs = {
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        # 初始化或更新遥测上下文，包含通用请求数据
        if "llm_request_body" not in params.trace_context:
            params.trace_context["llm_request_body"] = {
                "model": self.cfg.model,
                "provider": self.cfg.provider,
                "messages": [msg.model_dump() for msg in params.messages],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        # 初始化响应体容器
        params.trace_context["llm_response_body"] = {}
        params.trace_context["responses"] = []

        if params.request_id:
            attrs["http.request_id"] = params.request_id
        if params.session_id:
            attrs["user.session_id"] = params.session_id
        if params.user_id:
            attrs["user.id"] = params.user_id

        for key, val in (
            ("request_id", params.request_id),
            ("session_id", params.session_id),
            ("user_id", params.user_id),
        ):
            if val:
                set_baggage(key, val)

        with (
            self._tracer.start_as_current_span("llm.call", attributes=attrs),
            self._histogram.labels(self.cfg.provider).time(),
        ):
            params.trace_context["perf_metrics"] = {}
            try:
                async for response in self._run(params):
                    now = perf_counter()
                    if first:
                        self._first_token.labels(self.cfg.provider, self.cfg.model).observe(now - start)
                        params.trace_context["perf_metrics"]["first_package_latency"] = now - start
                        params.trace_context["perf_metrics"]["total_latency"] = now - start
                        first = False
                    else:
                        self._token_gap.labels(self.cfg.provider, self.cfg.model).observe(now - last)
                        params.trace_context["perf_metrics"]["total_latency"] = now - start
                    last = now
                    yield response

            except Exception as e:
                is_error = True
                result = "error"
                raise
            finally:
                self._inflight.labels(self.cfg.provider, "false").dec()
                self._request_counter.labels(self.cfg.provider, result, str(is_error).lower()).inc()

    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Internal method to be implemented by subclasses.
        
        Args:
            params: Runtime parameters for the model call
            
        Returns:
            AsyncGenerator yielding ModelResponse for non-streaming calls or 
            StreamingResponse for streaming calls.
        """
        raise NotImplementedError
        yield  # pragma: no cover - satisfies generator type

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
    
    # Backward compatibility alias
    async def close(self) -> None:
        """Deprecated: use aclose() instead."""
        import warnings
        warnings.warn("ModelClient.close() is deprecated, use aclose() instead", DeprecationWarning, stacklevel=2)
        await self.aclose()
    
    # Backward compatibility alias
    async def run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Deprecated: use arun() instead."""
        import warnings
        warnings.warn("ModelClient.run() is deprecated, use arun() instead", DeprecationWarning, stacklevel=2)
        async for response in self.arun(params):
            yield response


class SyncModelClient:
    """Synchronous base class for model clients."""

    provider: str = "generic"

    # Reuse the same metrics from async version to avoid duplication
    _counter = ModelClient._counter
    _histogram = ModelClient._histogram
    _inflight = ModelClient._inflight
    _request_counter = ModelClient._request_counter
    _first_token = ModelClient._first_token
    _token_gap = ModelClient._token_gap
    _prompt_tokens = ModelClient._prompt_tokens
    _completion_tokens = ModelClient._completion_tokens

    def __init__(
        self, cfg: ModelConfig, client: httpx.Client | None = None, is_debug: bool = False, **_: Any
    ) -> None:
        """Create the client with static :class:`ModelConfig` and optional HTTP client."""
        self.cfg = cfg
        self._client = client or httpx.Client(http2=True, timeout=httpx.Timeout(600))
        self._tracer = trace.get_tracer(__name__)
        self._logger = logging.getLogger("model_client")
        self._is_debug = is_debug

        if self._is_debug:
            self._client.event_hooks.setdefault("request", []).append(self._log_request)
            self._client.event_hooks.setdefault("response", []).append(self._log_response)
        else:
            self._client.event_hooks.setdefault("request", []).append(self._log_request_jsonl)
            self._client.event_hooks.setdefault("response", []).append(self._log_response_jsonl)

    def _log_request(self, request: httpx.Request) -> None:
        """Log outgoing HTTP request details as a cURL command."""
        import shlex

        command = f"curl -X {request.method} '{request.url}'"
        for k, v in request.headers.items():
            command += f" \\\n  -H '{k}: {v}'"

        body_bytes = request.content
        if body_bytes:
            body_str = ""
            try:
                body_str = body_bytes.decode()
            except UnicodeDecodeError:
                body_str = "<...binary data...>"

            command += f" \\\n  -d {shlex.quote(body_str)}"

        self._logger.info(f"http request as curl:\n{command}")

    def _log_response(self, response: httpx.Response) -> None:
        """Log incoming HTTP response details."""
        log_lines = [
            f"http response: {response.status_code} {response.reason_phrase}",
            f"  url: {response.url}"
        ]

        for k, v in response.headers.items():
            log_lines.append(f"  header '{k}: {v}'")

        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            content = response.read()
            text = content.decode(response.encoding or "utf-8", "replace")
            if text:
                log_lines.append(f"  body: {text}")
            response._content = content

        self._logger.info("\n".join(log_lines))

    def _log_request_jsonl(self, request: httpx.Request) -> None:
        """Log outgoing HTTP request in JSONL format for production use."""
        body_str = ""
        if request.content:
            try:
                body_str = request.content.decode()
            except UnicodeDecodeError:
                body_str = "<binary data>"

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "http_request",
            "method": request.method,
            "url": str(request.url),
            "headers": self._sanitize_headers(dict(request.headers)),
            "body": self._sanitize_body(body_str) if body_str else None,
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        self._logger.info(json.dumps(log_data, separators=(",", ":")))

    def _log_response_jsonl(self, response: httpx.Response) -> None:
        """Log incoming HTTP response in JSONL format for production use."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "http_response",
            "status_code": response.status_code,
            "reason_phrase": response.reason_phrase,
            "url": str(response.url),
            "headers": self._sanitize_headers(dict(response.headers)),
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("text/event-stream"):
            content = response.read()
            text = content.decode(response.encoding or "utf-8", "replace")
            log_data["body"] = self._sanitize_body(text) if text else None
            response._content = content
        else:
            log_data["body"] = "<streaming response>"

        self._logger.info(json.dumps(log_data, separators=(",", ":")))

    def _sanitize_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Remove sensitive information from headers."""
        sensitive_keys = {"authorization", "x-api-key", "api-key", "bearer"}
        sanitized = {}
        for k, v in headers.items():
            if k.lower() in sensitive_keys:
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = v
        return sanitized

    def _sanitize_body(self, body: str) -> dict[str, Any] | str:
        """Remove sensitive information from request/response body."""
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                sanitized = data.copy()
                sensitive_fields = {"api_key", "authorization", "token", "secret", "password"}
                for field in sensitive_fields:
                    if field in sanitized:
                        sanitized[field] = "[REDACTED]"
                return sanitized
            return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            return body[:1000] + "..." if len(body) > 1000 else body

    @retry(wait=wait_exponential_jitter(), stop=stop_after_attempt(3))
    def run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Execute the LLM call with dynamic ``params``.
        
        Returns:
            Generator yielding ModelResponse for non-streaming calls or 
            StreamingResponse for streaming calls.
        """
        is_error = False
        self._inflight.labels(self.cfg.provider, "false").inc()
        result = "success"
        start = perf_counter()
        first = True
        last = start
        attrs = {
            "provider": self.cfg.provider,
            "model": self.cfg.model,
        }

        if "llm_request_body" not in params.trace_context:
            params.trace_context["llm_request_body"] = {
                "model": self.cfg.model,
                "provider": self.cfg.provider,
                "messages": [msg.model_dump() for msg in params.messages],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        params.trace_context["llm_response_body"] = {}
        params.trace_context["responses"] = []

        if params.request_id:
            attrs["http.request_id"] = params.request_id
        if params.session_id:
            attrs["user.session_id"] = params.session_id
        if params.user_id:
            attrs["user.id"] = params.user_id

        for key, val in (
            ("request_id", params.request_id),
            ("session_id", params.session_id),
            ("user_id", params.user_id),
        ):
            if val:
                set_baggage(key, val)

        with (
            self._tracer.start_as_current_span("llm.call", attributes=attrs),
            self._histogram.labels(self.cfg.provider).time(),
        ):
            params.trace_context["perf_metrics"] = {}
            try:
                for response in self._run(params):
                    now = perf_counter()
                    if first:
                        self._first_token.labels(self.cfg.provider, self.cfg.model).observe(now - start)
                        params.trace_context["perf_metrics"]["first_package_latency"] = now - start
                        params.trace_context["perf_metrics"]["total_latency"] = now - start
                        first = False
                    else:
                        self._token_gap.labels(self.cfg.provider, self.cfg.model).observe(now - last)
                        params.trace_context["perf_metrics"]["total_latency"] = now - start
                    last = now
                    yield response

            except Exception as e:
                is_error = True
                result = "error"
                raise
            finally:
                self._inflight.labels(self.cfg.provider, "false").dec()
                self._request_counter.labels(self.cfg.provider, result, str(is_error).lower()).inc()

    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Internal method to be implemented by subclasses.
        
        Args:
            params: Runtime parameters for the model call
            
        Returns:
            Generator yielding ModelResponse for non-streaming calls or 
            StreamingResponse for streaming calls.
        """
        raise NotImplementedError
        yield  # pragma: no cover - satisfies generator type

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
