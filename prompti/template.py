"""Representation and execution of Jinja2-based prompt templates."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from time import perf_counter
from typing import Any

import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from prometheus_client import Histogram
from pydantic import BaseModel, PrivateAttr

from .message import Kind, Message
from .model_client import (
    ModelClient,
    RunParams,
    ToolParams,
    ToolSpec,
)

_env = SandboxedEnvironment(undefined=StrictUndefined)

_format_latency = Histogram(
    "prompt_format_latency_seconds",
    "Time spent formatting a prompt",
    labelnames=["template_id", "version"],
)


class PromptTemplate(BaseModel):
    """Prompt template defined in YAML messages format."""

    id: str
    name: str
    version: str
    labels: list[str] = []
    required_variables: list[str] = []
    yaml: str = ""

    _data: dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:
        """Parse YAML once after model initialization."""
        self._data = yaml.safe_load(self.yaml) if self.yaml else {}

    def format(
        self,
        variables: dict[str, Any],
        tag: str | None = None,
    ) -> list[Message]:
        start = perf_counter()
        try:
            missing = [v for v in self.required_variables if v not in variables]
            if missing:
                raise KeyError(f"missing variables: {missing}")

            messages = self._data.get("messages", [])

            results: list[Message] = []
            for msg in messages:
                role = msg.get("role")
                for part in msg.get("parts", []):
                    ptype = part.get("type")
                    if ptype == "text":
                        text = part.get("text", "").replace("\\n", "\n")
                        rendered = _env.from_string(text).render(**variables)
                        # Preserve trailing spaces but remove leading/trailing newlines
                        rendered = rendered.strip("\n")
                        results.append(Message(role=role, kind=Kind.TEXT, content=rendered))
                    elif ptype == "file":
                        results.append(Message(role=role, kind="file", content=part.get("file")))
            return results
        finally:
            _format_latency.labels(self.id, self.version).observe(perf_counter() - start)

    async def run(
        self,
        variables: dict[str, Any],
        tag: str | None,
        client: ModelClient,
        *,
        tool_params: ToolParams | list[ToolSpec] | list[dict] | None = None,
        **run_params: Any,
    ) -> AsyncGenerator[Message, None]:
        """Stream results from executing the template via ``client``."""
        messages = self.format(variables, tag)
        params = RunParams(messages=messages, tool_params=tool_params, **run_params)
        async for m in client.run(params):
            yield m
