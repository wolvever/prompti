"""Prompt template with variant selection and Jinja rendering."""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from prometheus_client import Histogram
from pydantic import BaseModel, Field, model_validator

from .message import Kind, Message
from .model_client import ModelConfig

_env = SandboxedEnvironment(undefined=StrictUndefined)

_format_latency = Histogram(
    "prompt_format_latency_seconds",
    "Time spent formatting a prompt",
    labelnames=["template_name", "version"],
    registry=None,
)

SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def _selector_to_flat(selector: dict[str, Any]) -> str:
    """Flatten selector to a lowercase JSON string for token matching."""
    return json.dumps(selector, separators=(",", ":")).lower()


class Variant(BaseModel):
    """Single experiment arm."""

    selector: list[str] = []
    model_cfg: ModelConfig | None = Field(None, alias="model_config")
    messages: list[dict]
    tools: list[dict] | None = None


class PromptTemplate(BaseModel):
    """Prompt template with multiple variants."""

    name: str
    description: str = ""
    version: str
    tags: list[str] = []
    variants: dict[str, Variant]
    yaml: str = ""
    id: str | None = None

    def choose_variant(self, selector: dict[str, Any]) -> str | None:
        """Return the first variant id whose tokens all appear in ``selector``."""
        haystack = _selector_to_flat(selector)
        for vid, var in self.variants.items():
            if all(tok.lower() in haystack for tok in var.selector):
                return vid
        return None

    @model_validator(mode="after")
    def _snake_names(self) -> PromptTemplate:
        if not SNAKE.fullmatch(self.name):
            raise ValueError("name must be lower_snake_case")
        bad = [v for v in self.variants if not SNAKE.fullmatch(v)]
        if bad:
            raise ValueError(f"variant IDs not snake_case: {bad}")
        if self.id is None:
            self.id = self.name
        return self

    def model_post_init(self, _context: Any) -> None:  # type: ignore[override]
        """Store raw YAML and populate variants if created from text."""
        if self.yaml and not self.variants:
            data = yaml.safe_load(self.yaml)
            self.description = data.get("description", self.description)
            self.version = str(data.get("version", self.version))
            self.tags = data.get("tags", self.tags) or []
            self.variants = {k: Variant(**v) for k, v in data.get("variants", {}).items()}

    def _render_messages(self, messages: list[dict], variables: dict[str, Any]) -> list[Message]:
        result: list[Message] = []
        for msg in messages:
            role = msg.get("role")
            for part in msg.get("parts", []):
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text", "").replace("\\n", "\n")
                    rendered = _env.from_string(text).render(**variables)
                    rendered = rendered.strip("\n")
                    result.append(Message(role=role, kind=Kind.TEXT, content=rendered))
                elif ptype == "file":
                    result.append(Message(role=role, kind="file", content=part.get("file")))
        return result

    def format(
        self,
        variables: dict[str, Any],
        *,
        variant: str | None = None,
        selector: dict[str, Any] | None = None,
        format: str = "openai",
    ) -> tuple[list[Message] | list[dict], Variant]:
        """Render the template and return messages in the requested format.

        Supported formats are ``openai`` (default), ``claude``, ``litellm`` and
        ``a2a``. ``litellm`` is an alias for ``openai`` and ``a2a`` returns the
        raw :class:`Message` objects used internally.
        """
        start = perf_counter()
        try:
            selector = selector or variables
            variant = variant or self.choose_variant(selector) or next(iter(self.variants))
            var = self.variants[variant]
            fmt = format.lower()

            formatter_map = {
                "openai": self._format_openai,
                "litellm": self._format_openai,
                "claude": self._format_claude,
                "a2a": self._format_a2a,
            }

            if fmt not in formatter_map:
                raise ValueError(f"Unknown format: {format}")

            return formatter_map[fmt](var, variables), var
        finally:
            _format_latency.labels(self.name, self.version).observe(perf_counter() - start)

    def _format_openai(self, variant: Variant, variables: dict[str, Any]) -> list[dict]:
        """Format messages for OpenAI API format."""
        return [
            {
                "role": msg.get("role"),
                "content": "\n".join(self._render_part_as_text(p, variables) for p in msg.get("parts", [])).strip("\n"),
            }
            for msg in variant.messages
        ]

    def _format_claude(self, variant: Variant, variables: dict[str, Any]) -> list[dict]:
        """Format messages for Claude API format."""
        return [
            {
                "role": msg.get("role"),
                "content": [self._render_part_as_block(p, variables) for p in msg.get("parts", [])],
            }
            for msg in variant.messages
        ]

    def _format_a2a(self, variant: Variant, variables: dict[str, Any]) -> list[Message]:
        """Format messages as internal Message objects."""
        return self._render_messages(variant.messages, variables)

    def _render_part_as_text(self, part: dict, variables: dict[str, Any]) -> str:
        """Render a message part as plain text."""
        if part.get("type") == "text":
            txt = part.get("text", "").replace("\\n", "\n")
            return _env.from_string(txt).render(**variables)
        if part.get("type") == "file":
            return f"[FILE]({part.get('file')})"
        raise ValueError(f"Unsupported part type: {part.get('type')}")

    def _render_part_as_block(self, part: dict, variables: dict[str, Any]) -> dict:
        """Render a message part as a structured block."""
        if part.get("type") == "text":
            txt = part.get("text", "").replace("\\n", "\n")
            rendered = _env.from_string(txt).render(**variables)
            return {"type": "text", "text": rendered}
        if part.get("type") == "file":
            return {"type": "image", "source": {"type": "url", "url": part.get("file")}}
        raise ValueError(f"Unsupported part type: {part.get('type')}")
