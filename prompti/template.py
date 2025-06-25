"""Prompt template with variant selection and Jinja rendering."""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any, Dict

import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from prometheus_client import Histogram
from pydantic import BaseModel, model_validator, Field

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


def _ctx_to_flat(ctx: Dict[str, Any]) -> str:
    """Flatten context to a lowercase JSON string for token matching."""
    return json.dumps(ctx, separators=(",", ":")).lower()


def choose_variant(tmpl: "PromptTemplate", ctx: Dict[str, Any]) -> str | None:
    """Return the first variant id whose tokens all appear in ``ctx``."""
    haystack = _ctx_to_flat(ctx)
    for vid, var in tmpl.variants.items():
        if all(tok.lower() in haystack for tok in var.contains):
            return vid
    return None


class Variant(BaseModel):
    """Single experiment arm."""

    contains: list[str] = []
    model_cfg: ModelConfig = Field(..., alias="model_config")
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

    @model_validator(mode="after")
    def _snake_names(self) -> "PromptTemplate":
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

    def _render_messages(self, messages: list[dict], variables: Dict[str, Any]) -> list[Message]:
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

    def format(self, variables: Dict[str, Any], *, variant: str | None = None, ctx: Dict[str, Any] | None = None) -> tuple[list[Message], Variant]:
        start = perf_counter()
        try:
            ctx = ctx or variables
            if variant is None:
                variant = choose_variant(self, ctx) or next(iter(self.variants))
            var = self.variants[variant]
            messages = self._render_messages(var.messages, variables)
            return messages, var
        finally:
            _format_latency.labels(self.name, self.version).observe(perf_counter() - start)

