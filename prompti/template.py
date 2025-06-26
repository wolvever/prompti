"""Prompt template with variant selection and Jinja rendering."""

import json
import re
from time import perf_counter

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


def _ctx_to_flat(ctx: dict[str, Any]) -> str:
    """Flatten context to a lowercase JSON string for token matching."""
    return json.dumps(ctx, separators=(",", ":")).lower()


class Variant(BaseModel):
    """Single experiment arm."""

    selector: list[str] = []
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

    def choose_variant(self, selector: dict[str, Any]) -> str | None:
        """Return the first variant id whose tokens all appear in ``selector``."""
        haystack = _ctx_to_flat(selector)
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
        ctx: dict[str, Any] | None = None,
        format: str = "openai",
    ) -> tuple[list[Message] | list[dict], Variant]:
        """Render the template and return messages in the requested format.

        Supported formats are ``openai`` (default), ``claude``, ``litellm`` and
        ``a2a``. ``litellm`` is an alias for ``openai`` and ``a2a`` returns the
        raw :class:`Message` objects used internally.
        """
        start = perf_counter()
        try:
            ctx = ctx or variables
            if variant is None:
                variant = self.choose_variant(ctx) or next(iter(self.variants))
            var = self.variants[variant]
            fmt = format.lower()
            if fmt in {"openai", "litellm"}:
                oa_messages: list[dict] = []

                def _part_to_str(part: dict) -> str:
                    if part.get("type") == "text":
                        txt = part.get("text", "").replace("\\n", "\n")
                        return _env.from_string(txt).render(**variables)
                    if part.get("type") == "file":
                        return f"[FILE]({part.get('file')})"
                    raise ValueError(f"Unsupported part type: {part.get('type')}")

                for msg in var.messages:
                    parts = [_part_to_str(p) for p in msg.get("parts", [])]
                    oa_messages.append({"role": msg.get("role"), "content": "\n".join(parts).strip("\n")})
                return oa_messages, var
            if fmt == "claude":
                claude_msgs: list[dict] = []

                def _part_to_block(part: dict) -> dict:
                    if part.get("type") == "text":
                        txt = part.get("text", "").replace("\\n", "\n")
                        rendered = _env.from_string(txt).render(**variables)
                        return {"type": "text", "text": rendered}
                    if part.get("type") == "file":
                        return {"type": "image", "source": {"type": "url", "url": part.get("file")}}
                    raise ValueError(f"Unsupported part type: {part.get('type')}")

                for msg in var.messages:
                    blocks = [_part_to_block(p) for p in msg.get("parts", [])]
                    claude_msgs.append({"role": msg.get("role"), "content": blocks})
                return claude_msgs, var
            if fmt == "a2a":
                messages = self._render_messages(var.messages, variables)
                return messages, var
            raise ValueError(f"Unknown format: {format}")
        finally:
            _format_latency.labels(self.name, self.version).observe(perf_counter() - start)
