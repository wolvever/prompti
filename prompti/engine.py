"""Core engine that resolves templates and executes them with model clients."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from typing import Any

import yaml
from async_lru import alru_cache
from opentelemetry import trace
from prometheus_client import Counter
from pydantic import BaseModel

from .experiment import ExperimentRegistry, bucket
from .loaders import HTTPLoader, MemoryLoader
from .loader import FileSystemLoader
from .message import Message
from .model_client import ModelClient, ModelConfig
from .template import PromptTemplate

_tracer = trace.get_tracer(__name__)
ab_counter = Counter(
    "prompt_ab_request_total",
    "AB experiment requests",
    labelnames=["experiment", "variant"],
)

TemplateLoader = Callable[
    [str, str | None], Awaitable[tuple[str, PromptTemplate]]
]





class PromptEngine:
    """Resolve templates and generate model responses."""

    def __init__(self, loaders: list[TemplateLoader], cache_ttl: int = 300) -> None:
        """Initialize the engine with a list of loaders."""
        self._loaders = loaders
        self._cache_ttl = cache_ttl
        self._resolve = alru_cache(maxsize=128, ttl=cache_ttl)(self._resolve)

    async def _resolve(self, name: str, label: str | None) -> PromptTemplate:
        for loader in self._loaders:
            version, tmpl = await loader(name, label)
            if tmpl:
                return tmpl
        raise FileNotFoundError(name)

    async def format(
        self, template_name: str, variables: dict[str, Any], tags: str | None = None
    ) -> list[Message]:
        """Return formatted messages for ``template_name``."""
        tmpl = await self._resolve(template_name, tags)
        return tmpl.format(variables, tags)

    async def run(
        self,
        template_name: str,
        variables: dict[str, Any],
        tags: str | None,
        model_cfg: ModelConfig,
        client: ModelClient,
        *,
        headers: dict[str, str] | None = None,
        registry: ExperimentRegistry | None = None,
        user_id: str = "anon",
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[Message, None]:
        """Stream messages produced by running the template via ``client``."""
        tmpl = await self._resolve(template_name, tags)

        exp_id: str | None = None
        variant: str | None = None
        if headers and "x-variant" in headers:
            exp_id = headers.get("x-exp", "") or None
            variant = headers.get("x-variant")
        elif registry is not None:
            split = await registry.get_split(template_name, user_id)
            exp_id = split.experiment_id
            variant = split.variant or bucket(user_id, split.traffic_split or {})

        tag = None
        if exp_id and variant:
            candidate = f"{exp_id}={variant}"
            if candidate in tmpl.labels:
                tag = candidate
        if tag is None and "prod" in tmpl.labels:
            tag = "prod"

        ab_counter.labels(exp_id or "none", variant or "control").inc()
        with _tracer.start_as_current_span(
            "prompt.run",
            attributes={
                "prompt.version": tmpl.version,
                "ab.experiment": exp_id or "none",
                "ab.variant": variant or "control",
            },
        ):
            async for msg in tmpl.run(
                variables,
                tag,
                model_cfg,
                client=client,
                tools=tools,
            ):
                yield msg

    @classmethod
    def from_setting(cls, setting: Setting) -> PromptEngine:
        """Create an engine instance from a :class:`Setting`."""
        loaders = [FileSystemLoader(Path(p)) for p in setting.template_paths]
        if setting.registry_url:
            loaders.append(HTTPLoader(setting.registry_url))
        if setting.memory_templates:
            loaders.append(MemoryLoader(setting.memory_templates))
        return cls(loaders, cache_ttl=setting.cache_ttl)


class Setting(BaseModel):
    """Configuration options for :class:`PromptEngine`."""
    template_paths: list[Path] = [Path("./prompts")]
    cache_ttl: int = 300
    registry_url: str | None = None
    memory_templates: dict[str, dict[str, str]] | None = None

