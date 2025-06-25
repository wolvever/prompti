"""Core engine that resolves templates and executes them with model clients."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Dict

from async_lru import alru_cache
from opentelemetry import trace

from pydantic import BaseModel

from .loader import FileSystemLoader, HTTPLoader, MemoryLoader, TemplateLoader
from .message import Message
from .model_client import ModelClient, RunParams, ToolParams, ToolSpec
from .template import PromptTemplate

_tracer = trace.get_tracer(__name__)


class PromptEngine:
    """Resolve templates and generate model responses."""

    def __init__(self, loaders: list[TemplateLoader], cache_ttl: int = 300) -> None:
        """Initialize the engine with a list of loaders."""
        self._loaders = loaders
        self._cache_ttl = cache_ttl
        self._resolve = alru_cache(maxsize=128, ttl=cache_ttl)(self._resolve_impl)

    async def _resolve_impl(self, name: str, label: str | None) -> PromptTemplate:
        for loader in self._loaders:
            version, tmpl = await loader(name, label)
            if tmpl:
                return tmpl
        raise FileNotFoundError(name)

    async def load(self, template_name: str) -> PromptTemplate:
        """Public entry: resolve & cache a template by name."""
        return await self._resolve(template_name, None)

    async def format(
        self,
        template_name: str,
        variables: Dict[str, Any],
        *,
        variant: str | None = None,
        ctx: Dict[str, Any] | None = None,
    ) -> list[Message]:
        """Return formatted messages for ``template_name``."""
        tmpl = await self._resolve(template_name, None)
        msgs, _ = tmpl.format(variables, variant=variant, ctx=ctx)
        return msgs

    async def run(
        self,
        template_name: str,
        variables: Dict[str, Any],
        client: ModelClient,
        *,
        variant: str | None = None,
        ctx: Dict[str, Any] | None = None,
        tool_params: ToolParams | list[ToolSpec] | list[dict] | None = None,
        **run_params: Any,
    ) -> AsyncGenerator[Message, None]:
        """Stream messages produced by running the template via ``client``."""
        tmpl = await self._resolve(template_name, None)
        ctx = ctx or variables

        if variant is None:
            variant = tmpl.choose_variant(ctx) or next(iter(tmpl.variants))

        messages, var = tmpl.format(variables, variant=variant, ctx=ctx)
        params = RunParams(messages=messages, tool_params=tool_params, **run_params)

        with _tracer.start_as_current_span(
            "prompt.run",
            attributes={
                "template.name": tmpl.name,
                "template.version": tmpl.version,
                "variant": variant,
            },
        ):
            client.cfg = var.model_cfg
            async for msg in client.run(params):
                yield msg

    @classmethod
    def from_setting(cls, setting: Setting) -> PromptEngine:
        """Create an engine instance from a :class:`Setting`."""
        loaders: list[TemplateLoader] = [FileSystemLoader(Path(p)) for p in setting.template_paths]
        if setting.registry_url:
            loaders.append(HTTPLoader(setting.registry_url))
        if setting.memory_templates:
            loaders.append(MemoryLoader(setting.memory_templates))
        if setting.config_loader:
            loaders.append(setting.config_loader)
        return cls(loaders, cache_ttl=setting.cache_ttl)


class Setting(BaseModel):
    """Configuration options for :class:`PromptEngine`."""

    model_config = {
        "arbitrary_types_allowed": True,
    }

    template_paths: list[Path] = [Path("./prompts")]
    cache_ttl: int = 300
    registry_url: str | None = None
    memory_templates: dict[str, dict[str, str]] | None = None
    config_loader: TemplateLoader | None = None
