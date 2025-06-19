from __future__ import annotations

"""Core engine that resolves templates and executes them with model clients."""

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, Tuple, Dict

from async_lru import alru_cache
from pydantic import BaseModel

from .message import Message
from .template import PromptTemplate
import yaml
from .model_client import ModelConfig, ModelClient
from .loaders import HTTPLoader, MemoryLoader

TemplateLoader = Callable[
    [str, str | None], Awaitable[Tuple[str, PromptTemplate]]
]


class FileSystemLoader:
    """Loader that reads templates from the local filesystem."""

    def __init__(self, base: Path) -> None:
        """Create loader with a base directory."""
        self.base = base

    async def __call__(self, name: str, label: str | None) -> Tuple[str, PromptTemplate]:
        """Load and return the template identified by ``name``."""
        path = self.base / f"{name}.jinja"
        text = path.read_text()
        data = yaml.safe_load(text)
        version = str(data.get("version", "0"))
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=version,
            labels=list(data.get("labels", [])),
            required_variables=list(data.get("required_variables", [])),
            messages=data.get("messages", []),
        )
        return version, tmpl


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
    ) -> AsyncGenerator[Message, None]:
        """Stream messages produced by running the template via ``client``."""
        tmpl = await self._resolve(template_name, tags)
        async for msg in tmpl.run(variables, tags, model_cfg, client=client):
            yield msg

    @classmethod
    def from_setting(cls, setting: "Setting") -> "PromptEngine":
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

