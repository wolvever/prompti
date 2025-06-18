from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, Tuple, Dict
import subprocess

from async_lru import alru_cache
from pydantic import BaseModel

from .message import Message
from .template import PromptTemplate
from .model_client import ModelConfig, ModelClient
from .loaders import HTTPLoader, MemoryLoader

TemplateLoader = Callable[[str, str | None], Awaitable[Tuple[str, PromptTemplate]]]


class FileSystemLoader:
    def __init__(self, base: Path) -> None:
        self.base = base

    async def __call__(self, name: str, label: str | None) -> Tuple[str, PromptTemplate]:
        path = self.base / f"{name}.jinja"
        text = path.read_text()
        version = "1.0.0"
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            commit = None
        tmpl = PromptTemplate(
            template_id=name,
            version=version,
            jinja_source=text,
            git_commit_id=commit,
        )
        return version, tmpl


class PromptEngine:
    def __init__(self, loaders: list[TemplateLoader], cache_ttl: int = 300) -> None:
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
        tmpl = await self._resolve(template_name, tags)
        return tmpl.format(variables, tags)

    async def run(
        self,
        template_name: str,
        variables: dict[str, Any],
        tags: str | None,
        model_cfg: ModelConfig,
        client: ModelClient | None = None,
    ) -> AsyncGenerator[Message, None]:
        tmpl = await self._resolve(template_name, tags)
        async for msg in tmpl.run(variables, tags, model_cfg, client=client):
            yield msg

    @classmethod
    def from_settings(cls, settings: "Settings") -> "PromptEngine":
        loaders = [FileSystemLoader(Path(p)) for p in settings.template_paths]
        if settings.template_registry_url:
            loaders.append(HTTPLoader(settings.template_registry_url))
        if settings.memory_templates:
            loaders.append(MemoryLoader(settings.memory_templates))
        return cls(loaders, cache_ttl=settings.cache_ttl)


class Settings(BaseModel):
    template_paths: list[Path] = [Path("./prompts")]
    cache_ttl: int = 300
    template_registry_url: str | None = None
    memory_templates: dict[str, dict[str, str]] | None = None

