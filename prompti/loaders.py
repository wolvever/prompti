from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Awaitable, Tuple

import httpx

from .template import PromptTemplate


class MemoryLoader:
    def __init__(self, mapping: dict[str, dict[str, str]]):
        self.mapping = mapping

    async def __call__(self, name: str, label: str | None) -> Tuple[str, PromptTemplate]:
        data = self.mapping.get(name)
        if not data:
            raise FileNotFoundError(name)
        version = data.get("version", "0")
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=version,
            jinja_source=data.get("jinja", ""),
        )
        return version, tmpl


class HTTPLoader:
    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient()

    async def __call__(self, name: str, label: str | None) -> Tuple[str, PromptTemplate]:
        params = {"label": label} if label else {}
        resp = await self.client.get(f"{self.base_url}/templates/{name}", params=params)
        if resp.status_code != 200:
            raise FileNotFoundError(name)
        data = resp.json()
        version = data["version"]
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=version,
            jinja_source=data["jinja"],
        )
        return version, tmpl
