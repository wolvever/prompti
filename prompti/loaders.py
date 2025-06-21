"""Template loader implementations."""

from __future__ import annotations

import httpx
import yaml

from .template import PromptTemplate


class MemoryLoader:
    """Load templates from an in-memory mapping."""

    def __init__(self, mapping: dict[str, dict[str, str]]):
        """Store the mapping of template name to template data."""
        self.mapping = mapping

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        """Return the template ``name`` from the mapping."""
        data = self.mapping.get(name)
        if not data:
            raise FileNotFoundError(name)
        text = data.get("yaml", "")
        ydata = yaml.safe_load(text)
        version = str(ydata.get("version", data.get("version", "0")))
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=version,
            labels=list(ydata.get("labels", [])),
            required_variables=list(ydata.get("required_variables", [])),
            yaml=text,
        )
        return version, tmpl


class HTTPLoader:
    """Fetch templates from an HTTP endpoint."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        """Initialize with ``base_url`` for the template registry."""
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient()

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        """Retrieve ``name`` from the remote registry."""
        params = {"label": label} if label else {}
        resp = await self.client.get(f"{self.base_url}/templates/{name}", params=params)
        if resp.status_code != 200:
            raise FileNotFoundError(name)
        data = resp.json()
        text = data.get("yaml", "")
        ydata = yaml.safe_load(text)
        version = str(ydata.get("version", data.get("version", "0")))
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=version,
            labels=list(ydata.get("labels", [])),
            required_variables=list(ydata.get("required_variables", [])),
            yaml=text,
        )
        return version, tmpl
