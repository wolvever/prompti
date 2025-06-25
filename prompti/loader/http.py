from __future__ import annotations

import httpx
import yaml

from ..template import PromptTemplate, Variant


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
            name=ydata.get("name", name),
            description=ydata.get("description", ""),
            version=version,
            tags=list(ydata.get("tags", [])),
            variants={k: Variant(**v) for k, v in ydata.get("variants", {}).items()},
            yaml=text,
        )
        return version, tmpl
