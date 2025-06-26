"""Fetch prompt templates from a remote HTTP service."""

import httpx
import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader, VersionEntry, TemplateNotFoundError


class HTTPLoader(TemplateLoader):
    """Fetch templates from an HTTP endpoint."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        """Initialize with ``base_url`` for the template registry."""
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient()

    async def list_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from HTTP endpoint."""
        try:
            resp = await self.client.get(f"{self.base_url}/templates/{name}/versions")
            if resp.status_code != 200:
                return []

            versions_data = resp.json()
            return [VersionEntry(id=str(v.get("version", "0")), tags=list(v.get("tags", []))) for v in versions_data]
        except (httpx.RequestError, ValueError, KeyError):
            return []

    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Retrieve specific version of template from the remote registry."""
        resp = await self.client.get(f"{self.base_url}/templates/{name}/{version}")
        if resp.status_code != 200:
            raise TemplateNotFoundError(
                f"Template {name} version {version} not found"
            )

        data = resp.json()
        text = data.get("yaml", "")
        ydata = yaml.safe_load(text)
        template_version = str(ydata.get("version", data.get("version", "0")))

        tmpl = PromptTemplate(
            id=name,
            name=ydata.get("name", name),
            description=ydata.get("description", ""),
            version=template_version,
            tags=list(ydata.get("tags", [])),
            variants={k: Variant(**v) for k, v in ydata.get("variants", {}).items()},
            yaml=text,
        )
        return tmpl
