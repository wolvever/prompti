"""Load templates from the PromptLayer service."""

from __future__ import annotations

import httpx
import yaml

from ..template import ModelConfig, PromptTemplate, Variant
from .base import TemplateLoader, VersionEntry, TemplateNotFoundError


class PromptLayerLoader(TemplateLoader):
    """Load templates from PromptLayer."""

    URL = "https://api.promptlayer.com/prompt-templates"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        """Create the loader with API key and optional HTTP client."""
        self.api_key = api_key
        self.client = client or httpx.AsyncClient()

    async def list_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from PromptLayer.

        Note: PromptLayer doesn't provide version listing, so we attempt to fetch
        the default version to see if the template exists.
        """
        try:
            resp = await self.client.post(
                f"{self.URL}/{name}/versions",
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={},
            )
            if resp.status_code != 200:
                return []

            versions_data = resp.json()
            return [VersionEntry(id=str(v.get("version", "0")), tags=list(v.get("tags", []))) for v in versions_data]
        except (httpx.RequestError, ValueError, KeyError):
            # Fallback: try to get current version to see if template exists
            try:
                resp = await self.client.post(
                    f"{self.URL}/{name}",
                    headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                    json={},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    version = str(data.get("version", "0"))
                    return [VersionEntry(id=version, tags=[])]
            except (httpx.RequestError, ValueError, KeyError):
                pass
            return []

    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Get specific version of template from PromptLayer."""
        body = {"version": version}
        resp = await self.client.post(
            f"{self.URL}/{name}",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json=body,
        )
        if resp.status_code != 200:
            raise TemplateNotFoundError(
                f"Template {name} version {version} not found"
            )

        data = resp.json()
        content = data["prompt_template"]["content"]
        template_version = str(data["version"])

        yaml_blob = yaml.safe_dump(
            {
                "variants": {
                    "default": {"model_config": {"provider": "litellm", "model": "unknown"}, "messages": content}
                }
            }
        )

        tmpl = PromptTemplate(
            id=name,
            name=name,
            description="",
            version=template_version,
            tags=[],
            variants={
                "default": Variant(model_config=ModelConfig(provider="litellm", model="unknown"), messages=content)
            },
            yaml=yaml_blob,
        )
        return tmpl
