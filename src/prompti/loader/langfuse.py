"""Load prompt templates using the Langfuse SDK."""

from __future__ import annotations

import asyncio

import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry


class LangfuseLoader(TemplateLoader):
    """Load templates via the Langfuse SDK."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        base_url: str = "https://cloud.langfuse.com",
    ) -> None:
        """Initialize the loader with API credentials."""
        from langfuse import get_client

        self.client = get_client(public_key=public_key, secret_key=secret_key, base_url=base_url)

    async def list_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from Langfuse."""
        try:
            # Get all versions of the prompt
            prompts = await asyncio.to_thread(self.client.prompts().get_prompt_versions, name)
            versions = []

            for prompt in prompts:
                yaml_blob = prompt.yaml
                meta = yaml.safe_load(yaml_blob) if yaml_blob else {}
                tags = meta.get("tags", [])
                versions.append(VersionEntry(id=str(prompt.version), tags=list(tags)))

            return versions
        except Exception:
            # Fallback: try to get current version to see if template exists
            try:
                prm = await asyncio.to_thread(self.client.prompts().get_prompt, name)
                yaml_blob = prm.yaml
                meta = yaml.safe_load(yaml_blob) if yaml_blob else {}
                tags = meta.get("tags", [])
                return [VersionEntry(id=str(prm.version), tags=list(tags))]
            except Exception:
                return []

    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Get specific version of template from Langfuse."""
        try:
            prm = await asyncio.to_thread(
                self.client.prompts().get_prompt,
                name,
                version=int(version),
            )
        except Exception as err:
            raise TemplateNotFoundError(
                f"Template {name} version {version} not found"
            ) from err

        yaml_blob = prm.yaml
        if not yaml_blob:
            raise TemplateNotFoundError(
                f"Template {name} version {version} has no YAML content"
            )

        meta = yaml.safe_load(yaml_blob)
        tmpl = PromptTemplate(
            id=name,
            name=meta.get("name", name),
            description=meta.get("description", ""),
            version=str(prm.version),
            tags=meta.get("tags", []),
            variants={k: Variant(**v) for k, v in meta.get("variants", {}).items()},
            yaml=yaml_blob,
        )
        return tmpl
