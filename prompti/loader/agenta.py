"""Loader that fetches prompt templates from the Agenta registry."""

from __future__ import annotations

import asyncio

import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry


class AgentaLoader(TemplateLoader):
    """Fetch templates from Agenta via the SDK."""

    def __init__(self, app_slug: str) -> None:
        """Create a loader for the given Agenta application slug."""
        import agenta as ag

        ag.init()
        self.app_slug = app_slug

    async def list_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from Agenta.

        Note: Agenta doesn't provide version listing, so we attempt to fetch
        the production version to see if the template exists.
        """
        import agenta as ag

        try:
            cfg = await asyncio.to_thread(
                ag.ConfigManager.get_from_registry,
                app_slug=self.app_slug,
                variant_slug=name,
                environment_slug="production",
            )
            yaml_blob = yaml.safe_dump(cfg["prompt"])
            meta = yaml.safe_load(yaml_blob) if yaml_blob else {}
            tags = meta.get("tags", ["production"])
            version = str(cfg.get("variant_version", "0"))

            return [VersionEntry(id=version, tags=list(tags))]
        except Exception:
            return []

    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Get specific version of template from Agenta."""
        import agenta as ag

        try:
            cfg = await asyncio.to_thread(
                ag.ConfigManager.get_from_registry,
                app_slug=self.app_slug,
                variant_slug=name,
                environment_slug="production",
                variant_version=version,
            )
        except Exception as err:
            raise TemplateNotFoundError(
                f"Template {name} version {version} not found"
            ) from err

        yaml_blob = yaml.safe_dump(cfg["prompt"])
        if not yaml_blob:
            raise TemplateNotFoundError(
                f"Template {name} version {version} has no prompt content"
            )

        meta = yaml.safe_load(yaml_blob)
        tmpl = PromptTemplate(
            id=name,
            name=meta.get("name", name),
            description=meta.get("description", ""),
            version=str(cfg.get("variant_version", "0")),
            tags=meta.get("tags", ["production"]),
            variants={k: Variant(**v) for k, v in meta.get("variants", {}).items()},
            yaml=yaml_blob,
        )
        return tmpl
