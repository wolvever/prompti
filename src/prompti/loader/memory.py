"""Simple loader that serves templates from an in-memory mapping."""

from __future__ import annotations

import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry


class MemoryLoader(TemplateLoader):
    """Load templates from an in-memory mapping."""

    def __init__(self, mapping: dict[str, dict[str, str]]):
        """Store the mapping of template name to template data."""
        self.mapping = mapping

    async def alist_versions(self, name: str) -> list[VersionEntry]:
        """Return available versions for the template name."""
        data = self.mapping.get(name)
        if not data:
            return []

        text = data.get("yaml", "")
        ydata = yaml.safe_load(text) if text else {}
        version = str(ydata.get("version", data.get("version", "0")))
        aliases = list(ydata.get("aliases", []))

        return [VersionEntry(id=version, aliases=aliases)]

    async def aget_template(self, name: str, version: str) -> PromptTemplate:
        """Return the template for the specific version."""
        data = self.mapping.get(name)
        if not data:
            raise TemplateNotFoundError(name)

        text = data.get("yaml", "")
        ydata = yaml.safe_load(text) if text else {}
        template_version = str(ydata.get("version", data.get("version", "0")))

        # Check if the requested version matches
        if version and version != template_version:
            raise TemplateNotFoundError(f"Version {version} not found for template {name}")

        tmpl = PromptTemplate(
            id=name,
            name=ydata.get("name", name),
            description=ydata.get("description", ""),
            version=template_version,
            aliases=list(ydata.get("aliases", [])),
            variants={k: Variant(**v) for k, v in ydata.get("variants", {}).items()},
        )
        return tmpl

    def list_versions_sync(self, name: str) -> list[VersionEntry]:
        """Synchronous version of alist_versions."""
        data = self.mapping.get(name)
        if not data:
            return []

        text = data.get("yaml", "")
        ydata = yaml.safe_load(text) if text else {}
        version = str(ydata.get("version", data.get("version", "0")))
        aliases = list(ydata.get("aliases", []))

        return [VersionEntry(id=version, aliases=aliases)]

    def get_template_sync(self, name: str, version: str) -> PromptTemplate:
        """Synchronous version of aget_template."""
        data = self.mapping.get(name)
        if not data:
            raise TemplateNotFoundError(name)

        text = data.get("yaml", "")
        ydata = yaml.safe_load(text) if text else {}
        template_version = str(ydata.get("version", data.get("version", "0")))

        # Check if the requested version matches
        if version and version != template_version:
            raise TemplateNotFoundError(f"Version {version} not found for template {name}")

        tmpl = PromptTemplate(
            id=name,
            name=ydata.get("name", name),
            description=ydata.get("description", ""),
            version=template_version,
            aliases=list(ydata.get("aliases", [])),
            variants={k: Variant(**v) for k, v in ydata.get("variants", {}).items()},
        )
        return tmpl
