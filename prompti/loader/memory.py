"""Simple loader that serves templates from an in-memory mapping."""

import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader, VersionEntry, TemplateNotFoundError


class MemoryLoader(TemplateLoader):
    """Load templates from an in-memory mapping."""

    def __init__(self, mapping: dict[str, dict[str, str]]):
        """Store the mapping of template name to template data."""
        self.mapping = mapping

    async def list_versions(self, name: str) -> list[VersionEntry]:
        """Return available versions for the template name."""
        data = self.mapping.get(name)
        if not data:
            return []

        text = data.get("yaml", "")
        ydata = yaml.safe_load(text) if text else {}
        version = str(ydata.get("version", data.get("version", "0")))
        tags = list(ydata.get("tags", []))

        return [VersionEntry(id=version, tags=tags)]

    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Return the template for the specific version."""
        data = self.mapping.get(name)
        if not data:
            raise TemplateNotFoundError(name)

        text = data.get("yaml", "")
        ydata = yaml.safe_load(text) if text else {}
        template_version = str(ydata.get("version", data.get("version", "0")))

        # Check if the requested version matches
        if version != template_version:
            raise TemplateNotFoundError(f"Version {version} not found for template {name}")

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
