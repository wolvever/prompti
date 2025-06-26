"""Filesystem-based loader for prompt templates."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry


class FileSystemLoader(TemplateLoader):
    """Loader that reads templates from the local filesystem."""

    def __init__(self, base: Path) -> None:
        """Create loader with a base directory."""
        self.base = base

    async def list_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from filesystem.

        For filesystem loader, we only have one version per template file.
        """
        path = self.base / f"{name}.yaml"
        if not path.exists():
            return []

        try:
            text = path.read_text()
            data = yaml.safe_load(text)
            version = str(data.get("version", "0"))
            tags = list(data.get("tags", []))

            return [VersionEntry(id=version, tags=tags)]
        except (FileNotFoundError, yaml.YAMLError, KeyError):
            return []

    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Load and return the template identified by name and version."""
        path = self.base / f"{name}.yaml"
        if not path.exists():
            raise TemplateNotFoundError(name)

        text = path.read_text()
        data = yaml.safe_load(text)
        template_version = str(data.get("version", "0"))

        # Check if the requested version matches
        if version != template_version:
            raise TemplateNotFoundError(f"Version {version} not found for template {name}")

        tmpl = PromptTemplate(
            id=name,
            name=data.get("name", name),
            description=data.get("description", ""),
            version=template_version,
            tags=list(data.get("tags", [])),
            variants={k: Variant(**v) for k, v in data.get("variants", {}).items()},
            yaml=text,
        )
        return tmpl
