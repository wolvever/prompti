"""Load prompt templates using the Pezzo cloud service."""

import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader, VersionEntry


class PezzoLoader(TemplateLoader):
    """Retrieve prompts via the Pezzo client."""

    def __init__(self, project: str) -> None:
        """Initialize the loader for the given Pezzo project."""
        from pezzo import PezzoClient

        self.client = PezzoClient(project=project)

    async def list_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from Pezzo.

        Note: Pezzo doesn't provide version listing, so we attempt to fetch
        the production version to see if the template exists.
        """
        try:
            prompt = await self.client.get_prompt(slug=name, environment="production")
            yaml_blob = prompt["yaml"]
            meta = yaml.safe_load(yaml_blob) if yaml_blob else {}
            tags = meta.get("tags", prompt.get("tags", []))
            version = str(prompt["version"])

            return [VersionEntry(id=version, tags=list(tags))]
        except Exception:
            return []

    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Get specific version of template from Pezzo."""
        try:
            prompt = await self.client.get_prompt(slug=name, environment="production", version=version)
        except Exception:
            raise FileNotFoundError(f"Template {name} version {version} not found")

        yaml_blob = prompt["yaml"]
        if not yaml_blob:
            raise FileNotFoundError(f"Template {name} version {version} has no YAML content")

        meta = yaml.safe_load(yaml_blob)
        tmpl = PromptTemplate(
            id=name,
            name=meta.get("name", name),
            description=meta.get("description", ""),
            version=str(prompt["version"]),
            tags=meta.get("tags", prompt.get("tags", [])),
            variants={k: Variant(**v) for k, v in meta.get("variants", {}).items()},
            yaml=yaml_blob,
        )
        return tmpl
