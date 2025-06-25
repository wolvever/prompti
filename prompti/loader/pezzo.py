from __future__ import annotations

import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader


class PezzoLoader(TemplateLoader):
    """Retrieve prompts via the Pezzo client."""

    def __init__(self, project: str) -> None:
        from pezzo import PezzoClient

        self.client = PezzoClient(project=project)

    async def load(self, name: str, tags: str | None) -> tuple[str, PromptTemplate]:
        prompt = await self.client.get_prompt(slug=name, environment="production", version_tag=tags)
        yaml_blob = prompt["yaml"]
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
        return tmpl.version, tmpl
