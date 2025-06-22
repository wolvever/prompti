from __future__ import annotations

import yaml

from ..template import PromptTemplate


class PezzoLoader:
    """Retrieve prompts via the Pezzo client."""

    def __init__(self, project: str) -> None:
        from pezzo import PezzoClient

        self.client = PezzoClient(project=project)

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        prompt = await self.client.get_prompt(slug=name, environment="production", version_tag=label)
        yaml_blob = prompt["yaml"]
        meta = yaml.safe_load(yaml_blob)
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=str(prompt["version"]),
            labels=meta.get("labels", prompt.get("tags", [])),
            yaml=yaml_blob,
        )
        return tmpl.version, tmpl
