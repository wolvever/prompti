from __future__ import annotations

import asyncio

import yaml

from ..template import PromptTemplate, Variant


class AgentaLoader:
    """Fetch templates from Agenta via the SDK."""

    def __init__(self, app_slug: str) -> None:
        import agenta as ag

        ag.init()
        self.app_slug = app_slug

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        cfg = await asyncio.to_thread(
            ag.ConfigManager.get_from_registry,
            app_slug=self.app_slug,
            variant_slug=name,
            environment_slug=label or "production",
        )
        yaml_blob = yaml.safe_dump(cfg["prompt"])
        meta = yaml.safe_load(yaml_blob)
        tmpl = PromptTemplate(
            id=name,
            name=meta.get("name", name),
            description=meta.get("description", ""),
            version=str(cfg.get("variant_version", "0")),
            tags=meta.get("tags", [label] if label else ["production"]),
            variants={k: Variant(**v) for k, v in meta.get("variants", {}).items()},
            yaml=yaml_blob,
        )
        return tmpl.version, tmpl
