from __future__ import annotations

import asyncio
import yaml

from ..template import PromptTemplate


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
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=str(cfg.get("variant_version", "0")),
            labels=[label] if label else ["production"],
            yaml=yaml_blob,
        )
        return tmpl.version, tmpl
