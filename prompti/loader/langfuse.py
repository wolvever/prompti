from __future__ import annotations

import asyncio

import yaml

from ..template import PromptTemplate, Variant


class LangfuseLoader:
    """Load templates via the Langfuse SDK."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        base_url: str = "https://cloud.langfuse.com",
    ) -> None:
        from langfuse import get_client

        self.client = get_client(public_key=public_key, secret_key=secret_key, base_url=base_url)

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        prm = await asyncio.to_thread(self.client.prompts().get_prompt, name, label=label)
        yaml_blob = prm.yaml
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
        return tmpl.version, tmpl
