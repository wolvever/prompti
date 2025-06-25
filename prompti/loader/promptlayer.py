from __future__ import annotations

import httpx
import yaml

from ..template import PromptTemplate, Variant, ModelConfig
from .base import TemplateLoader


class PromptLayerLoader(TemplateLoader):
    """Load templates from PromptLayer."""

    URL = "https://api.promptlayer.com/prompt-templates"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.client = client or httpx.AsyncClient()

    async def load(self, name: str, tags: str | None) -> tuple[str, PromptTemplate]:
        body = {"label": tags} if tags else {}
        resp = await self.client.post(
            f"{self.URL}/{name}",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json=body,
        )
        if resp.status_code != 200:
            raise FileNotFoundError(name)
        data = resp.json()
        content = data["prompt_template"]["content"]
        yaml_blob = yaml.safe_dump({"variants": {"default": {"model_config": {"provider": "litellm", "model": "unknown"}, "messages": content}}})
        tmpl = PromptTemplate(
            id=name,
            name=name,
            description="",
            version=str(data["version"]),
            tags=[tags] if tags else [],
            variants={"default": Variant(model_config=ModelConfig(provider="litellm", model="unknown"), messages=content)},
            yaml=yaml_blob,
        )
        return tmpl.version, tmpl
