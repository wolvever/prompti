from __future__ import annotations

import httpx
import yaml

from ..template import PromptTemplate


class PromptLayerLoader:
    """Load templates from PromptLayer."""

    URL = "https://api.promptlayer.com/prompt-templates"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.client = client or httpx.AsyncClient()

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        body = {"label": label} if label else {}
        resp = await self.client.post(
            f"{self.URL}/{name}",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json=body,
        )
        if resp.status_code != 200:
            raise FileNotFoundError(name)
        data = resp.json()
        content = data["prompt_template"]["content"]
        yaml_blob = yaml.safe_dump({"messages": content})
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=str(data["version"]),
            labels=[label] if label else [],
            yaml=yaml_blob,
        )
        return tmpl.version, tmpl
