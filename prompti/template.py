"""Representation and execution of Jinja2-based prompt templates."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import yaml
from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel

from .message import Message
from .model_client import ModelClient, ModelConfig

_env = SandboxedEnvironment(undefined=StrictUndefined)


class PromptTemplate(BaseModel):
    """Prompt template defined in YAML messages format."""

    id: str
    name: str
    version: str
    labels: list[str] = []
    required_variables: list[str] = []
    yaml: str = ""

    def format(
        self,
        variables: dict[str, Any],
        tag: str | None = None,
    ) -> list[Message]:
        missing = [v for v in self.required_variables if v not in variables]
        if missing:
            raise KeyError(f"missing variables: {missing}")

        ydata = yaml.safe_load(self.yaml) if self.yaml else {}
        messages = ydata.get("messages", [])

        results: list[Message] = []
        for msg in messages:
            role = msg.get("role")
            for part in msg.get("parts", []):
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text", "")
                    rendered = _env.from_string(text).render(**variables)
                    results.append(Message(role=role, kind="text", content=rendered))
                elif ptype == "file":
                    results.append(
                        Message(role=role, kind="file", content=part.get("file"))
                    )
        return results

    async def run(
        self,
        variables: dict[str, Any],
        tag: str | None,
        model_cfg: ModelConfig,
        client: ModelClient,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[Message, None]:
        """Stream results from executing the template via ``client``."""
        messages = self.format(variables, tag)
        async for m in client.run(messages, model_cfg=model_cfg, tools=tools):
            yield m
