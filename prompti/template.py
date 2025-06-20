from __future__ import annotations

"""Representation and execution of Jinja2-based prompt templates."""

from typing import Any, AsyncGenerator
import inspect
import json
from pydantic import BaseModel
from jinja2.sandbox import SandboxedEnvironment
from jinja2 import StrictUndefined

from .message import Message
from .model_client import ModelConfig, ModelClient

_env = SandboxedEnvironment(undefined=StrictUndefined)


class PromptTemplate(BaseModel):
    """Prompt template defined in YAML messages format."""

    id: str
    name: str
    version: str
    labels: list[str] = []
    required_variables: list[str] = []
    messages: list[dict]

    def format(
        self,
        variables: dict[str, Any],
        tag: str | None = None,
    ) -> list[Message]:
        missing = [v for v in self.required_variables if v not in variables]
        if missing:
            raise KeyError(f"missing variables: {missing}")

        results: list[Message] = []
        for msg in self.messages:
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
        tool_funcs: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[Message, None]:
        """Stream results from executing the template via ``client``."""
        history = self.format(variables, tag)
        while True:
            used = False
            async for msg in client.run(history, model_cfg=model_cfg, tools=tools):
                history.append(msg)
                yield msg
                if msg.kind == "tool_use" and tool_funcs:
                    data = msg.content
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except Exception:
                            data = {}
                    func = tool_funcs.get(data.get("name")) if isinstance(data, dict) else None
                    args = data.get("arguments", {}) if isinstance(data, dict) else {}
                    if func:
                        if inspect.iscoroutinefunction(func):
                            res = await func(**args)
                        else:
                            res = func(**args)
                        tool_msg = Message(role="tool", kind="tool_result", content=res)
                        history.append(tool_msg)
                        yield tool_msg
                        used = True
            if not used:
                break
