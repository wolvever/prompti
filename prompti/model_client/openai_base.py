"""Shared logic for OpenAI-like providers."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

from ..message import Message
from .base import ModelClient, ModelConfig


class _OpenAICore(ModelClient):
    """Shared logic for OpenAI-compatible providers."""

    api_url: str
    api_key_var: str

    async def _run(  # noqa: C901
        self,
        messages: list[Message],
        model_cfg: ModelConfig,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[Message, None]:
        """Translate A2A messages to the OpenAI format and stream the result."""

        oa_messages: list[dict[str, Any]] = []
        for m in messages:
            role = m.role
            if m.kind == "tool_result":
                role = "tool"

            msg: dict[str, Any] = {"role": role}

            if m.kind == "text" or m.kind == "thinking":
                msg["content"] = m.content
            elif m.kind == "image_url":
                msg["content"] = [
                    {"type": "image_url", "image_url": {"url": m.content}}
                ]
            elif m.kind == "tool_use":
                data = m.content if isinstance(m.content, dict) else json.loads(m.content)
                msg["content"] = None
                msg["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": data.get("name"),
                            "arguments": json.dumps(data.get("arguments", {})),
                        },
                    }
                ]
            elif m.kind == "tool_result":
                # Tool results are provided as role ``tool`` messages.
                msg["content"] = (
                    json.dumps(m.content) if not isinstance(m.content, str) else m.content
                )
            else:
                # drop unsupported kinds from request
                continue

            oa_messages.append(msg)

        payload = {"model": model_cfg.model, "messages": oa_messages}
        payload.update(model_cfg.parameters)
        if tools is not None:
            payload["tools"] = tools
        api_key = os.environ.get(self.api_key_var, "")
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = await self._client.post(self.api_url, json=payload, headers=headers)
        if resp.status_code != 200:
            yield Message(role="assistant", kind="error", content=resp.text)
            return
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                func = call["function"]
                yield Message(
                    role="assistant",
                    kind="tool_use",
                    content=json.dumps(
                        {
                            "name": func["name"],
                            "arguments": json.loads(func.get("arguments", "{}")),
                        }
                    ),
                )
        elif "function_call" in msg:
            func = msg["function_call"]
            yield Message(
                role="assistant",
                kind="tool_use",
                content={
                    "name": func.get("name"),
                    "arguments": json.loads(func.get("arguments", "{}")),
                },
            )
        elif msg.get("content"):
            yield Message(role="assistant", kind="text", content=msg["content"])
