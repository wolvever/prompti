from __future__ import annotations

"""Shared logic for OpenAI-like providers."""

import json
import os
from typing import AsyncGenerator

from ..message import Message
from .base import ModelClient, ModelConfig


class _OpenAICore(ModelClient):
    api_url: str
    api_key_var: str

    async def _run(
        self, messages: list[Message], model_cfg: ModelConfig
    ) -> AsyncGenerator[Message, None]:
        oa_messages = []
        for m in messages:
            if m.kind == "text":
                oa_messages.append({"role": m.role, "content": m.content})
            elif m.kind == "tool_use":
                data = json.loads(m.content)
                oa_messages.append(
                    {
                        "role": m.role,
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": data["name"],
                                    "arguments": json.dumps(data.get("arguments", {})),
                                },
                            }
                        ],
                    }
                )

        payload = {"model": model_cfg.model, "messages": oa_messages}
        payload.update(model_cfg.parameters)
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
        elif msg.get("content"):
            yield Message(role="assistant", kind="text", content=msg["content"])
