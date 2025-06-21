"""Claude (Anthropic) client implementation."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator

from ..message import Message
from .base import ModelClient, ModelConfig


class ClaudeClient(ModelClient):
    """Client for Anthropic Claude models."""

    provider = "claude"
    api_url = "https://api.anthropic.com/v1/messages"
    api_key_var = "ANTHROPIC_API_KEY"

    async def _run(
        self,
        messages: list[Message],
        model_cfg: ModelConfig,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[Message, None]:
        """Translate A2A messages to Claude blocks and stream the response."""

        url = self.api_url
        api_key = os.environ.get(self.api_key_var, "")
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

        claude_msgs: list[dict[str, Any]] = []
        for m in messages:
            blocks: list[dict[str, Any]] = []
            if m.kind == "text":
                blocks.append({"type": "text", "text": m.content})
            elif m.kind == "thinking":
                blocks.append({"type": "thinking", "thinking": m.content})
            elif m.kind in ("image", "image_url"):
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": m.content},
                    }
                )
            elif m.kind == "tool_use":
                data = m.content if isinstance(m.content, dict) else json.loads(m.content)
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": data.get("call_id"),
                        "name": data.get("name"),
                        "input": data.get("arguments", {}),
                    }
                )
            elif m.kind == "tool_result":
                blocks.append({"type": "tool_result", "content": m.content})

            if blocks:
                claude_msgs.append({"role": m.role, "content": blocks})

        payload: dict[str, Any] = {"model": model_cfg.model, "messages": claude_msgs}
        payload.update(model_cfg.parameters)
        if tools is not None:
            payload["tools"] = tools
        resp = await self._client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            yield Message(role="assistant", kind="error", content=resp.text)
            return

        data = resp.json()
        blocks = data.get("content", []) if isinstance(data, dict) else []
        for blk in blocks:
            if blk.get("type") == "thinking":
                yield Message(
                    role="assistant",
                    kind="thinking",
                    content=blk.get("thinking") or blk.get("text", ""),
                    meta={"visible": False, "signature": blk.get("signature")},
                )
            elif blk.get("type") == "tool_use":
                yield Message(
                    role="assistant",
                    kind="tool_use",
                    content={
                        "name": blk.get("name"),
                        "arguments": blk.get("input", {}),
                        "call_id": blk.get("id"),
                    },
                )
            elif blk.get("type") == "text":
                yield Message(role="assistant", kind="text", content=blk.get("text", ""))
