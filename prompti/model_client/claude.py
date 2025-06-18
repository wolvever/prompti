from __future__ import annotations

"""Claude (Anthropic) client implementation."""

import os
from typing import AsyncGenerator

from ..message import Message
from .base import ModelClient, ModelConfig


class ClaudeClient(ModelClient):
    """Client for Anthropic Claude models."""

    provider = "claude"

    async def _run(
        self, messages: list[Message], model_cfg: ModelConfig
    ) -> AsyncGenerator[Message, None]:
        url = "https://api.anthropic.com/v1/messages"
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}

        prompt = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.kind == "text"
        ]
        payload = {
            "model": model_cfg.model,
            "messages": prompt,
            "max_tokens": model_cfg.parameters.get("max_tokens", 16),
        }
        resp = await self._client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            yield Message(role="assistant", kind="error", content=resp.text)
            return
        data = resp.json()
        content = data.get("content", "") if isinstance(data, dict) else ""
        yield Message(role="assistant", kind="text", content=content)
