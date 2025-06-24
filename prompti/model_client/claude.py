"""Claude (Anthropic) client implementation."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from ..message import Message
from .base import ModelClient, ModelConfig, RunParams, ToolChoice, ToolParams, ToolSpec


class ClaudeClient(ModelClient):
    """Client for the Claude chat completion API."""

    provider = "claude"
    api_url = "https://api.anthropic.com/v1/messages"
    api_key_var = "ANTHROPIC_API_KEY"

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None, is_debug: bool = False) -> None:
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_url = cfg.api_url or self.api_url
        self.api_key_var = cfg.api_key_var or self.api_key_var
        self.api_key = cfg.api_key or os.environ.get(self.api_key_var, "") or ""

    async def _run(  # noqa: C901
        self,
        p: RunParams,
    ) -> AsyncGenerator[Message, None]:
        """Translate A2A messages to Claude blocks and stream the response."""

        url = self.api_url
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

        claude_msgs: list[dict[str, Any]] = []
        for m in p.messages:
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

        payload: dict[str, Any] = {"model": self.cfg.model, "messages": claude_msgs}

        for key in ("temperature", "top_p", "top_k", "max_tokens"):
            value = getattr(p, key)
            if value is not None:
                payload[key] = value

        payload["stream"] = p.stream
        if p.stop:
            payload["stop_sequences"] = p.stop if isinstance(p.stop, list) else [p.stop]

        if p.tool_params:
            if isinstance(p.tool_params, ToolParams):
                payload["tools"] = [
                    (
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "input_schema": spec.parameters,
                        }
                        if isinstance(spec, ToolSpec)
                        else spec
                    )
                    for spec in p.tool_params.tools
                ]
                if p.tool_params.choice == ToolChoice.REQUIRED:
                    payload["tool_choice"] = {"type": "any"}
            else:
                payload["tools"] = [
                    (
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "input_schema": spec.parameters,
                        }
                        if isinstance(spec, ToolSpec)
                        else spec
                    )
                    for spec in p.tool_params
                ]

        payload.update(p.extra_params)
        resp = await self._client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            yield Message(role="assistant", kind="error", content=resp.text)
            return

        if p.stream:
            # Handle streaming response (SSE format)
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("delta", {})

                        # Handle content delta
                        if "text" in delta:
                            yield Message(role="assistant", kind="text", content=delta["text"])
                        elif "thinking" in delta:
                            yield Message(role="assistant", kind="thinking", content=delta["thinking"])

                        # Handle tool use delta
                        if "tool_use" in delta:
                            tool = delta["tool_use"]
                            yield Message(
                                role="assistant",
                                kind="tool_use",
                                content={
                                    "name": tool.get("name"),
                                    "arguments": tool.get("input", {}),
                                    "call_id": tool.get("id"),
                                },
                            )

                        # Handle usage reporting from the final chunk
                        usage = data.get("usage", {})
                        if usage:
                            pt = usage.get("prompt_tokens") or usage.get("input_tokens")
                            ct = usage.get("completion_tokens") or usage.get("output_tokens")
                            if pt is not None:
                                self._prompt_tokens.labels(self.cfg.provider, self.cfg.model).inc(pt)
                            if ct is not None:
                                self._completion_tokens.labels(self.cfg.provider, self.cfg.model).inc(ct)
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON lines
        else:
            # Handle non-streaming response
            data = resp.json()
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            if usage:
                pt = usage.get("prompt_tokens") or usage.get("prompt_token") or usage.get("input_tokens")
                ct = usage.get("completion_tokens") or usage.get("output_tokens")
                if pt is not None:
                    self._prompt_tokens.labels(self.cfg.provider, self.cfg.model).inc(pt)
                if ct is not None:
                    self._completion_tokens.labels(self.cfg.provider, self.cfg.model).inc(ct)

            blocks = data.get("content", []) if isinstance(data, dict) else []
            for blk in blocks:
                if blk.get("type") == "thinking":
                    yield Message(
                        role="assistant",
                        kind="thinking",
                        content=blk.get("thinking") or blk.get("text", ""),
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
