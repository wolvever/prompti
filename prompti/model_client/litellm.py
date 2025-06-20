from __future__ import annotations

"""LiteLLM client implementation using the ``litellm`` package."""

import json
import os
from typing import Any, AsyncGenerator

import httpx

import litellm

from ..message import Message
from .base import ModelClient, ModelConfig


class LiteLLMClient(ModelClient):
    """Client that delegates LLM calls to ``litellm``."""

    provider = "litellm"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        client = client or httpx.AsyncClient(http2=False)
        super().__init__(client=client)

    async def _run(
        self, messages: list[Message], model_cfg: ModelConfig
    ) -> AsyncGenerator[Message, None]:
        oa_messages: list[dict[str, Any]] = []
        for m in messages:
            role = m.role
            if m.kind == "tool_result":
                role = "tool"

            msg: dict[str, Any] = {"role": role}

            if m.kind in ("text", "thinking"):
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
                msg["content"] = (
                    json.dumps(m.content) if not isinstance(m.content, str) else m.content
                )
            else:
                continue

            oa_messages.append(msg)

        payload = {"model": model_cfg.model, "messages": oa_messages}
        payload.update(model_cfg.parameters)
        endpoint = os.environ.get("LITELLM_ENDPOINT")
        api_key = os.environ.get("LITELLM_API_KEY")
        if endpoint:
            payload["base_url"] = endpoint
        if api_key:
            payload["api_key"] = api_key

        try:
            resp = await litellm.acompletion(**payload)
        except Exception as exc:  # pragma: no cover - network errors
            yield Message(role="assistant", kind="error", content=str(exc))
            return

        msg = resp.choices[0].message
        if msg.tool_calls:
            for call in msg.tool_calls:
                func = call.function
                yield Message(
                    role="assistant",
                    kind="tool_use",
                    content=json.dumps(
                        {
                            "name": func.name,
                            "arguments": json.loads(func.arguments or "{}"),
                        }
                    ),
                )
        elif msg.function_call:
            func = msg.function_call
            yield Message(
                role="assistant",
                kind="tool_use",
                content={
                    "name": func.name,
                    "arguments": json.loads(func.arguments or "{}"),
                },
            )
        elif msg.content:
            yield Message(role="assistant", kind="text", content=msg.content)

