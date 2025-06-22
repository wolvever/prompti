"""OpenAI client implementation."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from ..message import Message
from .base import ModelClient, ModelConfig, RunParams, ToolChoice, ToolParams, ToolSpec


class OpenAIClient(ModelClient):
    """Client for the OpenAI chat completion API."""

    provider = "openai"
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key_var = "OPENAI_API_KEY"

    def __init__(
        self,
        cfg: ModelConfig,
        client: httpx.AsyncClient | None = None,
        *,
        api_url: str | None = None,
        api_key_var: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(cfg, client)
        if api_url is not None:
            self.api_url = api_url
        if api_key_var is not None:
            self.api_key_var = api_key_var
        if api_key is not None:
            self.cfg.api_key = api_key

    async def _run(self, p: RunParams) -> AsyncGenerator[Message, None]:  # noqa: C901
        oa_messages: list[dict[str, Any]] = []
        for m in p.messages:
            role = m.role
            if m.kind == "tool_result":
                role = "tool"
            msg: dict[str, Any] = {"role": role}

            if m.kind in {"text", "thinking"}:
                msg["content"] = m.content
            elif m.kind == "image_url":
                msg["content"] = [
                    {"type": "image_url", "image_url": {"url": m.content}}
                ]
            elif m.kind == "tool_use":
                data = (
                    m.content if isinstance(m.content, dict) else json.loads(m.content)
                )
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
                    json.dumps(m.content)
                    if not isinstance(m.content, str)
                    else m.content
                )
            else:
                continue

            oa_messages.append(msg)

        payload: dict[str, Any] = {"model": self.cfg.model, "messages": oa_messages}

        for key in ("temperature", "top_p", "max_tokens", "n", "seed", "logit_bias"):
            value = getattr(p, key)
            if value is not None:
                payload[key] = value
        if p.stream is False:
            payload["stream"] = False
        if p.stop:
            payload["stop"] = p.stop
        if p.response_format:
            payload["response_format"] = {"type": p.response_format}
        if p.user_id:
            payload["user"] = p.user_id

        # tool parameters
        if p.tool_params:
            if isinstance(p.tool_params, ToolParams):
                tools = [
                    (
                        {"type": "function", "function": t.model_dump()}
                        if isinstance(t, ToolSpec)
                        else t
                    )
                    for t in p.tool_params.tools
                ]
                payload["tools"] = tools
                choice = p.tool_params.choice
                if isinstance(choice, ToolChoice):
                    if choice is not ToolChoice.AUTO:
                        payload["tool_choice"] = choice.value
                elif choice is not None:
                    payload["tool_choice"] = choice
            else:
                # assume raw list
                payload["tools"] = [
                    (
                        {"type": "function", "function": t.model_dump()}
                        if isinstance(t, ToolSpec)
                        else t
                    )
                    for t in p.tool_params
                ]

        payload.update(p.extra_params)

        api_key = self.cfg.api_key or os.environ.get(self.api_key_var, "")
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = await self._client.post(self.api_url, json=payload, headers=headers)
        if resp.status_code != 200:
            yield Message(role="assistant", kind="error", content=resp.text)
            return
        data = resp.json()
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        if usage:
            pt = (
                usage.get("prompt_tokens")
                or usage.get("prompt_token")
                or usage.get("input_tokens")
            )
            ct = usage.get("completion_tokens") or usage.get("output_tokens")
            if pt is not None:
                self._prompt_tokens.labels(self.cfg.provider, self.cfg.model).inc(pt)
            if ct is not None:
                self._completion_tokens.labels(self.cfg.provider, self.cfg.model).inc(
                    ct
                )
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
