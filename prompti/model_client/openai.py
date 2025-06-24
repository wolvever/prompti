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
        is_debug: bool = False,
    ) -> None:
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_url = cfg.api_url or self.api_url
        self.api_key_var = cfg.api_key_var or self.api_key_var
        self.api_key = cfg.api_key or os.environ.get(self.api_key_var, "")

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
                msg["content"] = [{"type": "image_url", "image_url": {"url": m.content}}]
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
                msg["content"] = json.dumps(m.content) if not isinstance(m.content, str) else m.content
            else:
                continue

            oa_messages.append(msg)

        payload: dict[str, Any] = {"model": self.cfg.model, "messages": oa_messages}

        for key in ("temperature", "top_p", "max_tokens", "n", "seed", "logit_bias"):
            value = getattr(p, key)
            if value is not None:
                payload[key] = value

        payload["stream"] = p.stream
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
                    ({"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t)
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
                    ({"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t)
                    for t in p.tool_params
                ]

        payload.update(p.extra_params)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = await self._client.post(self.api_url, json=payload, headers=headers)
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
                        response_choice = data.get("choices", [{}])[0]
                        delta = response_choice.get("delta", {})

                        if "tool_calls" in delta:
                            for call in delta["tool_calls"]:
                                func = call.get("function", {})
                                if func.get("name"):
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
                        elif "content" in delta and delta["content"]:
                            yield Message(role="assistant", kind="text", content=delta["content"])

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
