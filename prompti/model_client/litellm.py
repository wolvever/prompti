from __future__ import annotations

"""LiteLLM client implementation using the `litellm` package."""

from typing import Any, AsyncGenerator
import json
import os
import litellm

from ..message import Message
from .base import ModelClient, ModelConfig


class LiteLLMClient(ModelClient):
    """Client that routes requests through ``litellm``."""

    provider = "litellm"
    api_key_var = "LITELLM_API_KEY"
    endpoint_var = "LITELLM_ENDPOINT"

    async def _run(
        self, messages: list[Message], model_cfg: ModelConfig
    ) -> AsyncGenerator[Message, None]:
        """Translate A2A messages and execute via :func:`litellm.acompletion`."""

        oa_messages: list[dict[str, Any]] = []
        for m in messages:
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
                # drop unsupported kinds from request
                continue

            oa_messages.append(msg)

        params = dict(model_cfg.parameters)
        api_key = os.environ.get(self.api_key_var)
        base_url = os.environ.get(self.endpoint_var)

        response = await litellm.acompletion(
            model=model_cfg.model,
            messages=oa_messages,
            api_key=api_key,
            base_url=base_url,
            **params,
        )

        choice = response.choices[0]
        msg = choice.message

        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                func = call["function"]
                yield Message(
                    role="assistant",
                    kind="tool_use",
                    content=json.dumps(
                        {
                            "name": func.get("name"),
                            "arguments": json.loads(func.get("arguments", "{}")),
                        }
                    ),
                )
        elif getattr(msg, "function_call", None):
            func = msg.function_call
            yield Message(
                role="assistant",
                kind="tool_use",
                content={
                    "name": func.get("name"),
                    "arguments": json.loads(func.get("arguments", "{}")),
                },
            )
        elif getattr(msg, "content", None):
            yield Message(role="assistant", kind="text", content=msg.content)

