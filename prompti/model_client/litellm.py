"""LiteLLM client implementation using the `litellm` package."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import litellm

from ..message import Message
from .base import ModelClient, RunParams


class LiteLLMClient(ModelClient):
    """Client that routes requests through ``litellm``."""

    provider = "litellm"
    api_key_var = "LITELLM_API_KEY"
    endpoint_var = "LITELLM_ENDPOINT"

    async def _run(  # noqa: C901
        self,
        p: RunParams,
    ) -> AsyncGenerator[Message, None]:
        """Translate A2A messages and execute via :func:`litellm.acompletion`."""

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
                # drop unsupported kinds from request
                continue

            oa_messages.append(msg)

        params = dict(p.extra_params)
        if p.temperature is not None:
            params["temperature"] = p.temperature
        if p.max_tokens is not None:
            params["max_tokens"] = p.max_tokens
        api_key = os.environ.get(self.api_key_var) or self.cfg.api_key
        base_url = os.environ.get(self.endpoint_var) or self.cfg.api_base

        response = await litellm.acompletion(
            model=self.cfg.model,
            messages=oa_messages,
            api_key=api_key,
            base_url=base_url,
            **params,
        )

        # Handle the response safely
        try:
            # Try to access as dictionary first
            if isinstance(response, dict):
                choices = response.get("choices", [])
                choice = choices[0] if choices else None
                message_data = choice.get("message", {}) if choice else {}
            else:
                # Try to access as object
                choice = response.choices[0] if hasattr(response, "choices") else None  # type: ignore
                if choice is None:
                    raise AttributeError("No choices in response")
                message_data = choice.message if hasattr(choice, "message") else {}  # type: ignore

                # Convert to dict if needed
                if not isinstance(message_data, dict):
                    message_data = vars(message_data) if hasattr(message_data, "__dict__") else {}

            # Process tool calls if present
            if isinstance(message_data, dict) and "tool_calls" in message_data:
                for call in message_data["tool_calls"]:
                    if isinstance(call, dict) and "function" in call:
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
            # Process function call if present
            elif isinstance(message_data, dict) and "function_call" in message_data:
                func = message_data["function_call"]
                yield Message(
                    role="assistant",
                    kind="tool_use",
                    content={
                        "name": func.get("name"),
                        "arguments": json.loads(func.get("arguments", "{}")),
                    },
                )
            # Process content if present
            elif isinstance(message_data, dict) and "content" in message_data:
                content = message_data["content"]
                if content:
                    yield Message(role="assistant", kind="text", content=content)
            else:
                # Fallback for unexpected response format
                yield Message(role="assistant", kind="error", content="Could not extract content from response")
        except Exception as e:
            yield Message(role="assistant", kind="error", content=f"Error processing response: {str(e)}")

