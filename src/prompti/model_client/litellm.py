"""LiteLLM client implementation using the `litellm` package."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from ..message import Message
from .base import ModelClient, ModelConfig, RunParams, ToolChoice, ToolParams, ToolSpec


class LiteLLMClient(ModelClient):
    """Client for the LiteLLM API."""

    provider = "litellm"

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None, is_debug: bool = False) -> None:
        """Instantiate the client with configuration and optional HTTP client."""
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e

        super().__init__(cfg, client, is_debug=is_debug)
        self.api_url = cfg.api_url
        self.api_key_var = cfg.api_key_var
        self.api_key = cfg.api_key or (os.environ.get(self.api_key_var) if self.api_key_var else None)
        self.base_url = self.api_url

    async def _run(  # noqa: C901
        self,
        p: RunParams,
    ) -> AsyncGenerator[Message, None]:
        """Translate A2A messages and execute via :func:`litellm.acompletion`."""
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e
        is_claude = self.cfg.model.startswith("claude")
        oa_messages: list[dict[str, Any]] = []
        claude_msgs: list[dict[str, Any]] = []

        for m in p.messages:
            role = m.role
            if m.kind == "tool_result":
                role = "tool"

            if is_claude:
                blocks: list[dict[str, Any]] = []
                if m.kind == "text":
                    blocks.append({"type": "text", "text": m.content})
                elif m.kind == "thinking":
                    blocks.append({"type": "thinking", "thinking": m.content})
                elif m.kind in ("image", "image_url"):
                    blocks.append({"type": "image", "source": {"type": "url", "url": m.content}})
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
                    claude_msgs.append({"role": role, "content": blocks})
            else:
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

        params = dict(p.extra_params)
        if p.temperature is not None:
            params["temperature"] = p.temperature
        if p.max_tokens is not None:
            params["max_tokens"] = p.max_tokens

        if is_claude:
            if p.tool_params:
                if isinstance(p.tool_params, ToolParams):
                    params["tools"] = [
                        {
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.parameters,
                        }
                        if isinstance(t, ToolSpec)
                        else t
                        for t in p.tool_params.tools
                    ]
                    if p.tool_params.choice == ToolChoice.REQUIRED:
                        params["tool_choice"] = {"type": "any"}
                else:
                    params["tools"] = [
                        {
                            "name": t.name,
                            "description": t.description,
                            "input_schema": t.parameters,
                        }
                        if isinstance(t, ToolSpec)
                        else t
                        for t in p.tool_params
                    ]
        else:
            if p.tool_params:
                if isinstance(p.tool_params, ToolParams):
                    tools = [
                        {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                        for t in p.tool_params.tools
                    ]
                    params["tools"] = tools
                    choice = p.tool_params.choice
                    if isinstance(choice, ToolChoice):
                        if choice is not ToolChoice.AUTO:
                            params["tool_choice"] = choice.value
                    elif choice is not None:
                        params["tool_choice"] = choice
                else:
                    params["tools"] = [
                        {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                        for t in p.tool_params
                    ]

        params["stream"] = p.stream

        messages_param = claude_msgs if is_claude else oa_messages
        response = await litellm.acompletion(
            model=self.cfg.model,
            messages=messages_param,
            api_key=self.api_key,
            base_url=self.base_url,
            **params,
        )

        # Handle streaming response
        if p.stream:
            try:
                # LiteLLM returns an async generator for streaming
                async for chunk in response:
                    if hasattr(chunk, "choices") and chunk.choices:
                        choice = chunk.choices[0]
                        delta = choice.delta if hasattr(choice, "delta") else {}

                        # Handle content delta
                        if hasattr(delta, "content") and delta.content:
                            yield Message(role="assistant", kind="text", content=delta.content)

                        # Handle tool calls delta
                        if hasattr(delta, "tool_calls") and delta.tool_calls:
                            for call in delta.tool_calls:
                                if hasattr(call, "function") and call.function:
                                    func = call.function
                                    if hasattr(func, "name") and func.name:
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
            except Exception as e:
                yield Message(role="assistant", kind="error", content=f"Error processing streaming response: {str(e)}")
        else:
            # Handle non-streaming response (existing logic)
            try:
                # Try to access as dictionary first
                if isinstance(response, dict):
                    if is_claude:
                        blocks = response.get("content", [])
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
                        return
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
