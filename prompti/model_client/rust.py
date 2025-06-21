"""Rust-based model client implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
from opentelemetry import trace

from ..message import Message
from ..model_client_rs import model_client as rs_client
from .base import ModelClient, ModelConfig


class RustModelClient(ModelClient):
    """Rust-based model client implementation."""

    provider = "rust"

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the Rust model client using the native wrapper."""
        super().__init__(client)
        self._tracer = trace.get_tracer(__name__)
        self._rs_client = rs_client.ModelClient()

    async def _run(
        self,
        messages: list[Message],
        model_cfg: ModelConfig,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[Message, None]:
        """Execute the LLM call using the Rust implementation."""

        # Convert messages to the format expected by Rust
        rust_messages = []
        for msg in messages:
            rust_messages.append({"role": msg.role, "content": msg.content, "kind": msg.kind})

        # Prepare the request for Rust
        rust_request = {
            "messages": rust_messages,
            "model": model_cfg.model,
            "provider": model_cfg.provider,
        }

        # Flatten optional parameters onto the request so they map directly to
        # the Rust ``ChatRequest`` structure.
        rust_request.update(model_cfg.parameters)

        if tools is not None:
            rust_request["tools"] = tools

        # Pass the request directly to the Rust client
        rust_request["api_key"] = model_cfg.api_key or model_cfg.parameters.get("api_key")
        if model_cfg.api_base:
            rust_request["api_base"] = model_cfg.api_base

        async for chunk in self._rs_client.chat_stream(rust_request):
            if "content" in chunk:
                yield Message(role="assistant", content=chunk["content"], kind="text")

    async def close(self) -> None:
        """Close the client."""
        await super().close()
