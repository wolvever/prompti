"""Rust-based model client implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
from opentelemetry import trace
import os

from ..message import Message
from ..model_client_rs import model_client as rs_client
from .base import ModelClient, ModelConfig, RunParams


class RustModelClient(ModelClient):
    """Client for the Rust-based model client."""

    provider = "rust"

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None, is_debug: bool = False) -> None:
        """Initialize the Rust model client using the native wrapper."""
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_url = cfg.api_url
        self.api_key_var = cfg.api_key_var
        self.api_key = cfg.api_key or os.environ.get(self.api_key_var, "") if self.api_key_var else cfg.api_key
        self._tracer = trace.get_tracer(__name__)
        self._rs_client = rs_client.ModelClient()

    async def _run(
        self,
        p: RunParams,
    ) -> AsyncGenerator[Message, None]:
        """Execute the LLM call using the Rust implementation."""

        # Convert messages to the format expected by Rust
        rust_messages = []
        for msg in p.messages:
            rust_messages.append({"role": msg.role, "content": msg.content, "kind": msg.kind})

        # Prepare the request for Rust
        rust_request = {
            "messages": rust_messages,
            "model": self.cfg.model,
            "provider": self.cfg.provider,
            "stream": p.stream,
        }

        # Flatten optional parameters onto the request so they map directly to
        # the Rust ``ChatRequest`` structure.
        rust_request.update(p.extra_params)
        if p.tool_params:
            if hasattr(p.tool_params, "tools"):
                rust_request["tools"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in p.tool_params.tools]
                if getattr(p.tool_params, "choice", None) is not None:
                    choice = p.tool_params.choice
                    rust_request["tool_choice"] = choice.value if hasattr(choice, "value") else choice
            else:
                rust_request["tools"] = [t.model_dump() if hasattr(t, "model_dump") else t for t in p.tool_params]

        # Pass the request directly to the Rust client
        rust_request["api_key"] = self.api_key
        if self.api_url:
            rust_request["api_base"] = self.api_url

        async for chunk in self._rs_client.chat_stream(rust_request):
            if "content" in chunk:
                yield Message(role="assistant", content=chunk["content"], kind="text")

    async def close(self) -> None:
        """Close the client."""
        await super().close()
