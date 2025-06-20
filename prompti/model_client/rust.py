from __future__ import annotations

"""Rust-based model client implementation."""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from opentelemetry import trace
from prometheus_client import Counter, Histogram

from .base import ModelClient, ModelConfig
from ..message import Message


class RustModelClient(ModelClient):
    """Rust-based model client implementation."""

    provider = "rust"

    def __init__(self, rust_binary_path: str | None = None, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the Rust model client.
        
        Args:
            rust_binary_path: Path to the compiled Rust binary. If None, will try to find it.
            client: Optional HTTP client (not used by Rust implementation)
        """
        super().__init__(client)
        self.rust_binary_path = rust_binary_path or self._find_rust_binary()
        self._tracer = trace.get_tracer(__name__)

    def _find_rust_binary(self) -> str:
        """Find the compiled Rust binary."""
        # Look for the binary in the model_client_rs directory
        rust_dir = Path(__file__).parent.parent / "model_client_rs"
        binary_path = rust_dir / "target" / "release" / "model-client-rs"
        
        if not binary_path.exists():
            # Try debug build
            binary_path = rust_dir / "target" / "debug" / "model-client-rs"
            
        if not binary_path.exists():
            raise FileNotFoundError(
                f"Rust binary not found. Please build the Rust project first:\n"
                f"cd {rust_dir} && cargo build --release"
            )
            
        return str(binary_path)

    async def _run(
        self, messages: list[Message], model_cfg: ModelConfig
    ) -> AsyncGenerator[Message, None]:
        """Execute the LLM call using the Rust implementation."""
        
        # Convert messages to the format expected by Rust
        rust_messages = []
        for msg in messages:
            rust_messages.append({
                "role": msg.role,
                "content": msg.content,
                "kind": msg.kind
            })
        
        # Prepare the request for Rust
        rust_request = {
            "messages": rust_messages,
            "model": model_cfg.model,
            "provider": model_cfg.provider,
            "parameters": model_cfg.parameters
        }
        
        # Create temporary file for the request
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(rust_request, f)
            request_file = f.name
        
        try:
            # Execute the Rust binary
            process = await asyncio.create_subprocess_exec(
                self.rust_binary_path,
                "--request-file", request_file,
                "--stream",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "OPENAI_API_KEY": model_cfg.api_key or model_cfg.parameters.get("api_key", ""),
                    "ANTHROPIC_API_KEY": model_cfg.api_key or model_cfg.parameters.get("api_key", ""),
                }
            )
            
            # Read streaming output
            if process.stdout is not None:
                async for line in process.stdout:
                    line = line.decode().strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        if "content" in data:
                            yield Message(
                                role="assistant",
                                content=data["content"],
                                kind="text"
                            )
                    except json.JSONDecodeError:
                        # Skip non-JSON lines (logs, etc.)
                        continue
            
            # Wait for process to complete
            await process.wait()
            
            if process.returncode != 0:
                stderr_content = ""
                if process.stderr is not None:
                    stderr_content = (await process.stderr.read()).decode()
                raise RuntimeError(f"Rust client failed: {stderr_content}")
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(request_file)
            except OSError:
                pass

    async def close(self) -> None:
        """Close the client."""
        await super().close() 