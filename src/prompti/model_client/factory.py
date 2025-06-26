"""Factory for constructing model client implementations."""

from __future__ import annotations

from typing import Any

import httpx

from .base import ModelConfig


def create_client(cfg: ModelConfig, *, is_debug: bool = False, **httpx_kw: Any):
    """Return a ``ModelClient`` instance for ``cfg.provider``."""
    client = httpx.AsyncClient(http2=True, **httpx_kw) if httpx_kw else None
    if cfg.provider == "litellm":
        try:
            from .litellm import LiteLLMClient
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e
        return LiteLLMClient(cfg, client=client, is_debug=is_debug)
    raise ValueError(f"Unsupported provider: {cfg.provider}")
