from __future__ import annotations

from typing import Any

import httpx

from .base import ModelConfig
from .claude import ClaudeClient
from .litellm import LiteLLMClient
from .openai import OpenAIClient
from .openrouter import OpenRouterClient
from .rust import RustModelClient


def create_client(cfg: ModelConfig, **httpx_kw: Any):
    """Return a ``ModelClient`` instance for ``cfg.provider``."""
    client = httpx.AsyncClient(http2=True, **httpx_kw) if httpx_kw else None
    if cfg.provider == "openai":
        return OpenAIClient(cfg, client=client)
    if cfg.provider == "claude":
        return ClaudeClient(cfg, client=client)
    if cfg.provider == "openrouter":
        return OpenRouterClient(cfg, client=client)
    if cfg.provider == "litellm":
        return LiteLLMClient(cfg, client=client)
    if cfg.provider == "rust":
        return RustModelClient(cfg, client=client)
    raise ValueError(f"Unsupported provider: {cfg.provider}")
