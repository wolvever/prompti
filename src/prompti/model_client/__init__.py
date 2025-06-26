"""Model clients for various providers."""

from __future__ import annotations

from ..message import Message
from .base import (
    ModelClient,
    ModelConfig,
    RunParams,
    ToolChoice,
    ToolParams,
    ToolSpec,
)
from .factory import create_client

__all__ = [
    "ModelConfig",
    "ModelClient",
    "RunParams",
    "ToolSpec",
    "ToolParams",
    "ToolChoice",
    "create_client",
    "Message",
]

# Optional import for LiteLLMClient
try:
    from .litellm import LiteLLMClient

    __all__.append("LiteLLMClient")
except ImportError:
    pass
