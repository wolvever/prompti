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
from .config_loader import (
    ModelConfigLoader,
    FileModelConfigLoader,
    HTTPModelConfigLoader,
    ModelConfigNotFoundError,
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
    "ModelConfigLoader",
    "FileModelConfigLoader", 
    "HTTPModelConfigLoader",
    "ModelConfigNotFoundError",
]

# Optional import for LiteLLMClient
try:
    from .litellm import LiteLLMClient  # noqa: F401

    __all__.append("LiteLLMClient")
except ImportError:
    pass

# OpenAI clients
try:
    from .openai_client import OpenAIClient  # noqa: F401

    __all__.extend(["OpenAIClient"])
except ImportError:
    pass

try:
    from .qianfan_client import QianFanClient  # noqa: F401
    __all__.extend(["QianFanClient"])
except ImportError:
    pass
