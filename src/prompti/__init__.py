"""PromptI: provider-agnostic prompt engine."""

from __future__ import annotations

from .engine import PromptEngine
from .experiment import (
    ExperimentRegistry,
    ExperimentSplit,
    GrowthBookRegistry,
    UnleashRegistry,
    bucket,
)
from .loader import (
    AgentaLoader,
    FileSystemLoader,
    GitHubRepoLoader,
    HTTPLoader,
    LangfuseLoader,
    LocalGitRepoLoader,
    MemoryLoader,
    PezzoLoader,
    PromptLayerLoader,
    TemplateLoader,
    TemplateNotFoundError,
)
from .message import Kind, Message
from .model_client import (
    ModelClient,
    ModelConfig,
    RunParams,
    ToolChoice,
    ToolParams,
    ToolSpec,
    create_client,
)
from .replay import ModelClientRecorder, ReplayEngine
from .template import PromptTemplate

__all__ = [
    "Message",
    "PromptTemplate",
    "PromptEngine",
    "ModelClient",
    "ModelConfig",
    "RunParams",
    "ToolSpec",
    "ToolParams",
    "ToolChoice",
    "create_client",
    "ReplayEngine",
    "ModelClientRecorder",
    "ExperimentRegistry",
    "ExperimentSplit",
    "UnleashRegistry",
    "GrowthBookRegistry",
    "bucket",
    "Kind",
    "TemplateLoader",
    "TemplateNotFoundError",
    "HTTPLoader",
    "FileSystemLoader",
    "MemoryLoader",
    "PromptLayerLoader",
    "LangfuseLoader",
    "PezzoLoader",
    "AgentaLoader",
    "GitHubRepoLoader",
    "LocalGitRepoLoader",
]

# Optional imports - only available when litellm is installed
try:
    from .model_client import LiteLLMClient  # noqa: F401

    __all__.append("LiteLLMClient")
except ImportError:
    # LiteLLMClient not available - litellm dependency not installed
    pass
