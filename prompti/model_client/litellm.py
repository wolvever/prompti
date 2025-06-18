from __future__ import annotations

"""LiteLLM client implementation."""

import os

from .openai_base import _OpenAICore


class LiteLLMClient(_OpenAICore):
    provider = "litellm"
    api_url = os.environ.get("LITELLM_ENDPOINT", "http://localhost:4000/v1/chat/completions")
    api_key_var = "LITELLM_API_KEY"
