from __future__ import annotations

"""OpenRouter client implementation."""

import os

from .openai_base import _OpenAICore


class OpenRouterClient(_OpenAICore):
    provider = "openrouter"
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    api_key_var = "OPENROUTER_API_KEY"
