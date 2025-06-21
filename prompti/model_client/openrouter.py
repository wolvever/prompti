"""OpenRouter client implementation."""

from __future__ import annotations

import os

from .openai_base import _OpenAICore


class OpenRouterClient(_OpenAICore):
    """Client for the OpenRouter API."""

    provider = "openrouter"
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    api_key_var = "OPENROUTER_API_KEY"
