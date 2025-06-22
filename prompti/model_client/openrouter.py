"""OpenRouter client implementation."""

from __future__ import annotations

from .openai import OpenAIClient


class OpenRouterClient(OpenAIClient):
    """Client for the OpenRouter API."""

    provider = "openrouter"
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    api_key_var = "OPENROUTER_API_KEY"
