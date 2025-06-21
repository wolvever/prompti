"""OpenAI client implementation."""

from __future__ import annotations

import os

from .openai_base import _OpenAICore


class OpenAIClient(_OpenAICore):
    """Client for the OpenAI chat completion API."""

    provider = "openai"
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key_var = "OPENAI_API_KEY"
