from __future__ import annotations

"""OpenAI client implementation."""

import os

from .openai_base import _OpenAICore


class OpenAIClient(_OpenAICore):
    provider = "openai"
    api_url = "https://api.openai.com/v1/chat/completions"
    api_key_var = "OPENAI_API_KEY"
