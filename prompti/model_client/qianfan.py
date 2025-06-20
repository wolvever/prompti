from __future__ import annotations

"""OpenAI client implementation."""

import os

from .openai_base import _OpenAICore


class QianfanClient(_OpenAICore):
    """Client for the OpenAI chat completion API."""

    provider = "qianfan"
    api_url = "https://qianfan.baidubce.com/v2/chat/completions"
    api_key_var = "QIANFAN_API_KEY"
