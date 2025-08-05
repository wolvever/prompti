"""
qianfan client
"""
from .openai_client import OpenAIClient, SyncOpenAIClient


class QianfanClient(OpenAIClient):
    """OpenAI-compatible API client."""

    provider = "qianfan"


class SyncQianfanClient(SyncOpenAIClient):
    """Synchronous OpenAI-compatible API client."""
    provider = "qianfan"