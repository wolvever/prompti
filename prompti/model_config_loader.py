"""Load ModelConfig objects from various sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import httpx
import yaml

from .model_client import ModelConfig


class ModelConfigLoader(ABC):
    """Base class for loaders that return a :class:`ModelConfig`."""

    @abstractmethod
    def load(self) -> ModelConfig:
        """Return a :class:`ModelConfig` instance."""
        raise NotImplementedError


class FileModelConfigLoader(ModelConfigLoader):
    """Load a model configuration from a local YAML or JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ModelConfig:
        text = self.path.read_text()
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a mapping")
        return ModelConfig(**data)


class HTTPModelConfigLoader(ModelConfigLoader):
    """Fetch model configuration from an HTTP endpoint returning JSON."""

    def __init__(self, url: str, client: httpx.Client | None = None) -> None:
        self.url = url
        self.client = client or httpx.Client()

    def load(self) -> ModelConfig:
        resp = self.client.get(self.url)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("Config response must be a JSON object")
        return ModelConfig(**data)
