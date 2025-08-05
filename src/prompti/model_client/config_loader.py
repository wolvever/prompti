"""Load ModelConfig objects from various sources."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import httpx
import yaml

from .base import ModelConfig


class ModelConfigNotFoundError(Exception):
    """Raised when a model configuration is not found."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(f"Model configuration '{model_name}' not found")


class ModelConfigLoader(ABC):
    """Base class for loaders that return a :class:`ModelConfig`."""

    models: List[ModelConfig] = []
    
    def __init__(self, reload_interval: int = 300) -> None:
        self._last_loaded = 0
        self._reload_interval = reload_interval
        self._lock = threading.Lock()

    @abstractmethod
    def _do_load(self):
        """Internal method to perform the actual loading."""
        raise NotImplementedError
    
    def load(self):
        """Load model configurations with automatic reload every 5 minutes."""
        current_time = time.time()
        
        with self._lock:
            if current_time - self._last_loaded >= self._reload_interval:
                self._do_load()
                self._last_loaded = current_time

    @abstractmethod
    def get_model_config(self, model: str, provider: str=None) -> ModelConfig:
        """get model config from memory"""
        raise NotImplementedError


class FileModelConfigLoader(ModelConfigLoader):
    """Load model configurations from a local YAML or JSON file."""

    def __init__(self, path: str | Path = "./configs/models.yaml", reload_interval: int = 300) -> None:
        """Initialize the loader with a path to a local YAML or JSON file."""
        super().__init__(reload_interval)
        self.path = Path(path)
        self.models: List[ModelConfig] = []

    def _do_load(self):
        """Load model configurations from a local YAML or JSON file and store in memory."""
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")

        text = self.path.read_text()
        data = yaml.safe_load(text)

        if not isinstance(data, dict):
            raise ValueError("Config file must contain a mapping")

        # 支持两种格式：单个模型配置或多个模型配置
        if "models" in data:
            # 多模型配置格式
            models_data = data["models"]
            if not isinstance(models_data, list):
                raise ValueError("'models' field must be a list")

            self.models = []
            for model_data in models_data:
                if isinstance(model_data, dict):
                    self.models.append(ModelConfig(**model_data))
        else:
            self.models = []

    def get_model_config(self, model: str, provider: str=None) -> ModelConfig:
        """Get model config from memory by model name."""
        self.load()  # Check if reload is needed
        
        for model_config in self.models:
            if model_config.model == model:
                if provider and model_config.provider != provider:
                    continue
                return model_config

        raise ModelConfigNotFoundError(model)

    def list_models(self) -> List[str]:
        """List all available model names."""
        return [config.model for config in self.models]


class HTTPModelConfigLoader(ModelConfigLoader):
    """Fetch model configurations from an HTTP endpoint returning JSON."""

    def __init__(self, url: str, client: httpx.Client | None = None, registry_api_key: str=None, reload_interval: int = 300) -> None:
        """Initialize the loader with an HTTP endpoint returning JSON."""
        super().__init__(reload_interval)
        self.base_url = url
        self.client = client or httpx.Client(timeout=httpx.Timeout(30))
        self.models: List[ModelConfig] = []
        self.registry_api_key = registry_api_key

    def _do_load(self):
        """Fetch model configurations from an HTTP endpoint and store in memory."""
        try:
            headers = {
                "Authorization": f"Bearer {self.registry_api_key}"
            }
            model_list_url = f"{self.base_url}/model/list"
            model_resp = self.client.get(url=model_list_url, headers=headers)
            model_resp.raise_for_status()
            model_list_data = model_resp.json().get("data") or []

            model_token_url = f"{self.base_url}/llm-token/list"
            token_resp = self.client.get(url=model_token_url, headers=headers)
            token_resp.raise_for_status()
            token_list_data = token_resp.json().get("data") or []
            token_dict = {token["name"]: token for token in token_list_data}
            
            new_models = []
            for model in model_list_data:
                model_config = ModelConfig(
                    provider=model["provider"],
                    model=model["name"],
                    api_url=model["url"],
                )
                if model.get("llm_tokens"):
                    llm_token_name = model.get("llm_tokens")[0]
                    model_config.api_key = token_dict[llm_token_name].get('token_config', {}).get("api_key", "")
                new_models.append(model_config)
            
            self.models = new_models
        except Exception as e:
            print(e)


    def get_model_config(self, model: str, provider: str=None) -> ModelConfig:
        """Get model config from memory by model name."""
        self.load()  # Check if reload is needed
        
        for model_config in self.models:
            if model_config.model == model:
                if provider and model_config.provider != provider:
                    continue
                return model_config

        raise ModelConfigNotFoundError(model)

    def list_models(self) -> List[str]:
        """List all available model names."""
        return [config.model for config in self.models]


class MemoryModelConfigLoader(ModelConfigLoader):
    """Load model configurations from memory using manually provided model list and token list."""

    def __init__(self, model_list: List[dict] = None, token_list: List[dict] = None,
                 reload_interval: int = 300) -> None:
        """Initialize the loader with manually provided model and token lists.

        Args:
            model_list: List of model dictionaries with structure:
                [
                    {
                        "name": "gpt-4o",
                        "provider": "openai",
                        "url": "https://api.openai.com/v1",
                        "llm_tokens": ["openai_token"]
                    }
                ]
            token_list: List of token dictionaries with structure:
                [
                    {
                        "name": "openai_token",
                        "token_config": {
                            "api_key": "sk-..."
                        }
                    }
                ]
            reload_interval: Reload interval in seconds (default: 300)
        """
        super().__init__(reload_interval)
        self.model_list = model_list or []
        self.token_list = token_list or []
        self.models: List[ModelConfig] = []

    def _do_load(self):
        """Load model configurations from memory using provided lists."""
        try:
            # Create token lookup dictionary
            token_dict = {token["name"]: token for token in self.token_list}

            new_models = []
            for model in self.model_list:
                model_config = ModelConfig(
                    provider=model.get("provider", ""),
                    model=model.get("name", ""),
                    api_url=model.get("url", ""),
                )

                # Associate API key from token list if available
                if model.get("llm_tokens"):
                    llm_token_name = model.get("llm_tokens")[0]
                    if llm_token_name in token_dict:
                        token_config = token_dict[llm_token_name].get('token_config', {})
                        model_config.api_key = token_config.get("api_key", "")

                new_models.append(model_config)

            self.models = new_models
        except Exception as e:
            print(f"Error loading model configs from memory: {e}")

    def get_model_config(self, model: str, provider: str = None) -> ModelConfig:
        """Get model config from memory by model name."""
        self.load()  # Check if reload is needed

        for model_config in self.models:
            if model_config.model == model:
                if provider and model_config.provider != provider:
                    continue
                return model_config

        raise ModelConfigNotFoundError(model)

    def list_models(self) -> List[str]:
        """List all available model names."""
        self.load()  # Ensure models are loaded
        return [config.model for config in self.models]