import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

from prompti.model_client.base import ModelConfig
from prompti.model_client.config_loader import (
    FileModelConfigLoader,
    HTTPModelConfigLoader,
    ModelConfigNotFoundError,
)


class TestFileModelConfigLoader:
    def test_load_multiple_models(self):
        """Test loading multiple model configurations from file."""
        # Create temporary file with test data
        yaml_content = """
        models:
          - provider: openai
            model: gpt-4
            api_key: test_key1
            api_url: https://api.openai.com/v1
          - provider: anthropic
            model: claude-3-opus
            api_key: test_key2
            api_url: https://api.anthropic.com/v1
        """
        
        with NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            temp_file.write(yaml_content.encode())
            temp_file_path = temp_file.name
        
        try:
            # Create loader and load configs
            loader = FileModelConfigLoader(temp_file_path)
            loader.load()
            
            # Verify models were loaded correctly
            assert len(loader.models) == 2
            
            # Check first model
            assert loader.models[0].provider == "openai"
            assert loader.models[0].model == "gpt-4"
            assert loader.models[0].api_key == "test_key1"
            assert loader.models[0].api_url == "https://api.openai.com/v1"
            
            # Check second model
            assert loader.models[1].provider == "anthropic"
            assert loader.models[1].model == "claude-3-opus"
            assert loader.models[1].api_key == "test_key2"
            assert loader.models[1].api_url == "https://api.anthropic.com/v1"
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_load_empty_models(self):
        """Test loading file with empty models list."""
        # Create temporary file with test data
        yaml_content = """
        models: []
        """
        
        with NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            temp_file.write(yaml_content.encode())
            temp_file_path = temp_file.name
        
        try:
            # Create loader and load configs
            loader = FileModelConfigLoader(temp_file_path)
            loader.load()
            
            # Verify models list is empty
            assert len(loader.models) == 0
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_load_file_not_found(self):
        """Test error handling when file doesn't exist."""
        # Create loader with non-existent file
        loader = FileModelConfigLoader("/nonexistent/path/to/config.yaml")
        
        # Verify error is raised
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            loader.load()
    
    def test_load_invalid_yaml(self):
        """Test error handling with invalid YAML."""
        # Create temporary file with invalid YAML
        yaml_content = """
        models:
          - provider: openai
            model: gpt-4
          invalid indentation
        """
        
        with NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            temp_file.write(yaml_content.encode())
            temp_file_path = temp_file.name
        
        try:
            # Create loader
            loader = FileModelConfigLoader(temp_file_path)
            
            # Verify error is raised
            with pytest.raises(yaml.YAMLError):
                loader.load()
        finally:
            # Clean up
            os.unlink(temp_file_path)
    
    def test_get_model_config_found(self):
        """Test retrieving model config that exists."""
        # Create loader with mock data
        loader = FileModelConfigLoader()
        loader.models = [
            ModelConfig(provider="openai", model="gpt-4", api_key="test_key"),
            ModelConfig(provider="anthropic", model="claude-3", api_key="test_key2")
        ]
        
        # Retrieve model config
        config = loader.get_model_config("gpt-4")
        
        # Verify
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "test_key"
    
    def test_get_model_config_with_provider(self):
        """Test retrieving model config with provider specified."""
        # Create loader with mock data
        loader = FileModelConfigLoader()
        loader.models = [
            ModelConfig(provider="openai", model="gpt-4", api_key="test_key1"),
            ModelConfig(provider="anthropic", model="gpt-4", api_key="test_key2")  # Same model name, different provider
        ]
        
        # Retrieve model config with provider
        config = loader.get_model_config("gpt-4", provider="anthropic")
        
        # Verify
        assert config.provider == "anthropic"
        assert config.model == "gpt-4"
        assert config.api_key == "test_key2"
    
    def test_get_model_config_not_found(self):
        """Test error when model config isn't found."""
        # Create loader with mock data
        loader = FileModelConfigLoader()
        loader.models = [
            ModelConfig(provider="openai", model="gpt-4", api_key="test_key")
        ]
        
        # Attempt to retrieve non-existent model
        with pytest.raises(ModelConfigNotFoundError, match="Model configuration 'gpt-3.5-turbo' not found"):
            loader.get_model_config("gpt-3.5-turbo")
    
    def test_list_models(self):
        """Test listing available models."""
        # Create loader with mock data
        loader = FileModelConfigLoader()
        loader.models = [
            ModelConfig(provider="openai", model="gpt-4"),
            ModelConfig(provider="anthropic", model="claude-3")
        ]
        
        # List models
        models = loader.list_models()
        
        # Verify
        assert len(models) == 2
        assert "gpt-4" in models
        assert "claude-3" in models


class TestHTTPModelConfigLoader:
    def test_load_successful(self):
        """Test loading model configs from HTTP endpoint."""
        # Setup mock client
        mock_client = MagicMock()
        
        # Set up the mock responses directly in the mock client method
        mock_model_data = [
            {
                "provider": "openai", 
                "name": "gpt-4",
                "url": "https://api.openai.com/v1",
                "llm_tokens": ["openai_token"]
            },
            {
                "provider": "anthropic",
                "name": "claude-3",
                "url": "https://api.anthropic.com/v1",
                "llm_tokens": ["anthropic_token"]
            }
        ]
        
        mock_token_data = [
            {
                "name": "openai_token",
                "token_config": {"api_key": "openai_key"}
            },
            {
                "name": "anthropic_token",
                "token_config": {"api_key": "anthropic_key"}
            }
        ]
        
        # First call to get returns model list
        model_response = MagicMock()
        model_response.status_code = 200
        model_response.json.return_value = mock_model_data
        model_response.raise_for_status.return_value = None
        
        # Second call to get returns token list
        token_response = MagicMock()
        token_response.status_code = 200
        token_response.json.return_value = mock_token_data
        token_response.raise_for_status.return_value = None
        
        mock_client.get.side_effect = [model_response, token_response]
        
        # Create loader and load configs
        loader = HTTPModelConfigLoader("http://example.com/api", client=mock_client)
        loader.load()
        
        # Verify request calls
        assert mock_client.get.call_count == 2
        mock_client.get.assert_any_call("http://example.com/api/model/list")
        mock_client.get.assert_any_call("http://example.com/api/llm-token/list")
        
        # Verify models were loaded
        assert len(loader.models) == 2
        
        # Check first model
        assert loader.models[0].provider == "openai"
        assert loader.models[0].model == "gpt-4"
        assert loader.models[0].api_url == "https://api.openai.com/v1"
        assert loader.models[0].api_key == "openai_key"
        
        # Check second model
        assert loader.models[1].provider == "anthropic"
        assert loader.models[1].model == "claude-3"
        assert loader.models[1].api_url == "https://api.anthropic.com/v1"
        assert loader.models[1].api_key == "anthropic_key"
    
    def test_load_http_error(self):
        """Test error handling when HTTP request fails."""
        # Create mock client that raises exception
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.RequestError("Connection error", request=None)
        
        # Create loader
        loader = HTTPModelConfigLoader("http://example.com/api", client=mock_client)
        
        # Verify error is raised
        with pytest.raises(httpx.RequestError):
            loader.load()
    
    def test_load_http_non_200(self):
        """Test error handling when HTTP response is not 200."""
        # Create mock client
        mock_client = MagicMock()
        
        # Mock response with error status
        error_response = MagicMock()
        error_response.status_code = 404
        
        # Configure raise_for_status to raise an HTTPStatusError
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", 
            request=MagicMock(), 
            response=MagicMock()
        )
        
        mock_client.get.return_value = error_response
        
        # Create loader
        loader = HTTPModelConfigLoader("http://example.com/api", client=mock_client)
        
        # Verify error is raised
        with pytest.raises(httpx.HTTPStatusError):
            loader.load()
    
    def test_get_model_config_found(self):
        """Test retrieving model config that exists."""
        # Create loader with mock data
        loader = HTTPModelConfigLoader("http://example.com/api")
        loader.models = [
            ModelConfig(provider="openai", model="gpt-4", api_key="test_key"),
            ModelConfig(provider="anthropic", model="claude-3", api_key="test_key2")
        ]
        
        # Retrieve model config
        config = loader.get_model_config("gpt-4")
        
        # Verify
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.api_key == "test_key"
    
    def test_get_model_config_not_found(self):
        """Test error when model config isn't found."""
        # Create loader with mock data
        loader = HTTPModelConfigLoader("http://example.com/api")
        loader.models = [
            ModelConfig(provider="openai", model="gpt-4", api_key="test_key")
        ]
        
        # Attempt to retrieve non-existent model
        with pytest.raises(ModelConfigNotFoundError, match="Model configuration 'gpt-3.5-turbo' not found"):
            loader.get_model_config("gpt-3.5-turbo")
    
    def test_list_models(self):
        """Test listing available models."""
        # Create loader with mock data
        loader = HTTPModelConfigLoader("http://example.com/api")
        loader.models = [
            ModelConfig(provider="openai", model="gpt-4"),
            ModelConfig(provider="anthropic", model="claude-3")
        ]
        
        # List models
        models = loader.list_models()
        
        # Verify
        assert len(models) == 2
        assert "gpt-4" in models
        assert "claude-3" in models