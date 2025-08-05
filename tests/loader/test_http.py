import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from prompti.loader import HTTPLoader, TemplateNotFoundError
from prompti.template import PromptTemplate


@pytest.mark.asyncio
async def test_http_loader_list_versions_success():
    """Test listing versions with successful HTTP response."""
    # Mock response data
    mock_response = httpx.Response(
        200, 
        json=[
            {"version": "1.0", "aliases": ["latest", "prod"]},
            {"version": "0.9", "aliases": ["stable"]}
        ]
    )
    
    # Create mock client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    # Create loader with mock client
    loader = HTTPLoader("http://example.com/api", client=mock_client)
    
    # Call method
    versions = await loader.list_versions("test_template")
    
    # Verify
    mock_client.get.assert_called_once_with("http://example.com/api/template/test_template/versions")
    assert len(versions) == 2
    assert versions[0].id == "1.0"
    assert versions[0].aliases == ["latest", "prod"]
    assert versions[1].id == "0.9"
    assert versions[1].aliases == ["stable"]


@pytest.mark.asyncio
async def test_http_loader_list_versions_error():
    """Test listing versions with HTTP error."""
    # Create mock client
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.RequestError("Connection error", request=None)
    
    # Create loader with mock client
    loader = HTTPLoader("http://example.com/api", client=mock_client)
    
    # Call method
    versions = await loader.list_versions("test_template")
    
    # Verify
    assert versions == []


@pytest.mark.asyncio
async def test_http_loader_list_versions_non_200():
    """Test listing versions with non-200 HTTP response."""
    # Mock response data
    mock_response = httpx.Response(404)
    
    # Create mock client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    # Create loader with mock client
    loader = HTTPLoader("http://example.com/api", client=mock_client)
    
    # Call method
    versions = await loader.list_versions("test_template")
    
    # Verify
    assert versions == []


@pytest.mark.asyncio
async def test_http_loader_get_template_success():
    """Test getting template with successful HTTP response."""
    # Mock response data
    mock_response = httpx.Response(
        200, 
        json={
            "data": {
                "id": "test_template",
                "name": "test_template",
                "version": "1.0",
                "aliases": ["latest", "prod"],
                "variants": [
                    {
                        "model_config": {"provider": "openai", "model": "gpt-4"},
                        "messages_template": [
                            {"role": "user", "content": "Hello {{name}}"}
                        ],
                        "required_variables": ["name"]
                    }
                ]
            }
        }
    )
    
    # Create mock client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    # Create loader with mock client
    loader = HTTPLoader("http://example.com/api", client=mock_client)
    
    # Call method
    template = await loader.get_template("test_template", "1.0")
    
    # Verify
    mock_client.get.assert_called_once_with("http://example.com/api/template/test_template?label=1.0")
    assert isinstance(template, PromptTemplate)
    assert template.id == "test_template"
    assert template.name == "test_template"
    assert template.version == "1.0"
    assert template.aliases == ["latest", "prod"]
    assert "default" in template.variants


@pytest.mark.asyncio
async def test_http_loader_get_template_no_version():
    """Test getting template with no version specified."""
    # Mock response data
    mock_response = httpx.Response(
        200, 
        json={
            "data": {
                "id": "test_template",
                "name": "test_template",
                "version": "1.0",
                "variants": [
                    {
                        "model_config": {"provider": "openai", "model": "gpt-4"},
                        "messages_template": [
                            {"role": "user", "content": "Hello {{name}}"}
                        ]
                    }
                ]
            }
        }
    )
    
    # Create mock client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    # Create loader with mock client
    loader = HTTPLoader("http://example.com/api", client=mock_client)
    
    # Call method
    template = await loader.get_template("test_template", "")
    
    # Verify
    mock_client.get.assert_called_once_with("http://example.com/api/template/test_template")
    assert isinstance(template, PromptTemplate)
    assert template.name == "test_template"
    assert template.version == "1.0"


@pytest.mark.asyncio
async def test_http_loader_get_template_not_found():
    """Test getting template with 404 response."""
    # Mock response data
    mock_response = httpx.Response(404)
    
    # Create mock client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    
    # Create loader with mock client
    loader = HTTPLoader("http://example.com/api", client=mock_client)
    
    # Call method and verify exception
    with pytest.raises(TemplateNotFoundError, match="Template test_template version 1.0 not found"):
        await loader.get_template("test_template", "1.0")


@pytest.mark.asyncio
async def test_http_loader_base_url_normalization():
    """Test that base URL is normalized correctly."""
    # Create mock client
    mock_client = AsyncMock()
    mock_client.get.return_value = httpx.Response(200, json=[])
    
    # Create loader with trailing slash in URL
    loader = HTTPLoader("http://example.com/api/", client=mock_client)
    
    # Call method
    await loader.list_versions("test_template")
    
    # Verify the trailing slash was removed
    mock_client.get.assert_called_once_with("http://example.com/api/template/test_template/versions")


@pytest.mark.asyncio
async def test_http_loader_integration_with_mock_server():
    """Test HTTP loader with a mock server."""
    # Prepare test data
    test_data = [
        {
            "request": {"method": "GET", "path": "/template/test_template/versions"},
            "response": [
                {"version": "1.0", "aliases": ["latest"]}
            ]
        },
        {
            "request": {"method": "GET", "path": "/template/test_template?label=1.0"},
            "response": {
                "data": {
                    "id": "test_template",
                    "name": "test_template",
                    "version": "1.0",
                    "aliases": ["latest"],
                    "variants": [
                        {
                            "model_config": {"provider": "openai", "model": "gpt-4"},
                            "messages_template": [
                                {"role": "user", "content": "Hello {{name}}"}
                            ]
                        }
                    ]
                }
            }
        }
    ]
    
    # Create an actual HTTP client
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Configure the mock client to return appropriate responses
        async def mock_get(url):
            if "/versions" in url:
                return httpx.Response(200, json=test_data[0]["response"])
            else:
                return httpx.Response(200, json=test_data[1]["response"])
                
        mock_client.get.side_effect = mock_get
        
        # Create the loader
        loader = HTTPLoader("http://example.com/api")
        
        # Test list_versions
        versions = await loader.list_versions("test_template")
        assert len(versions) == 1
        assert versions[0].id == "1.0"
        
        # Test get_template
        template = await loader.get_template("test_template", "1.0")
        assert isinstance(template, PromptTemplate)
        assert template.name == "test_template"
        assert template.version == "1.0"