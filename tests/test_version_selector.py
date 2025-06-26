"""Tests for version selector functionality in TemplateLoader."""

import pytest

from prompti.loader.base import TemplateLoader, VersionEntry
from prompti.loader.memory import MemoryLoader


class TestVersionEntry:
    """Test VersionEntry model."""

    def test_version_entry_creation(self):
        """Test creating a VersionEntry."""
        entry = VersionEntry(id="1.0.0", tags=["prod", "stable"])
        assert entry.id == "1.0.0"
        assert entry.tags == ["prod", "stable"]

    def test_version_entry_default_tags(self):
        """Test VersionEntry with default empty tags."""
        entry = VersionEntry(id="1.0.0")
        assert entry.id == "1.0.0"
        assert entry.tags == []


class TestVersionSelectorParsing:
    """Test version selector parsing static method."""

    def test_parse_simple_version(self):
        """Test parsing simple version without tags."""
        version_spec, tags = TemplateLoader._parse_version_selector("1.0.0")
        assert version_spec == "1.0.0"
        assert tags == []

    def test_parse_version_with_single_tag(self):
        """Test parsing version with single tag."""
        version_spec, tags = TemplateLoader._parse_version_selector("1.0.0#prod")
        assert version_spec == "1.0.0"
        assert tags == ["prod"]

    def test_parse_version_with_multiple_tags(self):
        """Test parsing version with multiple tags."""
        version_spec, tags = TemplateLoader._parse_version_selector("1.0.0#prod+stable+release")
        assert version_spec == "1.0.0"
        assert tags == ["prod", "stable", "release"]

    def test_parse_wildcard_version(self):
        """Test parsing wildcard version."""
        version_spec, tags = TemplateLoader._parse_version_selector("1.x#beta")
        assert version_spec == "1.x"
        assert tags == ["beta"]

    def test_parse_range_version(self):
        """Test parsing range version."""
        version_spec, tags = TemplateLoader._parse_version_selector(">=1.2.0,<1.5.0")
        assert version_spec == ">=1.2.0,<1.5.0"
        assert tags == []

    def test_parse_empty_selector(self):
        """Test parsing empty selector raises error."""
        with pytest.raises(ValueError, match="Version selector cannot be empty"):
            TemplateLoader._parse_version_selector("")

    def test_parse_empty_version_spec(self):
        """Test parsing selector with empty version spec raises error."""
        with pytest.raises(ValueError, match="Version specification cannot be empty"):
            TemplateLoader._parse_version_selector("#prod")

    def test_parse_tags_with_whitespace(self):
        """Test parsing tags with extra whitespace."""
        version_spec, tags = TemplateLoader._parse_version_selector("1.0.0# prod + stable + ")
        assert version_spec == "1.0.0"
        assert tags == ["prod", "stable"]


class TestVersionSelectionStatic:
    """Test static version selection method directly."""

    def test_select_version_empty_list(self):
        """Test selecting from empty version list."""
        versions = []
        selected = TemplateLoader.select_version(versions, "1.0.0")
        assert selected is None

    def test_select_exact_version_match(self):
        """Test exact version matching."""
        versions = [
            VersionEntry(id="1.0.0", tags=["prod"]),
            VersionEntry(id="1.1.0", tags=["beta"]),
            VersionEntry(id="2.0.0", tags=["prod"]),
        ]

        selected = TemplateLoader.select_version(versions, "1.1.0")
        assert selected is not None
        assert selected.id == "1.1.0"

    def test_select_exact_version_with_tags(self):
        """Test exact version matching with required tags."""
        versions = [
            VersionEntry(id="1.0.0", tags=["prod", "stable"]),
            VersionEntry(id="1.1.0", tags=["beta"]),
            VersionEntry(id="2.0.0", tags=["prod"]),
        ]

        selected = TemplateLoader.select_version(versions, "1.0.0#prod")
        assert selected is not None
        assert selected.id == "1.0.0"

        # Should not match if required tag is missing
        selected = TemplateLoader.select_version(versions, "1.1.0#prod")
        assert selected is None

    def test_select_wildcard_major_version(self):
        """Test wildcard major version selection (1.x)."""
        versions = [
            VersionEntry(id="1.0.0", tags=["prod"]),
            VersionEntry(id="1.1.0", tags=["beta"]),
            VersionEntry(id="1.2.5", tags=["prod"]),
            VersionEntry(id="2.0.0", tags=["prod"]),
        ]

        selected = TemplateLoader.select_version(versions, "1.x")
        assert selected is not None
        assert selected.id == "1.2.5"  # Should select highest 1.x version

    def test_select_wildcard_minor_version(self):
        """Test wildcard minor version selection (1.2.x)."""
        versions = [
            VersionEntry(id="1.2.0", tags=["prod"]),
            VersionEntry(id="1.2.1", tags=["beta"]),
            VersionEntry(id="1.2.5", tags=["prod"]),
            VersionEntry(id="1.3.0", tags=["prod"]),
        ]

        selected = TemplateLoader.select_version(versions, "1.2.x")
        assert selected is not None
        assert selected.id == "1.2.5"  # Should select highest 1.2.x version

    def test_select_wildcard_with_tags(self):
        """Test wildcard version selection with required tags."""
        versions = [
            VersionEntry(id="1.0.0", tags=["prod"]),
            VersionEntry(id="1.1.0", tags=["beta"]),
            VersionEntry(id="1.2.0", tags=["prod", "stable"]),
        ]

        selected = TemplateLoader.select_version(versions, "1.x#prod")
        assert selected is not None
        assert selected.id == "1.2.0"  # Should select highest 1.x with prod tag

    def test_select_version_range(self):
        """Test version range selection."""
        versions = [
            VersionEntry(id="1.1.0", tags=["prod"]),
            VersionEntry(id="1.2.0", tags=["beta"]),
            VersionEntry(id="1.4.0", tags=["prod"]),
            VersionEntry(id="1.5.0", tags=["prod"]),
            VersionEntry(id="2.0.0", tags=["prod"]),
        ]

        selected = TemplateLoader.select_version(versions, ">=1.2.0,<1.5.0")
        assert selected is not None
        assert selected.id == "1.4.0"  # Should select highest in range

    def test_select_version_range_with_tags(self):
        """Test version range selection with required tags."""
        versions = [
            VersionEntry(id="1.2.0", tags=["beta"]),
            VersionEntry(id="1.3.0", tags=["prod"]),
            VersionEntry(id="1.4.0", tags=["beta"]),
        ]

        selected = TemplateLoader.select_version(versions, ">=1.2.0,<1.5.0#prod")
        assert selected is not None
        assert selected.id == "1.3.0"

    def test_select_no_matching_version(self):
        """Test when no version matches the criteria."""
        versions = [
            VersionEntry(id="1.0.0", tags=["prod"]),
            VersionEntry(id="2.0.0", tags=["beta"]),
        ]

        # No version 1.5.0
        selected = TemplateLoader.select_version(versions, "1.5.0")
        assert selected is None

        # No version with required tags
        selected = TemplateLoader.select_version(versions, "1.0.0#beta")
        assert selected is None

        # No version in range
        selected = TemplateLoader.select_version(versions, ">=3.0.0")
        assert selected is None

    def test_select_complex_range_specifications(self):
        """Test complex range specifications."""
        versions = [
            VersionEntry(id="1.0.0", tags=["prod"]),
            VersionEntry(id="1.1.0", tags=["prod"]),
            VersionEntry(id="1.2.0", tags=["prod"]),
            VersionEntry(id="1.3.0", tags=["prod"]),
            VersionEntry(id="2.0.0", tags=["prod"]),
        ]

        # Test different range formats
        selected = TemplateLoader.select_version(versions, ">=1.1.0")
        assert selected is not None
        assert selected.id == "2.0.0"  # Highest version >= 1.1.0

        selected = TemplateLoader.select_version(versions, "<1.2.0")
        assert selected is not None
        assert selected.id == "1.1.0"  # Highest version < 1.2.0

        selected = TemplateLoader.select_version(versions, ">1.0.0,<=1.2.0")
        assert selected is not None
        assert selected.id == "1.2.0"  # Highest version in range

    def test_select_non_semantic_versions(self):
        """Test selection with non-semantic version strings."""
        versions = [
            VersionEntry(id="v1.0", tags=["prod"]),
            VersionEntry(id="v1.1", tags=["prod"]),
            VersionEntry(id="v1.2", tags=["prod"]),
            VersionEntry(id="v2.0", tags=["prod"]),
        ]

        # Exact match should work
        selected = TemplateLoader.select_version(versions, "v1.1")
        assert selected is not None
        assert selected.id == "v1.1"

        # Wildcard should work with string matching
        selected = TemplateLoader.select_version(versions, "v1.x")
        assert selected is not None
        # Should select the highest v1.x version (v1.2)
        assert selected.id == "v1.2"

    def test_select_mixed_semantic_and_non_semantic(self):
        """Test selection with mixed semantic and non-semantic versions."""
        versions = [
            VersionEntry(id="1.0.0", tags=["prod"]),
            VersionEntry(id="1.1.0", tags=["prod"]),
            VersionEntry(id="v1.2", tags=["prod"]),
            VersionEntry(id="2.0.0", tags=["prod"]),
        ]

        # Should work with semantic versions
        selected = TemplateLoader.select_version(versions, "1.x")
        assert selected is not None
        assert selected.id == "1.1.0"  # Highest semantic 1.x version

        # Range should work with semantic versions only
        selected = TemplateLoader.select_version(versions, ">=1.0.0,<2.0.0")
        assert selected is not None
        assert selected.id == "1.1.0"  # v1.2 is ignored as it's not semantic

    def test_select_invalid_wildcard(self):
        """Test invalid wildcard patterns."""
        versions = [VersionEntry(id="1.0.0", tags=["prod"])]

        # Empty prefix
        selected = TemplateLoader.select_version(versions, ".x")
        assert selected is None

        # No .x suffix
        selected = TemplateLoader.select_version(versions, "1.0")
        assert selected is None  # Should be treated as exact match

    def test_select_invalid_range(self):
        """Test invalid range patterns."""
        versions = [VersionEntry(id="1.0.0", tags=["prod"])]

        # Invalid range syntax
        selected = TemplateLoader.select_version(versions, ">=invalid")
        assert selected is None

    def test_select_multiple_tags(self):
        """Test selection with multiple required tags."""
        versions = [
            VersionEntry(id="1.0.0", tags=["prod"]),
            VersionEntry(id="1.1.0", tags=["prod", "stable"]),
            VersionEntry(id="1.2.0", tags=["prod", "stable", "lts"]),
        ]

        # Single tag
        selected = TemplateLoader.select_version(versions, "1.x#prod")
        assert selected is not None
        assert selected.id == "1.2.0"  # Highest with prod tag

        # Multiple tags
        selected = TemplateLoader.select_version(versions, "1.x#prod+stable")
        assert selected is not None
        assert selected.id == "1.2.0"  # Highest with both tags

        # All three tags
        selected = TemplateLoader.select_version(versions, "1.x#prod+stable+lts")
        assert selected is not None
        assert selected.id == "1.2.0"  # Only version with all three tags

        # Impossible combination
        selected = TemplateLoader.select_version(versions, "1.x#prod+stable+lts+nonexistent")
        assert selected is None


class TestVersionSelectorFormats:
    """Test all supported version selector formats from the specification."""

    @pytest.fixture
    def versions(self):
        """Sample versions for testing."""
        return [
            VersionEntry(id="1.0.0", tags=["prod"]),
            VersionEntry(id="1.1.0", tags=["beta"]),
            VersionEntry(id="1.2.3", tags=["prod", "stable"]),
            VersionEntry(id="1.3.0", tags=["exp_a"]),
            VersionEntry(id="2.0.0", tags=["prod"]),
            VersionEntry(id="2.1.0", tags=["beta"]),
        ]

    def test_format_main_version(self, versions):
        """Test: my-tpl@1.x (main version)."""
        selected = TemplateLoader.select_version(versions, "1.x")
        assert selected is not None
        assert selected.id == "1.3.0"  # Highest 1.x version

    def test_format_main_version_with_tag(self, versions):
        """Test: my-tpl@1.x#prod (main version with tag)."""
        selected = TemplateLoader.select_version(versions, "1.x#prod")
        assert selected is not None
        assert selected.id == "1.2.3"  # Highest 1.x version with prod tag

    def test_format_main_version_with_multiple_tags(self, versions):
        """Test: my-tpl@1.x#prod+stable (main version with multiple tags)."""
        selected = TemplateLoader.select_version(versions, "1.x#prod+stable")
        assert selected is not None
        assert selected.id == "1.2.3"  # Only version with both prod and stable tags

    def test_format_secondary_version(self, versions):
        """Test: my-tpl@1.2.x (secondary version)."""
        selected = TemplateLoader.select_version(versions, "1.2.x")
        assert selected is not None
        assert selected.id == "1.2.3"  # Only 1.2.x version

    def test_format_secondary_version_with_tag(self, versions):
        """Test: my-tpl@1.2.x#beta (secondary version with tag)."""
        # No 1.2.x version has beta tag
        selected = TemplateLoader.select_version(versions, "1.2.x#beta")
        assert selected is None

    def test_format_version_range(self, versions):
        """Test: my-tpl@>=1.2.0,<1.5.0 (version range)."""
        selected = TemplateLoader.select_version(versions, ">=1.2.0,<1.5.0")
        assert selected is not None
        assert selected.id == "1.3.0"  # Highest version in range


class TestMemoryLoaderIntegration:
    """Test MemoryLoader with version selection."""

    @pytest.mark.asyncio
    async def test_memory_loader_list_versions(self):
        """Test MemoryLoader list_versions method."""
        mapping = {
            "test_template": {
                "yaml": """
name: test_template
version: 1.0.0
tags: [prod, stable]
variants:
  default:
    model_config: {provider: litellm, model: gpt-4}
    messages: []
"""
            }
        }
        loader = MemoryLoader(mapping)

        versions = await loader.list_versions("test_template")
        assert len(versions) == 1
        assert versions[0].id == "1.0.0"
        assert versions[0].tags == ["prod", "stable"]

    @pytest.mark.asyncio
    async def test_memory_loader_get_template(self):
        """Test MemoryLoader get_template method."""
        mapping = {
            "test_template": {
                "yaml": """
name: test_template
version: 1.0.0
tags: [prod]
variants:
  default:
    model_config: {provider: litellm, model: gpt-4}
    messages: []
"""
            }
        }
        loader = MemoryLoader(mapping)

        template = await loader.get_template("test_template", "1.0.0")
        assert template.version == "1.0.0"
        assert template.tags == ["prod"]

    @pytest.mark.asyncio
    async def test_memory_loader_version_not_found(self):
        """Test MemoryLoader when version doesn't match."""
        mapping = {
            "test_template": {
                "yaml": """
name: test_template
version: 1.0.0
variants:
  default:
    model_config: {provider: litellm, model: gpt-4}
    messages: []
"""
            }
        }
        loader = MemoryLoader(mapping)

        with pytest.raises(FileNotFoundError):
            await loader.get_template("test_template", "2.0.0")

    @pytest.mark.asyncio
    async def test_memory_loader_load_integration(self):
        """Test MemoryLoader load method with version selector."""
        mapping = {
            "test_template": {
                "yaml": """
name: test_template
version: 1.0.0
tags: [prod, stable]
variants:
  default:
    model_config: {provider: litellm, model: gpt-4}
    messages: []
"""
            }
        }
        loader = MemoryLoader(mapping)

        # Test loading with exact version
        template = await loader.load("test_template", "1.0.0")
        assert template.version == "1.0.0"

        # Test loading with tags
        template = await loader.load("test_template", "1.0.0#prod")
        assert template.version == "1.0.0"

        # Test loading with missing tag should fail
        with pytest.raises(ValueError):
            await loader.load("test_template", "1.0.0#beta")
