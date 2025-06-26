"""Abstract template loader interface and version utilities."""

from abc import ABC, abstractmethod

import semantic_version
from pydantic import BaseModel, Field

from ..template import PromptTemplate


class TemplateNotFoundError(Exception):
    """Raised when a template cannot be located by a loader."""




class VersionEntry(BaseModel):
    """Represents a version of a template with its ID and tags."""

    id: str
    tags: list[str] = Field(default_factory=list)


class TemplateLoader(ABC):
    """Base class; resolves <name>@<range>#tag+tag → PromptTemplate."""

    @abstractmethod
    async def list_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions of a template.

        Parameters
        ----------
        name: str
            Template name to list versions for.

        Returns
        -------
        List[VersionEntry]
            List of available versions with their IDs and tags.

        """
        raise NotImplementedError

    @abstractmethod
    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Get a specific version of a template.

        Parameters
        ----------
        name: str
            Template name to load.
        version: str
            Specific version ID to load.

        Returns
        -------
        PromptTemplate
            The template instance.

        """
        raise NotImplementedError

    async def load(self, name: str, version_selector: str) -> PromptTemplate:
        """Load a template using a version selector.

        Supports the following selector formats:
        - name@1.x                  # main version
        - name@1.x#prod             # main version with tag
        - name@1.x#prod+exp_a       # main version with multiple tags
        - name@1.2.x               # secondary version
        - name@1.2.x#beta          # secondary version with tag
        - name@>=1.2.0,<1.5.0      # version range

        Parameters
        ----------
        name: str
            Template name to load.
        version_selector: str
            Version selector string specifying which version to load.

        Returns
        -------
        PromptTemplate
            The resolved template instance.

        """
        # Get all available versions
        versions = await self.list_versions(name)

        # Find matching version using static method
        selected_version = self.select_version(versions, version_selector)

        if not selected_version:
            raise ValueError(f"No version found matching selector: {version_selector}")

        # Load the selected version
        return await self.get_template(name, selected_version.id)

    @staticmethod
    def select_version(versions: list[VersionEntry], version_selector: str) -> VersionEntry | None:
        """Select the best matching version from available versions.

        Parameters
        ----------
        versions: List[VersionEntry]
            Available versions to select from.
        version_selector: str
            Version selector string (e.g., "1.x", "1.2.x#prod", ">=1.2.0,<1.5.0").

        Returns
        -------
        VersionEntry | None
            The best matching version, or None if no match found.

        """
        if not versions:
            return None

        # Parse the version selector
        version_spec, required_tags = TemplateLoader._parse_version_selector(version_selector)

        # Filter versions that have all required tags
        if required_tags:
            candidates = [v for v in versions if all(tag in v.tags for tag in required_tags)]
        else:
            candidates = versions

        if not candidates:
            return None

        # Handle different version spec formats
        if TemplateLoader._is_version_range(version_spec):
            return TemplateLoader._select_from_range(candidates, version_spec)
        elif version_spec.endswith(".x"):
            return TemplateLoader._select_from_wildcard(candidates, version_spec)
        else:
            # Exact version match - return highest matching version
            matching_candidates = [c for c in candidates if c.id == version_spec]
            if matching_candidates:
                return matching_candidates[0]  # Exact match, just return first one
            return None

    @staticmethod
    def _parse_version_selector(selector: str) -> tuple[str, list[str]]:
        """Parse a version selector into version spec and required tags.

        Parameters
        ----------
        selector: str
            Version selector string to parse.

        Returns
        -------
        tuple[str, List[str]]
            Version specification and list of required tags.

        """
        if not selector:
            raise ValueError("Version selector cannot be empty")

        # Split on # to separate version from tags
        if "#" in selector:
            version_spec, tags_part = selector.split("#", 1)
            # Split tags on +
            required_tags = [tag.strip() for tag in tags_part.split("+") if tag.strip()]
        else:
            version_spec = selector
            required_tags = []

        version_spec = version_spec.strip()
        if not version_spec:
            raise ValueError("Version specification cannot be empty")

        return version_spec, required_tags

    @staticmethod
    def _is_version_range(version_spec: str) -> bool:
        """Check if version spec is a range (contains >= or < operators)."""
        return any(op in version_spec for op in [">=", "<=", ">", "<"])

    @staticmethod
    def _select_from_range(candidates: list[VersionEntry], version_spec: str) -> VersionEntry | None:
        """Select version from a range specification like '>=1.2.0,<1.5.0'."""
        try:
            spec = semantic_version.SimpleSpec(version_spec)
            valid_candidates = []

            for candidate in candidates:
                try:
                    version = semantic_version.Version(candidate.id)
                    if version in spec:
                        valid_candidates.append((version, candidate))
                except ValueError:
                    # Skip versions that don't parse as semantic versions
                    continue

            if valid_candidates:
                # Return the highest version that matches
                valid_candidates.sort(key=lambda x: x[0], reverse=True)
                return valid_candidates[0][1]

        except ValueError:
            # Invalid version spec
            pass

        return None

    @staticmethod
    def _select_from_wildcard(candidates: list[VersionEntry], version_spec: str) -> VersionEntry | None:
        """Select version from wildcard specification like '1.x' or '1.2.x'."""
        if not version_spec.endswith(".x") or not (prefix := version_spec[:-2]):
            return None

        matching_candidates = [
            (TemplateLoader._parse_version_for_sorting(candidate.id), candidate)
            for candidate in candidates
            if TemplateLoader._matches_wildcard_prefix(candidate.id, prefix)
        ]

        if not matching_candidates:
            return None

        # Sort and return the highest version
        matching_candidates.sort(key=lambda x: x[0], reverse=True)
        return matching_candidates[0][1]

    @staticmethod
    def _matches_wildcard_prefix(version_id: str, prefix: str) -> bool:
        """Check if version ID matches the wildcard prefix."""
        # Try semantic version matching first
        try:
            version = semantic_version.Version(version_id)
            version_parts = str(version).split(".")
            prefix_parts = prefix.split(".")
            return len(prefix_parts) <= len(version_parts) and all(
                version_parts[i] == prefix_parts[i] for i in range(len(prefix_parts))
            )
        except ValueError:
            # Fall back to string prefix matching
            return version_id.startswith(prefix + ".") or version_id == prefix

    @staticmethod
    def _parse_version_for_sorting(version_id: str):
        """Parse version ID into a sortable key."""
        # Try semantic version first
        try:
            return semantic_version.Version(version_id)
        except ValueError:
            pass

        # Try parsing as numeric tuple
        try:
            parts = version_id.split(".")
            return tuple(int(p) for p in parts if p.isdigit())
        except (ValueError, TypeError):
            # Fall back to string sorting
            return version_id
