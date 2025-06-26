"""Load templates stored in a local Git repository."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry


class LocalGitRepoLoader(TemplateLoader):
    """Read prompt files from a local Git repository."""

    def __init__(self, repo_path: Path, ref: str = "HEAD") -> None:
        """Create the loader pointing at ``repo_path`` and ``ref``."""
        import pygit2

        self.repo = pygit2.Repository(str(repo_path))
        self.ref = ref

    async def list_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from local Git repository.

        For local Git repo loader, we only have one version per ref.
        """
        try:
            commit = self.repo.revparse_single(self.ref)
            tree = commit.tree
            blob = tree[f"prompts/{name}.yaml"]
            text = blob.data.decode()
            meta = yaml.safe_load(text)
            tags = meta.get("tags", [])
            version = str(commit.hex[:7])

            return [VersionEntry(id=version, tags=list(tags))]
        except (KeyError, yaml.YAMLError, Exception):
            return []

    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Get specific version of template from local Git repository."""
        commit = self.repo.revparse_single(self.ref)
        commit_version = str(commit.hex[:7])

        if version != commit_version:
            raise TemplateNotFoundError(
                f"Version {version} not available, current commit is {commit_version}"
            )

        try:
            tree = commit.tree
            blob = tree[f"prompts/{name}.yaml"]
            text = blob.data.decode()
        except KeyError as err:
            raise TemplateNotFoundError(f"Template {name} not found") from err

        meta = yaml.safe_load(text)
        tmpl = PromptTemplate(
            id=name,
            name=meta.get("name", name),
            description=meta.get("description", ""),
            version=commit_version,
            tags=meta.get("tags", []),
            variants={k: Variant(**v) for k, v in meta.get("variants", {}).items()},
            yaml=text,
        )
        return tmpl
