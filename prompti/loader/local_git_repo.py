from __future__ import annotations

from pathlib import Path

import yaml

from ..template import PromptTemplate, Variant


class LocalGitRepoLoader:
    """Read prompt files from a local Git repository."""

    def __init__(self, repo_path: Path, ref: str = "HEAD") -> None:
        import pygit2

        self.repo = pygit2.Repository(str(repo_path))
        self.ref = ref

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        commit = self.repo.revparse_single(self.ref)
        tree = commit.tree
        blob = tree[f"prompts/{name}.yaml"]
        text = blob.data.decode()
        meta = yaml.safe_load(text)
        tmpl = PromptTemplate(
            id=name,
            name=meta.get("name", name),
            description=meta.get("description", ""),
            version=str(commit.hex[:7]),
            tags=meta.get("tags", []),
            variants={k: Variant(**v) for k, v in meta.get("variants", {}).items()},
            yaml=text,
        )
        return tmpl.version, tmpl
