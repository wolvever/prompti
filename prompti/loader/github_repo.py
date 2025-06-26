"""Load prompt templates directly from a GitHub repository."""

import base64
import codecs

import httpx
import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader, VersionEntry


class GitHubRepoLoader(TemplateLoader):
    """Fetch prompt files from a GitHub repository."""

    def __init__(self, repo: str, branch: str = "main", token: str | None = None, root: str = "prompts") -> None:
        """Initialize the loader with repository details."""
        self.repo = repo
        self.branch = branch
        self.root = root
        self.headers = {"Authorization": f"token {token}"} if token else {}
        self.client = httpx.AsyncClient()

    async def list_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from GitHub repository.

        For GitHub repo loader, we only have one version per branch.
        """
        path = f"{self.root}/{name}.yaml"
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"

        try:
            resp = await self.client.get(url, params={"ref": self.branch}, headers=self.headers)
            if resp.status_code != 200:
                return []

            data = resp.json()
            text = codecs.decode(base64.b64decode(data["content"]), "utf-8")
            meta = yaml.safe_load(text)
            tags = meta.get("tags", [])

            return [VersionEntry(id=self.branch, tags=list(tags))]
        except (httpx.RequestError, ValueError, KeyError, yaml.YAMLError):
            return []

    async def get_template(self, name: str, version: str) -> PromptTemplate:
        """Get specific version of template from GitHub repository."""
        if version != self.branch:
            raise FileNotFoundError(f"Version {version} not available, only {self.branch} branch is configured")

        path = f"{self.root}/{name}.yaml"
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"
        resp = await self.client.get(url, params={"ref": self.branch}, headers=self.headers)
        if resp.status_code != 200:
            raise FileNotFoundError(f"Template {name} not found")

        data = resp.json()
        text = codecs.decode(base64.b64decode(data["content"]), "utf-8")
        meta = yaml.safe_load(text)
        tmpl = PromptTemplate(
            id=name,
            name=meta.get("name", name),
            description=meta.get("description", ""),
            version=self.branch,
            tags=meta.get("tags", []),
            variants={k: Variant(**v) for k, v in meta.get("variants", {}).items()},
            yaml=text,
        )
        return tmpl
