from __future__ import annotations

import base64
import codecs

import httpx
import yaml

from ..template import PromptTemplate


class GitHubRepoLoader:
    """Fetch prompt files from a GitHub repository."""

    def __init__(self, repo: str, branch: str = "main", token: str | None = None, root: str = "prompts") -> None:
        self.repo = repo
        self.branch = branch
        self.root = root
        self.headers = {"Authorization": f"token {token}"} if token else {}
        self.client = httpx.AsyncClient()

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        path = f"{self.root}/{name}.yaml"
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"
        resp = await self.client.get(url, params={"ref": self.branch}, headers=self.headers)
        if resp.status_code != 200:
            raise FileNotFoundError(name)
        data = resp.json()
        text = codecs.decode(base64.b64decode(data["content"]), "utf-8")
        meta = yaml.safe_load(text)
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=self.branch,
            labels=meta.get("labels", []),
            yaml=text,
        )
        return tmpl.version, tmpl
