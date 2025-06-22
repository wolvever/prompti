"""Template loaders."""

from __future__ import annotations

from pathlib import Path

import httpx
import yaml

from .template import PromptTemplate


class FileSystemLoader:
    """Loader that reads templates from the local filesystem."""

    def __init__(self, base: Path) -> None:
        """Create loader with a base directory."""
        self.base = base

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        """Load and return the template identified by ``name``."""
        path = self.base / f"{name}.yaml"
        text = path.read_text()
        data = yaml.safe_load(text)
        version = str(data.get("version", "0"))
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=version,
            labels=list(data.get("labels", [])),
            required_variables=list(data.get("required_variables", [])),
            yaml=text,
        )
        return version, tmpl


class MemoryLoader:
    """Load templates from an in-memory mapping."""

    def __init__(self, mapping: dict[str, dict[str, str]]):
        """Store the mapping of template name to template data."""
        self.mapping = mapping

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        """Return the template ``name`` from the mapping."""
        data = self.mapping.get(name)
        if not data:
            raise FileNotFoundError(name)
        text = data.get("yaml", "")
        ydata = yaml.safe_load(text)
        version = str(ydata.get("version", data.get("version", "0")))
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=version,
            labels=list(ydata.get("labels", [])),
            required_variables=list(ydata.get("required_variables", [])),
            yaml=text,
        )
        return version, tmpl


class HTTPLoader:
    """Fetch templates from an HTTP endpoint."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        """Initialize with ``base_url`` for the template registry."""
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient()

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        """Retrieve ``name`` from the remote registry."""
        params = {"label": label} if label else {}
        resp = await self.client.get(f"{self.base_url}/templates/{name}", params=params)
        if resp.status_code != 200:
            raise FileNotFoundError(name)
        data = resp.json()
        text = data.get("yaml", "")
        ydata = yaml.safe_load(text)
        version = str(ydata.get("version", data.get("version", "0")))
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=version,
            labels=list(ydata.get("labels", [])),
            required_variables=list(ydata.get("required_variables", [])),
            yaml=text,
        )
        return version, tmpl


class PromptLayerLoader:
    """Load templates from PromptLayer."""

    URL = "https://api.promptlayer.com/prompt-templates"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self.client = client or httpx.AsyncClient()

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        body = {"label": label} if label else {}
        resp = await self.client.post(
            f"{self.URL}/{name}",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json=body,
        )
        if resp.status_code != 200:
            raise FileNotFoundError(name)
        data = resp.json()
        content = data["prompt_template"]["content"]
        yaml_blob = yaml.safe_dump({"messages": content})
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=str(data["version"]),
            labels=[label] if label else [],
            yaml=yaml_blob,
        )
        return tmpl.version, tmpl


class LangfuseLoader:
    """Load templates via the Langfuse SDK."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        base_url: str = "https://cloud.langfuse.com",
    ) -> None:
        from langfuse import get_client

        self.client = get_client(public_key=public_key, secret_key=secret_key, base_url=base_url)

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        prm = await asyncio.to_thread(self.client.prompts().get_prompt, name, label=label)
        yaml_blob = prm.yaml
        meta = yaml.safe_load(yaml_blob)
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=str(prm.version),
            labels=prm.labels,
            yaml=yaml_blob,
            required_variables=meta.get("required_variables", []),
        )
        return tmpl.version, tmpl


class PezzoLoader:
    """Retrieve prompts via the Pezzo client."""

    def __init__(self, project: str) -> None:
        from pezzo import PezzoClient

        self.client = PezzoClient(project=project)

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        prompt = await self.client.get_prompt(slug=name, environment="production", version_tag=label)
        yaml_blob = prompt["yaml"]
        meta = yaml.safe_load(yaml_blob)
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=str(prompt["version"]),
            labels=meta.get("labels", prompt.get("tags", [])),
            yaml=yaml_blob,
        )
        return tmpl.version, tmpl


class AgentaLoader:
    """Fetch templates from Agenta via the SDK."""

    def __init__(self, app_slug: str) -> None:
        import agenta as ag

        ag.init()
        self.app_slug = app_slug

    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        cfg = await asyncio.to_thread(
            ag.ConfigManager.get_from_registry,
            app_slug=self.app_slug,
            variant_slug=name,
            environment_slug=label or "production",
        )
        yaml_blob = yaml.safe_dump(cfg["prompt"])
        tmpl = PromptTemplate(
            id=name,
            name=name,
            version=str(cfg.get("variant_version", "0")),
            labels=[label] if label else ["production"],
            yaml=yaml_blob,
        )
        return tmpl.version, tmpl


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
        import base64
        import codecs

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
            name=name,
            version=str(commit.hex[:7]),
            labels=meta.get("labels", []),
            yaml=text,
        )
        return tmpl.version, tmpl
