"""Fetch prompt templates from a remote HTTP service."""

from __future__ import annotations

import httpx

from ..template import PromptTemplate, Variant, ModelConfig
from .base import TemplateLoader, TemplateNotFoundError, VersionEntry


class HTTPLoader(TemplateLoader):
    """Fetch templates from an HTTP endpoint."""

    def __init__(self, base_url: str, auth_token: str, client: httpx.AsyncClient | None = None) -> None:
        """Initialize with ``base_url`` for the template registry."""
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(30))
        self.sync_client = httpx.Client(timeout=httpx.Timeout(30))
        self.headers = {"Authorization": f"Bearer {auth_token}"}

    async def alist_versions(self, name: str) -> list[VersionEntry]:
        """List all available versions for a template from HTTP endpoint."""
        try:
            resp = await self.client.get(f"{self.base_url}/template/{name}/versions", headers=self.headers)
            if resp.status_code != 200:
                return []

            versions_data = resp.json()
            return [VersionEntry(id=str(v.get("version", "0")),
                                 aliases=list(v.get("aliases", []))) for v in versions_data]
        except (httpx.RequestError, ValueError, KeyError):
            return []

    async def aget_template(self, name: str, version: str) -> PromptTemplate:
        """Retrieve specific version of template from the remote registry."""
        try:
            if version:
                url = f"{self.base_url}/template/{name}?label={version}"
            else:
                url = f"{self.base_url}/template/{name}"
            resp = await self.client.get(url=url, headers=self.headers)
            if resp.status_code != 200:
                raise TemplateNotFoundError(
                    f"Template {name} version {version} not found"
                )

            data = resp.json()
            data = data.get("data", {})
            template_version = data.get("version")
            variants = data.get("variants", {})
            final_variants = {}
            for variant_name, variant in variants.items():
                model_cfg_dict = variant.get("model_cfg") or {}
                model_cfg = ModelConfig(
                    provider=model_cfg_dict.get("provider"),
                    model=model_cfg_dict.get("model"),
                    api_key=model_cfg_dict.get("api_key"),
                    api_url=model_cfg_dict.get("api_url"),
                    temperature=model_cfg_dict.get("temperature"),
                    top_p=model_cfg_dict.get("top_p"),
                    max_tokens=model_cfg_dict.get("max_tokens"),
                )
                final_variants[variant_name] = Variant(
                    selector=variant.get("selector", []),
                    model_cfg=model_cfg,
                    messages=variant["messages_template"],
                    required_variables=variant.get("required_variables") or [],
                )
            tmpl = PromptTemplate(
                id=data.get("template_id"),
                name=data.get("name", name),
                description="",
                version=template_version,
                aliases=list(data.get("aliases", [])),
                variants=final_variants,
            )
            return tmpl
        except Exception as e:
            print(
                f"Template {name} version {version} not found"
            )
            return None

    def list_versions_sync(self, name: str) -> list[VersionEntry]:
        """Synchronous version of alist_versions."""
        try:
            resp = self.sync_client.get(f"{self.base_url}/template/{name}/versions", headers=self.headers)
            if resp.status_code != 200:
                return []

            versions_data = resp.json()
            return [VersionEntry(id=str(v.get("version", "0")),
                                 aliases=list(v.get("aliases", []))) for v in versions_data]
        except (httpx.RequestError, ValueError, KeyError):
            return []

    def get_template_sync(self, name: str, version: str) -> PromptTemplate:
        """Synchronous version of aget_template."""
        try:
            if version:
                url = f"{self.base_url}/template/{name}?label={version}"
            else:
                url = f"{self.base_url}/template/{name}"
            resp = self.sync_client.get(url=url, headers=self.headers)
            if resp.status_code != 200:
                raise TemplateNotFoundError(
                    f"Template {name} version {version} not found"
                )

            data = resp.json()
            data = data.get("data", {})
            template_version = data.get("version")
            variants = data.get("variants", {})
            final_variants = {}
            for variant_name, variant in variants.items():
                model_cfg_dict = variant.get("model_cfg") or {}
                model_cfg = ModelConfig(
                    provider=model_cfg_dict.get("provider"),
                    model=model_cfg_dict.get("model"),
                    api_key=model_cfg_dict.get("api_key"),
                    api_url=model_cfg_dict.get("api_url"),
                    temperature=model_cfg_dict.get("temperature"),
                    top_p=model_cfg_dict.get("top_p"),
                    max_tokens=model_cfg_dict.get("max_tokens"),
                )
                final_variants[variant_name] = Variant(
                    selector=variant.get("selector", []),
                    model_cfg=model_cfg,
                    messages=variant["messages_template"],
                    required_variables=variant.get("required_variables") or [],
                )
            tmpl = PromptTemplate(
                id=data.get("template_id"),
                name=data.get("name", name),
                description="",
                version=template_version,
                aliases=list(data.get("aliases", [])),
                variants=final_variants,
            )
            return tmpl
        except Exception as e:
            print(
                f"Template {name} version {version} not found"
            )
            return None
