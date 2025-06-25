from __future__ import annotations

import yaml

from ..template import PromptTemplate, Variant
from .base import TemplateLoader


class MemoryLoader(TemplateLoader):
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
            name=ydata.get("name", name),
            description=ydata.get("description", ""),
            version=version,
            tags=list(ydata.get("tags", [])),
            variants={k: Variant(**v) for k, v in ydata.get("variants", {}).items()},
            yaml=text,
        )
        return version, tmpl
