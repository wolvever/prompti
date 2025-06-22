from __future__ import annotations

from pathlib import Path
import yaml

from ..template import PromptTemplate


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
