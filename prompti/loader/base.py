from __future__ import annotations

from abc import ABC, abstractmethod

from ..template import PromptTemplate


class TemplateLoader(ABC):
    """Abstract base class for template loaders."""

    @abstractmethod
    async def __call__(self, name: str, label: str | None) -> tuple[str, PromptTemplate]:
        """Return the template identified by ``name``.

        Parameters
        ----------
        name: str
            Template name to load.
        label: str | None
            Optional label used by some backends to select a version.
        Returns
        -------
        tuple[str, PromptTemplate]
            The template version and :class:`PromptTemplate` instance.
        """
        raise NotImplementedError
