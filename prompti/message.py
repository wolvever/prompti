from __future__ import annotations

"""Primitive message type used throughout the package."""

from typing import Any
from pydantic import BaseModel


class Message(BaseModel):
    """A2A protocol message."""

    role: str
    kind: str
    content: Any
