"""Primitive message type used throughout the package."""

from __future__ import annotations

from typing import Any
from enum import Enum

from pydantic import BaseModel


class Kind(str, Enum):
    """Valid message part kinds from the A2A specification."""

    TEXT = "text"
    FILE = "file"
    DATA = "data"


class Message(BaseModel):
    """A2A protocol message."""

    role: str
    kind: str
    content: Any
