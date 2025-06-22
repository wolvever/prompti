"""Python wrapper for the Rust `model_client_rs` library."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

try:
    import model_client_rs as _rs
except Exception:  # pragma: no cover - extension missing is not fatal at import
    _rs = None


class ModelClient:
    """Wrapper around the Rust ``ModelClient``."""

    def __init__(self) -> None:
        if _rs is None:
            raise RuntimeError(
                "The `model_client_rs` extension is required but could not be imported"
            )
        self._client = _rs.ModelClient()

    async def chat_stream(
        self, request: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Proxy ``chat_stream`` call to the Rust client."""
        async for chunk in self._client.chat_stream(request):
            yield chunk
