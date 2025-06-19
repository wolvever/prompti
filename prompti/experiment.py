from __future__ import annotations

"""SDK level A/B experiment helpers."""

from typing import Protocol, Dict
from pydantic import BaseModel
import httpx
import xxhash


class ExperimentSplit(BaseModel):
    """Result of an experiment lookup."""

    experiment_id: str | None = None
    variant: str | None = None
    traffic_split: Dict[str, float] | None = None


class ExperimentRegistry(Protocol):
    """Lookup experiment variant for a prompt/user."""

    async def get_split(self, prompt: str, user_id: str) -> ExperimentSplit:
        """Return the experiment split for ``prompt`` and ``user_id``."""
        ...


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------

def bucket(hash_key: str, split: Dict[str, float]) -> str:
    """Return variant bucket using xxhash based distribution."""
    h = xxhash.xxh32(hash_key).intdigest() / 2**32
    total = 0.0
    for variant, pct in split.items():
        total += pct
        if h < total:
            return variant
    return next(iter(split))


# ---------------------------------------------------------------------------
# Unleash adapter
# ---------------------------------------------------------------------------

class UnleashRegistry:
    """Experiment registry that queries an Unleash server."""

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None) -> None:
        """Create registry using ``base_url`` and optional HTTP ``client``."""
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient()

    async def get_split(self, prompt: str, user_id: str) -> ExperimentSplit:
        """Resolve the split for ``prompt`` from the Unleash server."""
        url = f"{self.base_url}/client/features/{prompt}"
        resp = await self._client.get(
            url,
            headers={
                "UNLEASH-APPNAME": "prompti",
                "UNLEASH-INSTANCEID": user_id,
            },
        )
        data = resp.json()
        variant = None
        if isinstance(data, dict):
            variant = (data.get("variant") or {}).get("name")
        if not variant or variant == "disabled":
            return ExperimentSplit()
        return ExperimentSplit(experiment_id=data.get("name"), variant=variant)


# ---------------------------------------------------------------------------
# GrowthBook adapter
# ---------------------------------------------------------------------------

class GrowthBookRegistry:
    """Simple GrowthBook adapter using an in-memory feature map."""

    def __init__(self, features: Dict[str, Dict[str, float | str]]) -> None:
        """Initialize with ``features`` describing experiments and weights."""
        self._features = features

    async def get_split(self, prompt: str, user_id: str) -> ExperimentSplit:
        """Return the configuration for ``prompt`` without selecting a variant."""
        conf = self._features.get(prompt)
        if not conf:
            return ExperimentSplit()
        variants = conf.get("variants", {})
        if not variants:
            return ExperimentSplit()
        exp_id = conf.get("id", prompt)
        return ExperimentSplit(experiment_id=exp_id, traffic_split=variants)
