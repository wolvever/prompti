"""Minimal example demonstrating PromptI with LiteLLM."""

import asyncio
import os

from prompti.engine import PromptEngine, Setting
from prompti.model_client import ModelConfig, create_client


async def main() -> None:
    """Render ``support_reply`` and print the response."""

    engine = PromptEngine.from_setting(Setting())
    cfg = ModelConfig(
        provider="litellm",
        model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        api_key=os.getenv("LITELLM_API_KEY"),
        api_url=os.getenv("LITELLM_ENDPOINT"),
    )
    client = create_client(cfg)

    async for msg in engine.run(
        "support_reply",
        {"name": "Ada", "issue": "login failed"},
        client=client,
        stream=True,
    ):
        print(msg.content)


if __name__ == "__main__":
    asyncio.run(main())
