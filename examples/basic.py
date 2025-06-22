"""Example demonstrating PromptI with the OpenAI client."""

import asyncio

from prompti.engine import PromptEngine, Setting
from prompti.model_client import ModelConfig, create_client


async def main():
    """Run a simple support-reply prompt via OpenAI and print the results."""
    settings = Setting(template_paths=["./prompts"])
    engine = PromptEngine.from_setting(settings)
    model_cfg = ModelConfig(provider="openai", model="gpt-4o")
    client = create_client(model_cfg)
    async for msg in engine.run(
        "support_reply",
        {"name": "Ada", "issue": "login failed"},
        tags=None,
        model_cfg=model_cfg,
        client=client,
    ):
        print(f"{msg.role}/{msg.kind}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
