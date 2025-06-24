"""Example demonstrating PromptI with the OpenAI client."""

import asyncio
import logging
import os

from dotenv import load_dotenv

from prompti.engine import PromptEngine, Setting
from prompti.model_client import ModelConfig, create_client


async def main():
    """Run a simple support-reply prompt via OpenAI and print the results."""

    # Configure logging to write to console
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure ModelClient logger to INFO level
    client_logger = logging.getLogger("model_client")
    client_logger.setLevel(logging.INFO)

    # Load environment variables
    load_dotenv()

    settings = Setting(template_paths=["./prompts"])
    engine = PromptEngine.from_setting(settings)
    model_cfg = ModelConfig(
        provider="openai",
        model=os.getenv("MODEL_NAME", "claude-3-7-sonnet-20250219"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_url=os.getenv("OPENAI_API_URL"),
    )
    client = create_client(model_cfg)
    async for msg in engine.run(
        "support_reply",
        {"name": "Ada", "issue": "login failed"},
        tags=None,
        client=client,
        stream=True,
    ):
        print(f"{msg.role}/{msg.kind}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
