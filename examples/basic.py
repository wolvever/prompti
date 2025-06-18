import asyncio

from prompti.engine import PromptEngine, Settings
from prompti.model_client import ModelConfig, OpenAIClient

async def main():
    settings = Settings(template_paths=["./prompts"])
    engine = PromptEngine.from_settings(settings)
    model_cfg = ModelConfig(provider="openai", model="gpt-4o")
    client = OpenAIClient()
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
