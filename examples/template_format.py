"""Format a template with :class:`PromptEngine` and send via LiteLLM."""

import asyncio
import os

import litellm

from prompti.engine import PromptEngine, Setting


async def main() -> None:
    """Render ``support_reply`` template and send via ``litellm``."""
    engine = PromptEngine.from_setting(Setting())

    # Format the template directly as OpenAI messages and send
    messages = await engine.format(
        "support_reply",
        {"name": "Ada", "issue": "login failed"},
        format="openai",
    )

    response = await litellm.acompletion(
        model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        messages=messages,
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url=os.getenv("LITELLM_ENDPOINT"),
    )

    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
