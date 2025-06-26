"""Format a template with :class:`PromptEngine` and send via LiteLLM."""

import asyncio
import os

import litellm

from prompti.engine import PromptEngine, Setting


async def main() -> None:
    """Render ``support_reply`` template and send via ``litellm``."""
    engine = PromptEngine.from_setting(Setting())

    # Format the template using ``engine.format``
    a2a_messages = await engine.format(
        "support_reply",
        {"name": "Ada", "issue": "login failed"},
    )

    # Convert A2A messages to OpenAI/LiteLLM format
    oa_messages = [
        {"role": m.role, "content": m.content}
        for m in a2a_messages
    ]

    # Send to LiteLLM using the rendered messages
    response = await litellm.acompletion(
        model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        messages=oa_messages,
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url=os.getenv("LITELLM_ENDPOINT"),
    )

    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
