"""Usage: python -m prompti.examples.chat_cli -q 'What is the weather in Tokyo?'."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
from datetime import datetime

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from prometheus_client import start_http_server

from prompti.model_client import (
    Message,
    ModelConfig,
    RunParams,
    ToolParams,
    ToolSpec,
    create_client,
)


def encode_file(path: str) -> dict[str, str]:
    """Return A2A file payload for *path*."""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as fh:
        data = fh.read()
    return {
        "name": os.path.basename(path),
        "mimeType": mime,
        "bytes": base64.b64encode(data).decode("ascii"),
    }


def get_time(_: dict | None = None) -> str:
    """Return the current UTC time in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def setup_observability(port: int = 8000) -> None:
    """Start Prometheus metrics server and configure console tracing."""
    start_http_server(port)
    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


async def main() -> None:  # noqa: C901 - command-line interface complexity
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description="Simple LLM CLI")
    parser.add_argument("-q", "--query", required=True, help="Query text to send")
    parser.add_argument(
        "-f",
        "--file",
        action="append",
        help="Path to a file to attach (may repeat)",
    )
    parser.add_argument(
        "--time-tool",
        action="store_true",
        help="Enable built-in get_time tool",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Request reasoning messages if supported",
    )
    parser.add_argument("--api-url", help="Base URL for the LLM API")
    parser.add_argument("--api-key", help="API key for the provider")
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Model name (default: gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Stream responses (default)",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming",
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("PROMPTI_PROVIDER", "litellm"),
        help="Model provider (default from PROMPTI_PROVIDER)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    setup_observability()

    cfg = ModelConfig(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_url=args.api_url,
    )
    client = create_client(cfg)

    messages: list[Message] = []
    if args.file:
        for path in args.file:
            messages.append(Message(role="user", kind="file", content=encode_file(path)))
    messages.append(Message(role="user", kind="text", content=args.query))

    tool_params = None
    if args.time_tool:
        tool = ToolSpec(
            name="get_time",
            description="Return the current UTC time",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        tool_params = ToolParams(tools=[tool], choice={"type": "function", "function": {"name": "get_time"}})

    stream = args.stream
    extra_params = {}
    if args.reasoning:
        extra_params["enable_reasoning"] = True

    params = RunParams(
        messages=messages,
        tool_params=tool_params,
        stream=stream,
        extra_params=extra_params,
    )

    while True:
        logging.info("=== Response ===")
        tool_call = None
        async for msg in client.run(params):
            print(f"{msg.role}/{msg.kind}: {msg.content}")
            if msg.kind == "tool_use":
                tool_call = msg
                break
        if tool_call is None:
            break

        call = tool_call.content
        if isinstance(call, str):
            call = json.loads(call)
        if call.get("name") == "get_time":
            result = get_time(call.get("arguments"))
        else:
            result = f"No handler for tool {call.get('name')}"

        messages.append(tool_call)
        messages.append(Message(role="user", kind="tool_result", content=result))
        params = RunParams(messages=messages, stream=stream)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
