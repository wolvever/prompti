from __future__ import annotations

from typing import Any, AsyncGenerator, List
from pydantic import BaseModel
from jinja2.sandbox import SandboxedEnvironment

from .message import Message
from .model_client import ModelConfig, ModelClient

_env = SandboxedEnvironment()


class PromptTemplate(BaseModel):
    id: str
    name: str
    version: str
    jinja_source: str
    tags: set[str] = set()

    def format(
        self,
        variables: dict[str, Any],
        tag: str | None = None,
    ) -> List[Message]:
        template = _env.from_string(self.jinja_source)
        rendered = template.render(**variables, tag=tag)
        # A simple parser: split lines like '<role:kind ...>' into Message
        messages: list[Message] = []
        for line in rendered.splitlines():
            if line.startswith("<") and ":" in line and ">" in line:
                # Format: <assistant:tool_use name="...">JSON</>
                closing = line.find(">")
                header = line[1:closing]
                body = line[closing + 1 :]
                role, kind = header.split(":", 1)
                content = body.strip()
                messages.append(Message(role=role, kind=kind, content=content))
            else:
                if line.strip():
                    messages.append(
                        Message(role="assistant", kind="text", content=line)
                    )
        return messages

    async def run(
        self,
        variables: dict[str, Any],
        tag: str | None,
        model_cfg: ModelConfig,
        client: ModelClient,
    ) -> AsyncGenerator[Message, None]:
        messages = self.format(variables, tag)
        async for m in client.run(messages, model_cfg=model_cfg):
            yield m
