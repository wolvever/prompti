"""Minimal example demonstrating PromptI with synchronous completion."""

from __future__ import annotations
import uuid
import logging
from prompti.engine import PromptEngine, Setting
from prompti.model_client.base import ModelConfig, RunParams, ToolParams, ToolSpec

logging.basicConfig(level=logging.INFO)

setting = Setting(
    registry_url="http://10.224.55.241/api/v1",
    registry_api_key="7e5d106c-e701-4587-a95a-b7c7c02ee619",
)
engine = PromptEngine.from_setting(setting)



def stream_call() -> None:
    """Render ``simple-demo`` and print the response using sync completion."""

    try:
        for msg in engine.completion(
            "chatbot",
            variables={"instruction": "你是图像分析大师",
                       "query": "这张图片是什么？", "chat_history": ""},
            stream=True,
            variant="default",
            request_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            user_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parant_span_id=str(uuid.uuid4()),
            model_cfg={
                "provider": "openai",
                "model": "claude-sonnet-4-20250514"
            }

        ):
            print(msg)
    finally:
        # Note: In sync version, we don't need await
        # engine.close() is not async in sync context
        pass


def no_stream_call() -> None:
    """Render template and print the response without streaming."""
    try:
        for msg in engine.completion(
            "simple-demo",
            variables={"user_name": "小明",
                       "tasks": [{"name": "task_a", "priority": 2}, {"name": "task_b", "priority": 2}], "urgent": 1},
            stream=False,
            variant="use-jinja2",
        ):
            print(msg)
    finally:
        pass


def multi_modal_call() -> None:
    """Render multimodal template and print the response."""
    try:
        for msg in engine.completion(
            "simple-demo",
            variables={
                "instruction": "你是图像分析大师",
                "query": "这张图片是什么？",
                # 多张图片使用 "image_url": ["image1_url", "image2_url"]
                "image_url": "https://agentos-promptstore.bj.bcebos.com/files/test/images/default/2989be85-9bfb-4e18-a339-48466320bf0f.jpg?authorization=bce-auth-v1%2FALTAKQ5esVBHZqtt4HtwEwoNQh%2F2025-07-22T07%3A35%3A56Z%2F604800%2F%2F5761014c2df87427a1ab41a1d054820ecbb2f5cd1516e998dd2430a75549764a&response-content-disposition=inline&response-content-type=image%2Fjpeg",
            },
            variant="multimodal",
            stream=False,
            model_cfg=ModelConfig(
                provider="qianfan",
                model="ernie-4.5-turbo-vl-32k"
            )
        ):
            print(msg)
    finally:
        pass


def tool_call():
    """Demonstrate tool calling with sync completion."""
    try:
        for msg in engine.completion(
            "simple-demo",
            variables={
                "instruction": "你好",
                "query": "1+1=？"
            },
            stream=False,
            tool_params={"tools": [
                {
                    "name": "get_weather",
                    "description": "获取某地当前天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称，如北京"
                            }
                        },
                        "required": ["location"]
                    }

                },
                {
                    "name": "calculate",
                    "description": "执行基本的数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "要计算的数学表达式，例如：2+3*4"
                            }
                        },
                        "required": ["expression"]
                    }

                }
            ]}
        ):
            print(msg)
    finally:
        pass


def multi_chat() -> None:
    """Demonstrate multi-turn chat with sync completion."""
    try:
        for msg in engine.completion(
            "test_gxy_test",
            variables={},
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个聊天助手",
                },
                {
                    "role": "user",
                    "content": "你好啊",
                },
                {
                    "role": "assistant",
                    "content": "嗨~",
                },
                {
                    "role": "user",
                    "content": "你叫什么名字呀",
                },
            ]
        ):
            print(msg)
    finally:
        pass


def tool_call2() -> None:
    """Demonstrate tool calling with message history."""
    try:
        for msg in engine.completion(
            "coding_agent",
            version="1.0.4",
            variables={"instruction": "你是一个聊天助手", "query": "你好啊"},
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": "北京现在的天气怎么样？"
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tool_call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"北京\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_call_1",
                    "content": "{\"temperature\": \"30°C\", \"condition\": \"晴\"}"
                }
            ],
            model_cfg=ModelConfig(
                # provider="litellm",
                # model="anthropic/claude-sonnet-4-20250514",
                # api_key="sk-n2cV4S5ti02gnrNX5xhwQi8xUlFXgjfmsYKaZCYW8RIKts6x",
                # api_url="https://aiproxy.usw.sealos.io",
                provider="openai",
                model="claude-3-7-sonnet-20250219",
                temperature=0.7,
                top_p=0.5,
                max_tokens=1024,
            )
        ):
            print(msg)
    finally:
        pass


if __name__ == "__main__":
    stream_call()
    no_stream_call()
    multi_modal_call()
    tool_call()
    multi_chat()
    tool_call2()
