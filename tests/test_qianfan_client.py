import json
import pytest
import httpx
import os

from prompti.model_client import ModelConfig, QianfanClient
from prompti import Message
from openai_mock_server import OpenAIMockServer

@pytest.mark.asyncio
async def test_qianfan_client():
    with OpenAIMockServer("tests/data/qianfan_record.jsonl") as url:
        os.environ["QIANFAN_API_KEY"] = "testkey"
        client = QianfanClient(client=httpx.AsyncClient())
        client.api_url = url  # type: ignore

        cfg = ModelConfig(provider="qianfan", model="ernie-3.5-8k")
        messages = [Message(role="user", kind="text", content="你好")]
        out = [m async for m in client.run(messages, cfg)]
        assert out[0].content.startswith("你好！我是文心一言")

        messages = [Message(role="user", kind="text", content="计算15加27等于多少？")]
        out = [m async for m in client.run(messages, cfg)]
        assert "42" in out[0].content

        messages = [Message(role="user", kind="text", content="请写一首关于春天的诗")]
        out = [m async for m in client.run(messages, cfg)]
        assert "春风" in out[0].content
        assert "花开" in out[0].content

        messages = [Message(role="user", kind="text", content="如何用Python读取文件？")]
        out = [m async for m in client.run(messages, cfg)]
        assert "open" in out[0].content
        assert "with" in out[0].content

        messages = [Message(role="user", kind="text", content="请将'Hello World'翻译成中文")]
        out = [m async for m in client.run(messages, cfg)]
        assert "你好，世界" in out[0].content

        messages = [Message(role="user", kind="text", content="写一个关于小猫的短故事")]
        out = [m async for m in client.run(messages, cfg)]
        assert "小猫" in out[0].content or "小橘" in out[0].content

        messages = [Message(role="user", kind="text", content="如何保持健康的作息时间？")]
        out = [m async for m in client.run(messages, cfg)]
        assert "睡眠" in out[0].content
        assert "时间" in out[0].content

        messages = [Message(role="user", kind="text", content="推荐一个适合春季旅游的地方")]
        out = [m async for m in client.run(messages, cfg)]
        assert "西湖" in out[0].content or "旅游" in out[0].content

        messages = [Message(role="user", kind="text", content="如何提高学习效率？")]
        out = [m async for m in client.run(messages, cfg)]
        assert "学习" in out[0].content
        assert "效率" in out[0].content

        messages = [Message(role="user", kind="text", content="如何缓解工作压力？")]
        out = [m async for m in client.run(messages, cfg)]
        assert "压力" in out[0].content
        assert "工作" in out[0].content

        await client.close() 