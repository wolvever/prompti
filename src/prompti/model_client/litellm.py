"""LiteLLM client implementation using the `litellm` package."""

from __future__ import annotations

import time
import uuid
import traceback
from collections.abc import AsyncGenerator, Generator
from typing import Any, Dict, List, Union

import httpx

from ..message import Message, ModelResponse, StreamingModelResponse, Usage, Choice, StreamingChoice
from .base import ModelClient, SyncModelClient, ModelConfig, RunParams, ToolChoice, ToolParams, ToolSpec


class LiteLLMClient(ModelClient):
    """Client for the LiteLLM API."""

    provider = "litellm"

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None, is_debug: bool = False) -> None:
        """Instantiate the client with configuration and optional HTTP client."""
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e

        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url
        self.base_url = cfg.api_url  # 为了兼容性，将api_url赋值给base_url

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """构建LiteLLM API请求数据。"""
        # 转换消息格式
        messages = [m.to_openai() for m in params.messages]

        # 基础请求数据
        request_data = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": params.stream,
        }
        if params.stream:
            request_data.update({"stream_options": {"include_usage": True}})

        # 添加认证和API地址
        if self.api_key:
            request_data["api_key"] = self.api_key
        if self.api_url:
            request_data["api_base"] = self.api_url  # litellm 使用 api_base 参数

        # 添加可选参数
        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature

        if params.top_p is not None:
            request_data["top_p"] = params.top_p
        elif self.cfg.top_p is not None:
            request_data["top_p"] = self.cfg.top_p

        if params.max_tokens is not None:
            request_data["max_tokens"] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            request_data["max_tokens"] = self.cfg.max_tokens

        if params.stop:
            request_data["stop"] = params.stop

        if params.n is not None:
            request_data["n"] = params.n

        if params.seed is not None:
            request_data["seed"] = params.seed

        if params.logit_bias:
            request_data["logit_bias"] = params.logit_bias

        if params.response_format:
            request_data["response_format"] = {"type": params.response_format}

        if params.user_id:
            request_data["user"] = params.user_id

        # 处理工具参数
        if params.tool_params:
            if isinstance(params.tool_params, ToolParams):
                tools = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params.tools
                ]
                request_data["tools"] = tools
                choice = params.tool_params.choice
                if isinstance(choice, ToolChoice):
                    if choice is not ToolChoice.AUTO:
                        request_data["tool_choice"] = choice.value
                elif choice is not None:
                    request_data["tool_choice"] = choice
            else:
                request_data["tools"] = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params
                ]

        # 添加额外参数
        request_data.update(params.extra_params)
        params.trace_context["llm_request"] = request_data
        return request_data

    async def _aprocess_streaming_response(self, response) -> AsyncGenerator[StreamingModelResponse, None]:
        """处理流式响应。"""
        async for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta if hasattr(choice, "delta") else {}

                # 处理流式工具调用 - 转换为字典格式
                tool_calls = None
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tool_calls = []
                    for tool_call in delta.tool_calls:
                        # 构建工具调用字典
                        tool_call_dict = {"type": "function"}

                        # 处理ID
                        if hasattr(tool_call, "id"):
                            tool_call_dict["id"] = tool_call.id

                        # 处理函数信息
                        if hasattr(tool_call, "function"):
                            function_dict = {}
                            if hasattr(tool_call.function, "name"):
                                function_dict["name"] = tool_call.function.name
                            if hasattr(tool_call.function, "arguments"):
                                function_dict["arguments"] = tool_call.function.arguments
                            tool_call_dict["function"] = function_dict

                        tool_calls.append(tool_call_dict)

                # 创建StreamingChoice对象
                streaming_choice = StreamingChoice(
                    index=choice.index if hasattr(choice, "index") else 0,
                    delta=Message(
                        role="assistant",
                        content=delta.content if hasattr(delta, "content") else None,
                        tool_calls=tool_calls
                    ),
                    finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
                )

                # 创建StreamingResponse对象
                streaming_response = StreamingModelResponse(
                    id=chunk.id if hasattr(chunk, "id") else str(uuid.uuid4()),
                    created=chunk.created if hasattr(chunk, "created") else int(time.time()),
                    model=chunk.model if hasattr(chunk, "model") else self.cfg.model,
                    choices=[streaming_choice],
                    usage=Usage(
                        prompt_tokens=chunk.usage.prompt_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        completion_tokens=chunk.usage.completion_tokens if hasattr(chunk,
                                                                                   "usage") and chunk.usage else 0,
                        total_tokens=chunk.usage.total_tokens if hasattr(chunk, "usage") and chunk.usage else 0
                    ) if hasattr(chunk, "usage") and chunk.usage else None
                )

                yield streaming_response

    def _process_non_streaming_response(self, response) -> ModelResponse:
        """处理非流式响应。"""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message if hasattr(choice, "message") else {}

            # 处理工具调用 - LiteLLM返回的是LiteLLM特有的对象，需要转换为字典
            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    # 转换工具调用为字典格式
                    tool_call_dict = {
                        "id": tool_call.id if hasattr(tool_call, "id") else "",
                        "type": tool_call.type if hasattr(tool_call, "type") else "function",
                        "function": {
                            "name": tool_call.function.name if hasattr(tool_call, "function") and hasattr(
                                tool_call.function, "name") else "",
                            "arguments": tool_call.function.arguments if hasattr(tool_call, "function") and hasattr(
                                tool_call.function, "arguments") else "{}"
                        }
                    }
                    tool_calls.append(tool_call_dict)

            # 创建Choice对象
            model_choice = Choice(
                index=choice.index if hasattr(choice, "index") else 0,
                message=Message(
                    role="assistant",
                    content=message.content if hasattr(message, "content") else None,
                    tool_calls=tool_calls
                ),
                finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
            )

            # 创建Usage对象
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0,
                    completion_tokens=response.usage.completion_tokens if hasattr(response.usage,
                                                                                  "completion_tokens") else 0,
                    total_tokens=response.usage.total_tokens if hasattr(response.usage, "total_tokens") else 0
                )

            # 创建ModelResponse对象
            return ModelResponse(
                id=response.id if hasattr(response, "id") else str(uuid.uuid4()),
                created=response.created if hasattr(response, "created") else int(time.time()),
                model=response.model if hasattr(response, "model") else self.cfg.model,
                choices=[model_choice],
                usage=usage
            )
        else:
            # 返回空响应
            return ModelResponse(
                id=str(uuid.uuid4()),
                model=self.cfg.model,
                choices=[Choice(
                    index=0,
                    message=Message(role="assistant", content=""),
                    finish_reason="stop"
                )]
            )

    def _create_error_response(self, error_message: str, is_streaming: bool = False) -> Union[
        ModelResponse, StreamingModelResponse]:
        """创建错误响应。"""
        error_object = {
            "message": error_message,
            "type": "litellm_error",
            "code": "request_error"
        }

        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)

    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Execute the LiteLLM API call."""
        try:
            # 确保已安装litellm
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for LiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e

        # 构建请求数据
        request_data = self._build_request_data(params)
        self._logger.info(f"litellm request data: {request_data}")
        try:
            if params.stream:
                # 处理流式响应
                response = await litellm.acompletion(
                    **request_data
                )
                async for message in self._aprocess_streaming_response(response):
                    yield message
            else:
                # 处理非流式响应
                response = await litellm.acompletion(
                    **request_data
                )
                yield self._process_non_streaming_response(response)

        except litellm.exceptions.BadRequestError as e:
            # LiteLLM 请求错误
            error_detail = str(e)
            self._logger.error(f"LiteLLM API error: {error_detail}")
            yield self._create_error_response(error_detail, is_streaming=params.stream)

        except litellm.exceptions.AuthenticationError as e:
            # 认证错误
            error_detail = f"Authentication error: {str(e)}"
            self._logger.error(error_detail)
            yield self._create_error_response(error_detail, is_streaming=params.stream)

        except httpx.RequestError as e:
            # 网络连接错误
            error_msg = f"Network error: {str(e)}"
            self._logger.error(error_msg)
            yield self._create_error_response(error_msg, is_streaming=params.stream)

        except Exception as e:
            # 其他错误
            error_msg = f"Unexpected error: {str(e)}"
            self._logger.error(error_msg)
            traceback.print_exc()
            yield self._create_error_response(error_msg, is_streaming=params.stream)

    async def aclose(self) -> None:
        """Close the underlying HTTP client and clean up LiteLLM resources."""
        # Close our httpx client
        await super().aclose()

        # Try to clean up LiteLLM's internal aiohttp sessions
        try:
            import aiohttp
            import gc

            # Force garbage collection of aiohttp connectors
            gc.collect()

            # Try to close any open aiohttp sessions
            for obj in gc.get_objects():
                if isinstance(obj, aiohttp.ClientSession):
                    if not obj.closed:
                        try:
                            await obj.close()
                        except Exception:
                            pass

        except Exception:
            # Ignore errors in cleanup
            pass


class SyncLiteLLMClient(SyncModelClient):
    """Synchronous client for the LiteLLM API."""

    provider = "litellm"

    def __init__(self, cfg: ModelConfig, client: httpx.Client | None = None, is_debug: bool = False) -> None:
        """Instantiate the client with configuration and optional HTTP client."""
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "litellm is required for SyncLiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e

        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url
        self.base_url = cfg.api_url

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """构建LiteLLM API请求数据。"""
        messages = [m.to_openai() for m in params.messages]

        request_data = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": params.stream,
        }
        if params.stream:
            request_data.update({"stream_options": {"include_usage": True}})

        if self.api_key:
            request_data["api_key"] = self.api_key
        if self.api_url:
            request_data["api_base"] = self.api_url

        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature

        if params.top_p is not None:
            request_data["top_p"] = params.top_p
        elif self.cfg.top_p is not None:
            request_data["top_p"] = self.cfg.top_p

        if params.max_tokens is not None:
            request_data["max_tokens"] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            request_data["max_tokens"] = self.cfg.max_tokens

        if params.stop:
            request_data["stop"] = params.stop

        if params.n is not None:
            request_data["n"] = params.n

        if params.seed is not None:
            request_data["seed"] = params.seed

        if params.logit_bias:
            request_data["logit_bias"] = params.logit_bias

        if params.response_format:
            request_data["response_format"] = {"type": params.response_format}

        if params.user_id:
            request_data["user"] = params.user_id

        if params.tool_params:
            if isinstance(params.tool_params, ToolParams):
                tools = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params.tools
                ]
                request_data["tools"] = tools
                choice = params.tool_params.choice
                if isinstance(choice, ToolChoice):
                    if choice is not ToolChoice.AUTO:
                        request_data["tool_choice"] = choice.value
                elif choice is not None:
                    request_data["tool_choice"] = choice
            else:
                request_data["tools"] = [
                    {"type": "function", "function": t.model_dump()} if isinstance(t, ToolSpec) else t
                    for t in params.tool_params
                ]

        request_data.update(params.extra_params)
        params.trace_context["llm_request"] = request_data
        return request_data

    def _process_streaming_response(self, response) -> Generator[StreamingModelResponse, None, None]:
        """处理流式响应。"""
        for chunk in response:
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta if hasattr(choice, "delta") else {}

                tool_calls = None
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tool_calls = []
                    for tool_call in delta.tool_calls:
                        tool_call_dict = {"type": "function"}

                        if hasattr(tool_call, "id"):
                            tool_call_dict["id"] = tool_call.id

                        if hasattr(tool_call, "function"):
                            function_dict = {}
                            if hasattr(tool_call.function, "name"):
                                function_dict["name"] = tool_call.function.name
                            if hasattr(tool_call.function, "arguments"):
                                function_dict["arguments"] = tool_call.function.arguments
                            tool_call_dict["function"] = function_dict

                        tool_calls.append(tool_call_dict)

                streaming_choice = StreamingChoice(
                    index=choice.index if hasattr(choice, "index") else 0,
                    delta=Message(
                        role="assistant",
                        content=delta.content if hasattr(delta, "content") else None,
                        tool_calls=tool_calls
                    ),
                    finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
                )

                streaming_response = StreamingModelResponse(
                    id=chunk.id if hasattr(chunk, "id") else str(uuid.uuid4()),
                    created=chunk.created if hasattr(chunk, "created") else int(time.time()),
                    model=chunk.model if hasattr(chunk, "model") else self.cfg.model,
                    choices=[streaming_choice],
                    usage=Usage(
                        prompt_tokens=chunk.usage.prompt_tokens if hasattr(chunk, "usage") and chunk.usage else 0,
                        completion_tokens=chunk.usage.completion_tokens if hasattr(chunk,
                                                                                   "usage") and chunk.usage else 0,
                        total_tokens=chunk.usage.total_tokens if hasattr(chunk, "usage") and chunk.usage else 0
                    ) if hasattr(chunk, "usage") and chunk.usage else None
                )

                yield streaming_response

    def _process_non_streaming_response(self, response) -> ModelResponse:
        """处理非流式响应。"""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message if hasattr(choice, "message") else {}

            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_call_dict = {
                        "id": tool_call.id if hasattr(tool_call, "id") else "",
                        "type": tool_call.type if hasattr(tool_call, "type") else "function",
                        "function": {
                            "name": tool_call.function.name if hasattr(tool_call, "function") and hasattr(
                                tool_call.function, "name") else "",
                            "arguments": tool_call.function.arguments if hasattr(tool_call, "function") and hasattr(
                                tool_call.function, "arguments") else "{}"
                        }
                    }
                    tool_calls.append(tool_call_dict)

            model_choice = Choice(
                index=choice.index if hasattr(choice, "index") else 0,
                message=Message(
                    role="assistant",
                    content=message.content if hasattr(message, "content") else None,
                    tool_calls=tool_calls
                ),
                finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None
            )

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0,
                    completion_tokens=response.usage.completion_tokens if hasattr(response.usage,
                                                                                  "completion_tokens") else 0,
                    total_tokens=response.usage.total_tokens if hasattr(response.usage, "total_tokens") else 0
                )

            return ModelResponse(
                id=response.id if hasattr(response, "id") else str(uuid.uuid4()),
                created=response.created if hasattr(response, "created") else int(time.time()),
                model=response.model if hasattr(response, "model") else self.cfg.model,
                choices=[model_choice],
                usage=usage
            )
        else:
            return ModelResponse(
                id=str(uuid.uuid4()),
                model=self.cfg.model,
                choices=[Choice(
                    index=0,
                    message=Message(role="assistant", content=""),
                    finish_reason="stop"
                )]
            )

    def _create_error_response(self, error_message: str, is_streaming: bool = False) -> Union[
        ModelResponse, StreamingModelResponse]:
        """创建错误响应。"""
        error_object = {
            "message": error_message,
            "type": "litellm_error",
            "code": "request_error"
        }

        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)

    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Execute the LiteLLM API call."""
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm is required for SyncLiteLLMClient. Install with: pip install 'prompti[litellm]'"
            ) from e

        request_data = self._build_request_data(params)
        self._logger.info(f"litellm request data: {request_data}")
        try:
            if params.stream:
                response = litellm.completion(
                    **request_data
                )
                for message in self._process_streaming_response(response):
                    yield message
            else:
                response = litellm.completion(
                    **request_data
                )
                yield self._process_non_streaming_response(response)

        except Exception as e:
            try:
                import litellm.exceptions
                if isinstance(e, litellm.exceptions.BadRequestError):
                    error_detail = str(e)
                    self._logger.error(f"LiteLLM API error: {error_detail}")
                    yield self._create_error_response(error_detail, is_streaming=params.stream)
                elif isinstance(e, litellm.exceptions.AuthenticationError):
                    error_detail = f"Authentication error: {str(e)}"
                    self._logger.error(error_detail)
                    yield self._create_error_response(error_detail, is_streaming=params.stream)
                else:
                    raise
            except (ImportError, AttributeError):
                pass

            if isinstance(e, httpx.RequestError):
                error_msg = f"Network error: {str(e)}"
                self._logger.error(error_msg)
                yield self._create_error_response(error_msg, is_streaming=params.stream)
            else:
                error_msg = f"Unexpected error: {str(e)}"
                self._logger.error(error_msg)
                traceback.print_exc()
                yield self._create_error_response(error_msg, is_streaming=params.stream)
