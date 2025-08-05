"""Core engine that resolves templates and executes them with model clients."""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache

import yaml
from collections.abc import AsyncGenerator, Generator
from typing import Union
from pathlib import Path
from typing import Any, cast, ClassVar
import time

from async_lru import alru_cache
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict

from .loader import (
    FileSystemLoader,
    MemoryLoader,
    HTTPLoader,
    TemplateLoader,
    TemplateNotFoundError,
)
from .message import Message, ModelResponse, StreamingModelResponse
from .trace import TraceService, TraceEvent
from .model_client import ModelConfig, RunParams, ToolParams, ToolSpec
from .model_client.factory import create_client
from .model_client.config_loader import ModelConfigLoader, FileModelConfigLoader, \
    HTTPModelConfigLoader, ModelConfigNotFoundError, MemoryModelConfigLoader
from .template import PromptTemplate

_tracer = trace.get_tracer(__name__)


class PromptEngine:
    """Resolve templates and generate model responses."""

    def __init__(
        self,
        prompt_loaders: list[TemplateLoader],
        model_loaders: list[ModelConfigLoader] | None = None,
        cache_ttl: int = 300,
        global_model_config: ModelConfig | None = None,
        trace_service: TraceService | None = None,
    ) -> None:
        """Initialize the engine with prompt loaders, model loaders and optional global config."""
        self._prompt_loaders = prompt_loaders
        self._model_loaders = model_loaders or []
        self._cache_ttl = cache_ttl
        self._global_cfg = global_model_config
        self._trace_service = trace_service
        self._resolve = alru_cache(maxsize=128, ttl=cache_ttl)(self._resolve_impl)
        self._sync_resolve = lru_cache(maxsize=128)(self._sync_resolve_impl)

    async def _resolve_impl(self, name: str, version: str | None) -> PromptTemplate:
        for loader in self._prompt_loaders:
            tmpl = await loader.aget_template(name, version)
            if not tmpl:
                continue
            return tmpl
        raise TemplateNotFoundError(name)
    
    def _sync_resolve_impl(self, name: str, version: str | None) -> PromptTemplate:
        """Synchronous template resolution implementation."""
        for loader in self._prompt_loaders:
            # For sync resolution, we need to handle different loader types
            if hasattr(loader, 'get_template_sync'):
                tmpl = loader.get_template_sync(name, version)
                if tmpl:
                    return tmpl
            # else:
            #     # For loaders that only have async methods, we run them in sync context
            #     import asyncio
            #     try:
            #         loop = asyncio.get_event_loop()
            #         if loop.is_running():
            #             # If we're in an async context, we can't use run_until_complete
            #             # In this case, we should use a different approach
            #             raise RuntimeError("Cannot resolve template synchronously from async context")
            #         else:
            #             tmpl = loop.run_until_complete(loader.aget_template(name, version))
            #     except RuntimeError:
            #         # No event loop, create one
            #         tmpl = asyncio.run(loader.aget_template(name, version))

        raise TemplateNotFoundError(name)

    async def aload(self, template_name: str, version: str = None) -> PromptTemplate:
        """Public entry: resolve & cache a template by name."""
        return await self._resolve(template_name, version)
    
    # Backward compatibility alias
    async def load(self, template_name: str, version: str = None) -> PromptTemplate:
        """Deprecated: use aload() instead."""
        import warnings
        warnings.warn("load() is deprecated, use aload() instead", DeprecationWarning, stacklevel=2)
        return await self.aload(template_name, version)

    def get_model_config(self, model_name: str, provider: str = None) -> ModelConfig | None:
        """Get model configuration by name from loaded model loaders."""
        for loader in self._model_loaders:
            try:
                return loader.get_model_config(model_name, provider)
            except ModelConfigNotFoundError:
                continue
        return None


    def _merge_model_configs(self, input_cfg: ModelConfig | None, template_cfg: ModelConfig | None) -> ModelConfig:
        """Merge model configurations with priority: input_cfg > template_cfg > global_cfg.
        
        Args:
            input_cfg: Configuration passed to run method (highest priority)
            template_cfg: Configuration from template variant (medium priority)
            
        Returns:
            Merged ModelConfig with proper field precedence
            
        Raises:
            ValueError: If no valid configuration could be determined
        """
        # 如果有输入配置，优先使用它作为基础
        if input_cfg is not None:
            merged_cfg = input_cfg.model_copy()

            # 补充缺失的字段（从模板配置或全局配置获取）
            base_cfg = template_cfg or self._global_cfg
            if base_cfg is not None:
                for field_name, field_info in ModelConfig.model_fields.items():
                    src_value = getattr(merged_cfg, field_name)
                    if src_value is None or (isinstance(src_value, str) and src_value == ""):
                        base_value = getattr(base_cfg, field_name)
                        if base_value is not None:
                            setattr(merged_cfg, field_name, base_value)
        else:
            # 没有输入配置，使用模板配置或全局配置
            base_cfg = template_cfg or self._global_cfg
            if base_cfg is None:
                raise ValueError("ModelConfig required but not provided in template or globally")

            merged_cfg = base_cfg.model_copy()

        # 从注册表获取API配置（如果需要）
        if merged_cfg.model:
            registry_cfg = self.get_model_config(merged_cfg.model)
            if registry_cfg is not None:
                # 只补充API相关字段（如果缺失）
                api_fields = ["api_key", "api_url"]
                for field in api_fields:
                    if getattr(merged_cfg, field) is None and getattr(registry_cfg, field) is not None:
                        setattr(merged_cfg, field, getattr(registry_cfg, field))

        # 确保配置完整性
        if not merged_cfg.provider:
            raise ValueError("Provider is required in model configuration")
        if not merged_cfg.model:
            raise ValueError("Model name is required in model configuration")

        return merged_cfg

    def list_available_models(self) -> list[str]:
        """List all available models from all model loaders."""
        models = []
        for loader in self._model_loaders:
            models.extend(loader.list_models())
        return list(set(models))  # Remove duplicates

    def load_model_configs(self):
        """Load all model configurations from model loaders."""
        for loader in self._model_loaders:
            loader.load()

    async def aformat(
        self,
        template_name: str,
        variables: dict[str, Any],
        *,
        variant: str | None = None,
        version: str | None = None,
        selector: dict[str, Any] | None = None
    ) -> list[Message] | list[dict]:
        """Return formatted messages for ``template_name`` in ``format``."""
        tmpl = await self._resolve(template_name, version)
        msgs, _ = tmpl.format(
            variables,
            variant=variant,
            selector=selector
        )
        return msgs

    async def acompletion(
        self,
        template_name: str,
        variables: dict[str, Any],
        model_cfg: ModelConfig | dict[str, Any] | None = None,
        *,
        version: str | None = None,
        variant: str | None = None,
        ctx: dict[str, Any] | None = None,
        tool_params: ToolParams | list[ToolSpec] | list[dict] | dict[str, Any] | None = None,
        messages: list[Message] | list[dict] | None = None,
        template: PromptTemplate | None = None,
        **run_params: Any,
    ) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Stream messages produced by running the template.
        
        Args:
            template_name: Name of the template to run
            variables: Variables to use for template rendering
            model_cfg: Optional ModelConfig to use or override template's config.
                      Can be a ModelConfig object or a dict with config parameters.
                      If None, will use the template's model configuration.
                      If provided, will merge with template's config (with this taking precedence).
            variant: Optional variant name to use (otherwise auto-selected)
            ctx: Optional context for variant selection (defaults to variables)
            tool_params: Optional tool parameters for model calls.
                        Can be a ToolParams object, list of ToolSpec objects, 
                        list of dicts (converted to ToolSpec), or dict with ToolParams fields.
            messages: Optional direct messages to use instead of template.
                     Can be a list of Message objects or list of dicts (converted to Message).
                     If provided, template_name and variables will be ignored.
            **run_params: Additional parameters passed to model run
            
        Returns:
            AsyncGenerator yielding ModelResponse or StreamingResponse objects
        """
        # Convert basic types to objects
        converted_model_cfg = self._convert_model_cfg(model_cfg) if model_cfg is not None else None
        converted_tool_params = self._convert_tool_params(tool_params) if tool_params is not None else None
        converted_messages = self._convert_messages(messages) if messages is not None else None

        # 如果直接提供了messages，则使用提供的messages，否则使用模板解析
        tmpl_name = template_name
        tmpl = await self._resolve(template_name, version) if template is None else template
        if converted_messages is not None:
            # 直接使用提供的messages
            params = RunParams(messages=converted_messages, tool_params=converted_tool_params, **run_params)
            _, var = tmpl.format(
                variables,
                variant=variant,
                selector=ctx,
            )
        else:
            # 使用模板解析
            ctx = ctx or variables

            if variant is None:
                variant = tmpl.choose_variant(ctx) or next(iter(tmpl.variants))

            messages, var = tmpl.format(
                variables,
                variant=variant,
                selector=ctx,
            )
            params = RunParams(messages=cast(list[Message], messages), tool_params=converted_tool_params, **run_params)

        # 设置跟踪属性
        span_attrs = {
            "template.name": tmpl_name,
        }

        if var is not None:
            # 只有使用模板时才有这些属性
            span_attrs["template.version"] = getattr(var, "version", None) or ""
            span_attrs["variant"] = variant or ""

        with _tracer.start_as_current_span(
            "prompt.run",
            attributes=span_attrs,
        ):
            # 合并配置：传入的model_cfg > 模板的var.model_cfg > 全局配置
            template_cfg = var.model_cfg if var is not None else None

            cfg = self._merge_model_configs(input_cfg=converted_model_cfg, template_cfg=template_cfg)
            # 创建model client
            model_client = create_client(cfg)

            # 记录开始时间用于计算请求持续时间
            start_time = time.time()
            responses = []
            event = TraceEvent(
                template_name=tmpl_name if tmpl else "",
                template_version=tmpl.version if tmpl else "",
                template_id=tmpl.id if tmpl else "",
                variant=variant,
                model=cfg.model,
                messages_template=var.messages if var else "",
                timestamp=start_time,
                variables=variables,
                user_id=params.user_id,
                request_id=params.request_id,
                conversation_id=params.session_id,
                span_id=params.span_id,
                parent_span_id=params.parent_span_id,
                source=params.source,
                ext=params.extra_params,
            )
            try:
                async for response in model_client.arun(params):
                    # 收集所有响应用于trace上报
                    yield response
                    responses.append(response.model_dump(exclude_none=True))

                # 流式响应结束后，仅上报一次完整的trace数据
                if self._trace_service and responses:
                    # 从model_client传递过来的数据填充到事件中
                    if hasattr(params, "trace_context") and params.trace_context:
                        # 请求体
                        if "llm_request" in params.trace_context:
                            event.llm_request_body = params.trace_context["llm_request"]

                        # 响应体
                        if "llm_response_body" in params.trace_context:
                            event.llm_response_body = responses

                    event.perf_metrics = params.trace_context["perf_metrics"]
                    event.token_usage = responses[-1].get("usage", {})
                    if "error" in responses[-1]:
                        event.error = json.dumps(response.error, ensure_ascii=False)
                    # 添加额外的元数据
                    if run_params.get("metadata"):
                        event.ext = run_params.get("metadata", {})

                    # 上报一次trace事件
                    await self._trace_service.areport(event)
            except Exception as e:
                # 如果启用了trace服务，报告错误
                if self._trace_service:
                    # 构建错误trace事件
                    event.error = str(e)

                    # 从params中获取trace_context数据（如果存在）
                    if hasattr(params, "trace_context") and params.trace_context:
                        if "llm_request" in params.trace_context:
                            event.llm_request_body = params.trace_context["llm_request"]
                    else:
                        # 如果没有trace_context，则创建基本请求信息
                        event.llm_request_body = {
                            "model": getattr(cfg, "model", ""),
                            "messages": [msg.model_dump() for msg in params.messages],
                            "tools": tool_params
                        }

                    # 添加额外的元数据
                    if run_params.get("metadata"):
                        event.ext = run_params.get("metadata", {})

                    # 上报错误事件
                    await self._trace_service.areport(event)

                # 重新抛出异常
                raise
            finally:
                # 确保关闭客户端连接，防止出现未关闭client_session的警告
                await model_client.aclose()

    def completion(
        self,
        template_name: str,
        variables: dict[str, Any],
        model_cfg: ModelConfig | dict[str, Any] | None = None,
        *,
        version: str | None = None,
        variant: str | None = None,
        ctx: dict[str, Any] | None = None,
        tool_params: ToolParams | list[ToolSpec] | list[dict] | dict[str, Any] | None = None,
        messages: list[Message] | list[dict] | None = None,
        template: PromptTemplate | None = None,
        **run_params: Any,
    ) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Synchronous version: Stream messages produced by running the template.
        
        Args:
            template_name: Name of the template to run
            variables: Variables to use for template rendering
            model_cfg: Optional ModelConfig to use or override template's config
            variant: Optional variant name to use (otherwise auto-selected)
            ctx: Optional context for variant selection (defaults to variables)
            tool_params: Optional tool parameters for model calls
            messages: Optional direct messages to use instead of template
            **run_params: Additional parameters passed to model run
            
        Returns:
            Generator yielding ModelResponse or StreamingResponse objects
        """
        # Convert basic types to objects
        converted_model_cfg = self._convert_model_cfg(model_cfg) if model_cfg is not None else None
        converted_tool_params = self._convert_tool_params(tool_params) if tool_params is not None else None
        converted_messages = self._convert_messages(messages) if messages is not None else None

        # Resolve template synchronously
        tmpl_name = template_name
        tmpl = self._sync_resolve(template_name, version) if template is None else template
        if converted_messages is not None:
            params = RunParams(messages=converted_messages, tool_params=converted_tool_params, **run_params)
            _, var = tmpl.format(
                variables,
                variant=variant,
                selector=ctx,
            )
        else:
            ctx = ctx or variables

            if variant is None:
                try:
                    variant = tmpl.choose_variant(ctx) or next(iter(tmpl.variants))
                except StopIteration:
                    raise ValueError(f"Template {template_name} has no variants")

            messages, var = tmpl.format(
                variables,
                variant=variant,
                selector=ctx,
            )
            params = RunParams(messages=cast(list[Message], messages), tool_params=converted_tool_params, **run_params)

        # Set trace attributes
        span_attrs = {
            "template.name": tmpl_name,
        }

        if var is not None:
            span_attrs["template.version"] = getattr(var, "version", None) or ""
            span_attrs["variant"] = variant or ""

        with _tracer.start_as_current_span(
            "prompt.run",
            attributes=span_attrs,
        ):
            # Merge configurations
            template_cfg = var.model_cfg if var is not None else None
            cfg = self._merge_model_configs(input_cfg=converted_model_cfg, template_cfg=template_cfg)
            
            # Create sync model client
            from .model_client.factory import create_sync_client
            model_client = create_sync_client(cfg)

            # Record start time for request duration calculation
            start_time = time.time()
            responses = []
            event = TraceEvent(
                template_name=tmpl_name if tmpl else "",
                template_version=tmpl.version if tmpl else "",
                template_id=tmpl.id if tmpl else "",
                variant=variant,
                model=cfg.model,
                messages_template=var.messages if var else "",
                timestamp=start_time,
                variables=variables,
                user_id=params.user_id,
                request_id=params.request_id,
                conversation_id=params.session_id,
                parent_span_id=params.parent_span_id,
                span_id=params.span_id,
                source=params.source,
                ext=params.extra_params,
            )
            try:
                for response in model_client.run(params):
                    yield response
                    responses.append(response.model_dump(exclude_none=True))

                # Report trace data after streaming response ends
                if self._trace_service and responses:
                    if hasattr(params, "trace_context") and params.trace_context:
                        if "llm_request" in params.trace_context:
                            event.llm_request_body = params.trace_context["llm_request"]
                        if "llm_response_body" in params.trace_context:
                            event.llm_response_body = responses

                    event.perf_metrics = params.trace_context["perf_metrics"]
                    event.token_usage = responses[-1].get("usage", {})
                    if "error" in responses[-1]:
                        event.error = json.dumps(response.error, ensure_ascii=False)
                    if run_params.get("metadata"):
                        event.ext = run_params.get("metadata", {})

                    # Use synchronous report method for sync completion
                    self._trace_service.report(event)
                        
            except Exception as e:
                # Handle errors in sync context
                if self._trace_service:
                    event.error = str(e)
                    if hasattr(params, "trace_context") and params.trace_context:
                        if "llm_request" in params.trace_context:
                            event.llm_request_body = params.trace_context["llm_request"]
                    else:
                        event.llm_request_body = {
                            "model": getattr(cfg, "model", ""),
                            "messages": [msg.model_dump() for msg in params.messages],
                            "tools": tool_params
                        }

                    if run_params.get("metadata"):
                        event.ext = run_params.get("metadata", {})

                    # Use synchronous report method for sync completion
                    self._trace_service.report(event)

                raise
            finally:
                # Close client connection
                model_client.close()

    def _convert_model_cfg(self, model_cfg: ModelConfig | dict[str, Any]) -> ModelConfig:
        """Convert dict to ModelConfig object if needed."""
        if isinstance(model_cfg, dict):
            return ModelConfig(**model_cfg)
        return model_cfg

    def _convert_tool_params(self,
                             tool_params: ToolParams | list[ToolSpec] | list[dict] | dict[str, Any]) -> ToolParams:
        """Convert dict/list to ToolParams object if needed."""
        if isinstance(tool_params, dict):
            # If it's a dict, assume it contains ToolParams fields
            tools = tool_params.get('tools', [])
            if isinstance(tools, list) and tools:
                # Convert tool dicts to ToolSpec objects
                converted_tools = []
                for tool in tools:
                    if isinstance(tool, dict):
                        converted_tools.append(ToolSpec(**tool))
                    else:
                        converted_tools.append(tool)
                tool_params = tool_params.copy()
                tool_params['tools'] = converted_tools
            return ToolParams(**tool_params)
        elif isinstance(tool_params, list):
            # Convert list of dicts/ToolSpecs to ToolParams
            converted_tools = []
            for tool in tool_params:
                if isinstance(tool, dict):
                    converted_tools.append(ToolSpec(**tool))
                else:
                    converted_tools.append(tool)
            return ToolParams(tools=converted_tools)
        return tool_params

    def _convert_messages(self, messages: list[Message] | list[dict]) -> list[Message]:
        """Convert list of dicts to list of Message objects if needed."""
        if messages and isinstance(messages[0], dict):
            # Convert list of dicts to Message objects
            return [Message(**msg) if isinstance(msg, dict) else msg for msg in messages]
        return messages

    async def aclose(self):
        """Close all resources including trace service."""
        if self._trace_service:
            await self._trace_service.aclose()
    
    # Backward compatibility alias
    async def close(self):
        """Deprecated: use aclose() instead."""
        import warnings
        warnings.warn("close() is deprecated, use aclose() instead", DeprecationWarning, stacklevel=2)
        await self.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()

    @classmethod
    def from_setting(cls, setting: Setting) -> PromptEngine:
        """Create an engine instance from a :class:`Setting`.
        
        Args:
            setting: Setting instance. If None, will load from default config file.
            
        Returns:
            Configured PromptEngine instance
        """
        # 如果未提供setting，从配置文件加载
        if setting is None:
            raise ValueError("No setting provided")

        # 创建prompt loaders
        prompt_loaders: list[TemplateLoader] = [FileSystemLoader(Path(p)) for p in setting.template_paths]
        if setting.registry_url:
            prompt_loaders.append(HTTPLoader(base_url=setting.registry_url, auth_token=setting.registry_api_key))
        if setting.memory_templates:
            prompt_loaders.append(MemoryLoader(setting.memory_templates))
        # if setting.config_loader:
        #     prompt_loaders.append(setting.config_loader)

        # 创建model loaders
        model_loaders: list[ModelConfigLoader] = []
        # 按照优先级顺序遍历model_loaders，并尝试加载每个loader中的模型配置
        if setting.memory_model_configs:
            model_list = setting.memory_model_configs.get("model_list", [])
            token_list = setting.memory_model_configs.get("token_list", [])
            model_loaders.append(MemoryModelConfigLoader(model_list=model_list, token_list=token_list))
        if setting.registry_url and setting.registry_api_key:
            model_loaders.append(
                HTTPModelConfigLoader(
                    url=setting.registry_url, 
                    registry_api_key=setting.registry_api_key,
                    reload_interval=setting.model_cache_ttl
                ))
        if setting.model_config_path:
            model_loaders.append(FileModelConfigLoader(setting.model_config_path))

        # 确定global config
        global_cfg = setting.default_model_config
        if setting.global_config_loader:
            global_cfg = setting.global_config_loader.load()

        # 创建trace服务（如果配置了）
        trace_service = None
        if getattr(setting, "registry_url", None):
            trace_service = TraceService(
                endpoint_url=setting.registry_url,
            )

        # 创建引擎实例
        engine = cls(
            prompt_loaders=prompt_loaders,
            model_loaders=model_loaders,
            cache_ttl=setting.cache_ttl,
            global_model_config=global_cfg,
            trace_service=trace_service
        )

        # 加载所有模型配置
        engine.load_model_configs()

        return engine


class Setting(BaseModel):
    """Configuration options for :class:`PromptEngine`.

    Configuration can be loaded from a YAML file with the from_file method.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    template_paths: list[Path] = []
    model_config_path: Path | None = None
    cache_ttl: int = 300
    model_cache_ttl: int = 300
    registry_url: str | None = None
    registry_api_key: str | None = None
    memory_templates: dict[str, dict[str, str]] | None = None
    memory_model_configs: dict[str, Any] | None = None
    config_loader: TemplateLoader | None = None
    global_config_loader: ModelConfigLoader | None = None
    default_model_config: ModelConfig | None = None
    model_config_loaders: list[ModelConfigLoader] | None = None

    @classmethod
    def from_file(cls, file_path: str | None = None) -> "Setting":
        """Load settings from a YAML configuration file.
        
        Args:
            file_path: Path to the configuration file. If None, will try default locations.
            
        Returns:
            Loaded Setting instance
            
        Raises:
            FileNotFoundError: If no configuration file could be found
        """
        # 如果未指定文件路径，尝试默认路径
        if file_path is None:
            raise FileNotFoundError(f"No configuration file found: {file_path}")

        # 从文件加载配置
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)

        # 处理Path类型字段
        if "template_paths" in config_data and isinstance(config_data["template_paths"], list):
            config_data["template_paths"] = [Path(p) for p in config_data["template_paths"]]

        if "model_config_path" in config_data and isinstance(config_data["model_config_path"], str):
            config_data["model_config_path"] = Path(config_data["model_config_path"])

        # 处理ModelConfig字段
        if "default_model_config" in config_data and config_data["default_model_config"]:
            config_data["default_model_config"] = ModelConfig(**config_data["default_model_config"])

        return cls(**config_data)
