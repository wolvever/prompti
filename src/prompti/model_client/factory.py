"""Factory for constructing model client implementations."""

import pkgutil
import importlib
import inspect
from typing import Type, Dict, Any
from .base import ModelClient, SyncModelClient
import httpx

_CLIENT_CLASS_REGISTRY: Dict[str, Type[ModelClient]] = {}
_SYNC_CLIENT_CLASS_REGISTRY: Dict[str, Type[SyncModelClient]] = {}
_IS_REGISTRY_INITIALIZED = False


def _initialize_client_registry():
    """扫描本模块目录下所有 ModelClient 子类，并注册到全局字典中"""
    global _IS_REGISTRY_INITIALIZED
    if _IS_REGISTRY_INITIALIZED:
        return

    from . import __path__ as model_client_pkg_path, __name__ as model_client_pkg_name

    for finder, module_name, _ in pkgutil.iter_modules(model_client_pkg_path):
        module = importlib.import_module(f"{model_client_pkg_name}.{module_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, ModelClient) and obj is not ModelClient:
                name = getattr(obj, "provider", None)
                if name:
                    _CLIENT_CLASS_REGISTRY[name] = obj
            elif issubclass(obj, SyncModelClient) and obj is not SyncModelClient:
                name = getattr(obj, "provider", None)
                if name:
                    _SYNC_CLIENT_CLASS_REGISTRY[name] = obj

    _IS_REGISTRY_INITIALIZED = True


def create_client(cfg, *, is_debug: bool = False, **httpx_kw: Any):
    """基于 cfg.provider 从注册表中创建 ModelClient 实例"""
    _initialize_client_registry()

    cls = _CLIENT_CLASS_REGISTRY.get(cfg.provider)
    if not cls:
        raise ValueError(f"Unsupported provider: {cfg.provider}")

    client = httpx.AsyncClient(http2=True, **httpx_kw) if httpx_kw else None
    return cls(cfg, client=client, is_debug=is_debug)


def create_sync_client(cfg, *, is_debug: bool = False, **httpx_kw: Any):
    """基于 cfg.provider 从注册表中创建 SyncModelClient 实例"""
    _initialize_client_registry()

    cls = _SYNC_CLIENT_CLASS_REGISTRY.get(cfg.provider)
    if not cls:
        raise ValueError(f"Unsupported sync provider: {cfg.provider}")

    client = httpx.Client(http2=True, **httpx_kw) if httpx_kw else None
    return cls(cfg, client=client, is_debug=is_debug)
