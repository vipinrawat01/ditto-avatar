###############################################################################
#  Plugin registry — register via decorators, create instances by name
###############################################################################

from typing import Dict, Type, Any
from utils.logger import logger

_REGISTRY: Dict[str, Dict[str, Type]] = {
    "stt": {},
    "llm": {},
    "tts": {},
    "avatar": {},
    "output": {},
}


def register(category: str, name: str):
    """
    Decorator: register a plugin class into the global registry.

    Usage::

        @register("tts", "edgetts")
        class EdgeTTS(BaseTTS): ...
    """
    def decorator(cls):
        if category not in _REGISTRY:
            _REGISTRY[category] = {}
        _REGISTRY[category][name] = cls
        logger.info(f"Registered {category}/{name}: {cls.__name__}")
        return cls
    return decorator


def create(category: str, name: str, **kwargs) -> Any:
    """
    Create a plugin instance by name.

    Usage::

        tts = registry.create("tts", "edgetts", opt=opt)
    """
    if category not in _REGISTRY or name not in _REGISTRY[category]:
        available = list(_REGISTRY.get(category, {}).keys())
        raise ValueError(
            f"Plugin '{name}' not found in category '{category}'. "
            f"Available: {available}"
        )
    cls = _REGISTRY[category][name]
    return cls(**kwargs)


def list_plugins(category: str = None) -> Dict[str, list]:
    """List the registered plugins"""
    if category:
        return {category: list(_REGISTRY.get(category, {}).keys())}
    return {cat: list(plugins.keys()) for cat, plugins in _REGISTRY.items()}
