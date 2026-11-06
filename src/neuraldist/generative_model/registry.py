from typing import Callable

_REGISTRY: dict[str, type] = {}

def register_model(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls
    return deco

def build_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available models: {list_models()}")
    return _REGISTRY[name](**kwargs)

def list_models() -> list[str]:
    return list(_REGISTRY.keys())