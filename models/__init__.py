from typing import Callable, Any

from models.default_models import (
    build_m1_default,
    build_m2_default,
    build_m3_default,
    build_m1_mlp,
    build_m2_mlp,
    build_m3_mlp,
    build_m1_deep,
    build_m2_deep,
    build_m3_deep,
    build_m1_resnet_lite,
    build_m2_resnet_lite,
    build_m3_resnet_lite,
)
from models.tiny_cnn_models import (
    build_m1_tiny_cnn,
    build_m2_tiny_cnn,
    build_m3_tiny_cnn,
)


# Registries for model builders
M1_BUILDERS: dict[str, Callable[..., Any]] = {
    "default": build_m1_default,
    "mlp": build_m1_mlp,
    "deep": build_m1_deep,
    "resnet-lite": build_m1_resnet_lite,
    "tiny-cnn": build_m1_tiny_cnn,
}

M2_BUILDERS: dict[str, Callable[..., Any]] = {
    "default": build_m2_default,
    "mlp": build_m2_mlp,
    "deep": build_m2_deep,
    "resnet-lite": build_m2_resnet_lite,
    "tiny-cnn": build_m2_tiny_cnn,
}

M3_BUILDERS: dict[str, Callable[..., Any]] = {
    "default": build_m3_default,
    "mlp": build_m3_mlp,
    "deep": build_m3_deep,
    "resnet-lite": build_m3_resnet_lite,
    "tiny-cnn": build_m3_tiny_cnn,
}


def build_m1(model_name: str = "default", **kwargs):
    builder = M1_BUILDERS.get(model_name)
    if not builder:
        raise ValueError(f"Unknown M1 model '{model_name}'")
    return builder(**kwargs)


def build_m2(model_name: str = "default", **kwargs):
    builder = M2_BUILDERS.get(model_name)
    if not builder:
        raise ValueError(f"Unknown M2 model '{model_name}'")
    return builder(**kwargs)


def build_m3(model_name: str = "default", **kwargs):
    builder = M3_BUILDERS.get(model_name)
    if not builder:
        raise ValueError(f"Unknown M3 model '{model_name}'")
    return builder(**kwargs)
