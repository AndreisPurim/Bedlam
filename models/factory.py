from typing import Tuple

from models import build_m1, build_m2, build_m3


def build_split_models(
    m1_name: str = "default",
    m2_name: str = "default",
    m3_name: str = "default",
    m1_kwargs: dict | None = None,
    m2_kwargs: dict | None = None,
    m3_kwargs: dict | None = None,
):
    """
    Convenience wrapper to build (M1, M2, M3) given model names and optional kwargs.
    """
    m1_kwargs = m1_kwargs or {}
    m2_kwargs = m2_kwargs or {}
    m3_kwargs = m3_kwargs or {}
    m1 = build_m1(model_name=m1_name, **m1_kwargs)
    m2 = build_m2(model_name=m2_name, **m2_kwargs)
    m3 = build_m3(model_name=m3_name, **m3_kwargs)
    return m1, m2, m3
