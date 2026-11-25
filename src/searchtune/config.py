"""配置：YAML → dataclass，未知字段直接报错（DESIGN.md §5）。

同时集中定义跨模块共享的配置类。环境版本变化必须升 ``ENV_VERSION``（specs/training.md §2）。
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

ENV_VERSION = "env_v1"
SEED = 20261031

T = TypeVar("T")


class ConfigError(ValueError):
    """配置文件有未知字段、类型错误或无法解析。"""


@dataclasses.dataclass
class EnvConfig:
    """env_v1 的全部参数；锁定后不可改（改了就升 ENV_VERSION）。"""

    max_tool_calls: int = 6  # search + view 合计
    search_top_k: int = 10
    search_title_tokens: int = 32
    view_bullet_tokens: int = 200
    submit_top_k: int = 10
    max_gen_tokens_per_turn: int = 512  # 由 rollout 驱动方执行
    max_context_tokens: int = 6144  # 由 rollout 驱动方执行


@dataclasses.dataclass
class RewardConfig:
    """L2 奖励 v1（specs/training.md §4.2）。E5 对照把 bad_weight、call_cost 置 0。"""

    non_ok_reward: float = -0.2
    bad_weight: float = 0.3
    bad_top_n: int = 3
    free_calls: int = 4
    call_cost: float = 0.02


@dataclasses.dataclass
class DataConfig:
    """数据版本参数（specs/data.md §4、§5）。任一项变化都要升 data_version。"""

    data_version: str = "v1"
    seed: int = SEED
    locale: str = "us"
    n_test: int = 1000
    n_dev: int = 500
    n_dev_quick: int = 200
    n_rl_l1: int = 8000
    n_sft_traj: int = 4000
    n_rl_l2: int = 3000
    n_bm25_distractors: int = 50
    n_random_distractors: int = 50
    jaccard_threshold: float = 0.5
    brand_min_count: int = 50  # 切片“指定品牌”：品牌在目录中至少出现的商品数（太小会混入 without、not 之类的伪品牌）
    long_query_words: int = 6  # 切片“长 query”


def _check_type(value: Any, tp: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        for a in typing.get_args(tp):
            try:
                return _check_type(value, a, path)
            except ConfigError:
                continue
        raise ConfigError(f"{path}: 类型不符，期望 {tp}，实际 {type(value).__name__}")
    if tp is type(None):
        if value is None:
            return None
        raise ConfigError(f"{path}: 期望 null")
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: 期望 list")
        (inner,) = typing.get_args(tp)
        return [_check_type(v, inner, f"{path}[{i}]") for i, v in enumerate(value)]
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: 期望 mapping")
        return from_dict(tp, value, path)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is Any:
        return value
    elif isinstance(value, tp) and not (tp is int and isinstance(value, bool)):
        return value
    raise ConfigError(f"{path}: 类型不符，期望 {getattr(tp, '__name__', tp)}，实际 {type(value).__name__}")


def from_dict(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    """dict → dataclass；未知字段、类型不符都报错，缺失字段用默认值。"""
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: 期望 mapping")
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{path or cls.__name__}: 未知字段 {unknown}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _check_type(value, hints[name], f"{path}.{name}" if path else name)
    try:
        return cls(**kwargs)
    except TypeError as e:  # 缺少无默认值的必填字段
        raise ConfigError(f"{path or cls.__name__}: {e}") from e


def load_config(path: str | Path, cls: type[T]) -> T:
    """读取 YAML 并构造 ``cls``。空文件等价于全部使用默认值。"""
    with open(path, encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"{path}: YAML 解析失败: {e}") from e
    return from_dict(cls, raw or {})
