"""按 query 配对的 bootstrap（specs/eval.md §6）。

只用标准库的 ``random.Random(seed).random()``，其序列在各平台、各 Python 版本上一致，
因此同一份输入、同一个 seed 的结果逐位相同。
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from searchtune.config import SEED


@dataclass(frozen=True)
class CI:
    metric: str
    n_queries: int
    n_boot: int
    seed: int
    mean_a: float
    mean_b: float
    diff: float  # a - b
    lo: float
    hi: float
    a_significantly_better: bool  # 区间下界 > 0

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap(
    a: Mapping[str, float | None],
    b: Mapping[str, float | None],
    metric: str = "ndcg",
    n: int = 1000,
    seed: int = SEED,
) -> CI:
    """a、b：query_id → 单 query 指标值。只在两边都有定义的 query 上比较。

    resample 的是 query（成对抽取），统计量为 mean(a) - mean(b)，取 2.5% 与 97.5% 分位数。
    """
    if set(a) != set(b):
        raise ValueError("两个系统覆盖的 query 不一致，无法配对")
    qids = sorted(q for q in a if a[q] is not None and b[q] is not None)
    if not qids:
        raise ValueError("没有可配对的 query")
    va = [float(a[q]) for q in qids]  # type: ignore[arg-type]
    vb = [float(b[q]) for q in qids]  # type: ignore[arg-type]
    d = [x - y for x, y in zip(va, vb)]
    m = len(d)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = 0.0
        for _ in range(m):
            s += d[int(rng.random() * m)]
        means.append(s / m)
    means.sort()
    lo = means[int(0.025 * (n - 1))]
    hi = means[math.ceil(0.975 * (n - 1))]
    return CI(
        metric=metric,
        n_queries=m,
        n_boot=n,
        seed=seed,
        mean_a=sum(va) / m,
        mean_b=sum(vb) / m,
        diff=sum(d) / m,
        lo=lo,
        hi=hi,
        a_significantly_better=lo > 0,
    )
