"""L1 / L2 奖励（specs/training.md §4）。NDCG 直接复用评测打分器的实现。"""

from __future__ import annotations

from searchtune.config import RewardConfig
from searchtune.data.pool import Pool
from searchtune.env.episode import EpisodeResult
from searchtune.eval.metrics import n_bad_in_top, ndcg_at_10

__all__ = ["reward_l1", "reward_l2", "ndcg_at_10"]


def reward_l1(result: EpisodeResult, pool: Pool, non_ok_reward: float = -0.2) -> float:
    """L1：提交列表的 NDCG@10；非 ok（格式错误等）为 -0.2。"""
    if result.status != "ok":
        return non_ok_reward
    return ndcg_at_10(result.submitted, pool)


def reward_l2(result: EpisodeResult, pool: Pool, cfg: RewardConfig | None = None) -> float:
    """L2 v1：NDCG@10 − bad_weight × (前 3 个中 I 的数量 / 3) − call_cost × max(0, 调用次数 − free_calls)。

    非 ok 状态直接为 ``non_ok_reward``。E5 对照：``bad_weight = call_cost = 0``。
    """
    cfg = cfg or RewardConfig()
    if result.status != "ok":
        return cfg.non_ok_reward
    r = ndcg_at_10(result.submitted, pool)
    r -= cfg.bad_weight * n_bad_in_top(result.submitted, pool, cfg.bad_top_n) / cfg.bad_top_n
    r -= cfg.call_cost * max(0, result.n_calls - cfg.free_calls)
    return r
