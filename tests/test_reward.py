import math

import pytest

from searchtune.config import RewardConfig
from searchtune.data.pool import GAIN, Catalog, Pool, PoolItem, Product
from searchtune.env import reward as reward_mod
from searchtune.env.episode import Episode, EpisodeResult
from searchtune.env.reward import reward_l1, reward_l2
from searchtune.eval import metrics

POOL = Pool(
    "q1", "x", "dev",
    tuple(PoolItem(p, lab, GAIN[lab], "labeled") for p, lab in
          [("a", "E"), ("b", "S"), ("c", "I"), ("d", "I"), ("e", "E"), ("f", "C")]),
    "v1",
)


def res(submitted, status="ok", n_search=1, n_view=0):
    return EpisodeResult(submitted=submitted, status=status, n_search=n_search, n_view=n_view)


def test_perfect_submission_gets_ndcg_one():
    assert reward_l1(res(["a", "e", "b", "f"]), POOL) == pytest.approx(1.0)
    assert reward_l2(res(["a", "e", "b", "f"]), POOL) == pytest.approx(1.0)


def test_empty_submission():
    assert reward_l1(res([]), POOL) == 0.0
    assert reward_l2(res([]), POOL) == 0.0  # ok 但没提交任何商品：NDCG 为 0，没有惩罚项


def test_bad_example_penalty_counts_I_in_top3_only():
    base = metrics.ndcg_at_10(["a", "c", "b"], POOL)
    assert reward_l2(res(["a", "c", "b"]), POOL) == pytest.approx(base - 0.3 * 1 / 3)
    two = metrics.ndcg_at_10(["c", "d", "a"], POOL)
    assert reward_l2(res(["c", "d", "a"]), POOL) == pytest.approx(two - 0.3 * 2 / 3)
    # I 在第 4 位不扣
    r4 = res(["a", "e", "b", "c"])
    assert reward_l2(r4, POOL) == pytest.approx(metrics.ndcg_at_10(["a", "e", "b", "c"], POOL))


def test_call_cost_only_after_free_calls():
    sub = ["a", "e", "b", "f"]
    assert reward_l2(res(sub, n_search=3, n_view=1), POOL) == pytest.approx(1.0)  # 4 次免费
    assert reward_l2(res(sub, n_search=3, n_view=3), POOL) == pytest.approx(1.0 - 0.02 * 2)


@pytest.mark.parametrize("status", ["format_error", "budget_exhausted", "truncated"])
def test_non_ok_status_is_flat_penalty(status):
    r = res(["a", "e"], status=status)  # 即使带了好的提交也不算
    assert reward_l1(r, POOL) == -0.2
    assert reward_l2(r, POOL) == -0.2


def test_e5_ablation_keeps_only_ndcg():
    cfg = RewardConfig(bad_weight=0.0, call_cost=0.0)
    r = res(["c", "d", "a"], n_search=6, n_view=3)
    assert reward_l2(r, POOL, cfg) == pytest.approx(metrics.ndcg_at_10(["c", "d", "a"], POOL))


def test_reward_uses_the_scorers_ndcg():
    """training §10：奖励与评测打分器共用同一份 NDCG 实现。"""
    assert reward_mod.ndcg_at_10 is metrics.ndcg_at_10


def test_reward_matches_scorer_on_real_episode():
    prods = [Product(p.product_id, f"red shoe {p.product_id}", None, None, None, None) for p in POOL.items]
    ep = Episode(POOL, Catalog(prods))
    ep.search("red shoe")
    ep.submit(["b", "a", "zzz", "c"])
    r = ep.result
    per_query = metrics.episode_metrics({"query_id": "q1", "submitted": r.submitted, "status": r.status}, POOL).ndcg
    cfg = RewardConfig(bad_weight=0.0, call_cost=0.0)
    assert reward_l2(r, POOL, cfg) == per_query == pytest.approx(
        (0.1 + 1 / math.log2(3) + 0.0) / (1 + 1 / math.log2(3) + 0.1 / math.log2(4) + 0.01 / math.log2(5))
    )
