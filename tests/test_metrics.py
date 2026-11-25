import math

import pytest

from searchtune.data.pool import GAIN, Pool, PoolItem
from searchtune.eval.metrics import (
    aggregate,
    bad_at_3,
    e_recall_at_10,
    episode_metrics,
    ndcg_at_10,
    normalize_submission,
    percentile,
    u_share_at_10,
)


def mk_pool(spec: dict[str, str], qid="q1") -> Pool:
    items = tuple(PoolItem(p, lab, GAIN[lab], "bm25" if lab == "U" else "labeled") for p, lab in spec.items())
    return Pool(qid, "x", "dev", items, "v1")


# 手算样例：池 a:E b:S c:C d:I e:U f:E，理想增益排序 [1, 1, .1, .01, 0, 0]
POOL = mk_pool({"a": "E", "b": "S", "c": "C", "d": "I", "e": "U", "f": "E"})
IDCG = 1 + 1 / math.log2(3) + 0.1 / math.log2(4) + 0.01 / math.log2(5)


def test_ndcg_perfect_is_one():
    assert ndcg_at_10(["a", "f", "b", "c", "d", "e"], POOL) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    # 提交 [b, a]：DCG = 0.1/log2(2) + 1/log2(3)
    expected = (0.1 + 1 / math.log2(3)) / IDCG
    assert ndcg_at_10(["b", "a"], POOL) == pytest.approx(expected)
    assert expected == pytest.approx(0.4337, abs=1e-4)  # 手算数值，防止公式与实现同时写错


def test_ndcg_short_submission_pads_with_zero():
    assert ndcg_at_10(["a"], POOL) == pytest.approx(1.0 / IDCG)


def test_ndcg_empty_and_unknown_ids_are_zero():
    assert ndcg_at_10([], POOL) == 0.0
    assert ndcg_at_10(["zzz"], POOL) == 0.0


def test_duplicates_and_unknown_ids_dropped_before_top10():
    ids, dropped = normalize_submission(["a", "a", "zzz", "b"], POOL)
    assert ids == ["a", "b"] and dropped == 2
    assert ndcg_at_10(["a", "a", "b"], POOL) == pytest.approx(ndcg_at_10(["a", "b"], POOL))


def test_only_first_ten_count():
    big = mk_pool({f"p{i}": "I" for i in range(12)} | {"gold": "E"})
    assert ndcg_at_10([f"p{i}" for i in range(10)] + ["gold"], big) == 0.0  # E 排第 11，不计分


def test_e_recall():
    assert e_recall_at_10(["a", "b"], POOL) == pytest.approx(1 / 2)  # 池里 2 个 E
    assert e_recall_at_10(["a", "f"], POOL) == pytest.approx(1.0)
    assert e_recall_at_10([], POOL) == 0.0
    no_e = mk_pool({"x": "S", "y": "I"})
    assert e_recall_at_10(["x"], no_e) is None  # 池中无 E：无定义


def test_bad_rate_counts_only_I_in_top3_not_U():
    assert bad_at_3(["a", "b", "d"], POOL) is True
    assert bad_at_3(["a", "b", "e", "d"], POOL) is False  # I 在第 4 位；U 不算坏例
    assert bad_at_3([], POOL) is False


def test_u_share():
    assert u_share_at_10(["a", "e"], POOL) == 0.5
    assert u_share_at_10([], POOL) is None


def test_non_ok_status_scored_as_empty_but_stays_in_denominator():
    ok_row = {"query_id": "q1", "submitted": ["a", "f"], "status": "ok"}
    for st in ("format_error", "budget_exhausted", "truncated"):
        # 即使带了完美的 submitted，非 ok 也不计分
        bad_row = {"query_id": "q2", "submitted": ["a", "f"], "status": st}
        assert episode_metrics(bad_row, POOL).ndcg == 0.0
        pools = {"q1": POOL, "q2": Pool("q2", "x", "dev", POOL.items, "v1")}
        agg = aggregate([ok_row, bad_row], pools)
        assert agg["n_queries"] == 2
        full = (1 + 1 / math.log2(3)) / IDCG  # [a, f] 的 NDCG，手算
        assert agg["ndcg_at_10"] == pytest.approx(full / 2)  # (full + 0) / 2：分母包含失败的 episode
        assert agg[f"{st}_rate"] == 0.5 and agg["ok_rate"] == 0.5


def test_unknown_status_rejected():
    with pytest.raises(ValueError):
        episode_metrics({"query_id": "q1", "submitted": [], "status": "weird"}, POOL)


def test_aggregate_full_metrics_and_slices():
    rows = [
        {"query_id": "q1", "submitted": ["a", "f"], "status": "ok", "n_search": 2, "n_view": 1, "gen_tokens": 100, "wall_time_s": 1.0},
        {"query_id": "q2", "submitted": ["d"], "status": "ok", "n_search": 1, "n_view": 0, "gen_tokens": 50, "wall_time_s": 3.0},
    ]
    pools = {"q1": POOL, "q2": Pool("q2", "x", "dev", POOL.items, "v1")}
    agg = aggregate(rows, pools, {"neg": {"q2"}})
    assert agg["bad_rate_at_3"] == 0.5
    assert agg["n_search_mean"] == 1.5 and agg["n_view_mean"] == 0.5 and agg["n_calls_mean"] == 2.0
    assert agg["gen_tokens_mean"] == 75 and agg["wall_time_s_p90"] == pytest.approx(2.8)
    assert agg["slices"]["neg"] == {"n": 1, "ndcg_at_10": 0.0}


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert math.isnan(percentile([], 50))
