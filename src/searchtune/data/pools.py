"""商品池构造（specs/data.md §5）：L(q) + BM25 难干扰 + 随机干扰，并排除疑似相关商品。"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence

from searchtune.config import DataConfig
from searchtune.data.pool import GAIN, Pool, PoolItem
from searchtune.env.pool_index import BM25Index
from searchtune.text import tokenize


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def distractor_text(title: str, brand: str | None) -> str:
    """难干扰检索的索引文本：标题 + 品牌（data §5.1）。"""
    return f"{title} {brand}" if brand else title


def build_distractor_index(products: Iterable[tuple[str, str, str | None]]) -> BM25Index:
    """products：(product_id, title, brand)。"""
    return BM25Index((pid, tokenize(distractor_text(t, b))) for pid, t, b in products)


class PoolBuilder:
    """构造单个 query 的商品池。

    ``es_queries``：product_id → 把它标为 E 或 S 的 query_id 列表（来自全部清洗后的标注，
    用于 §5.2 的“疑似相关”排除）。``query_words``：query_id → 规范化词集合。
    """

    def __init__(
        self,
        index: BM25Index,
        es_queries: Mapping[str, Sequence[str]],
        query_words: Mapping[str, frozenset[str]],
        cfg: DataConfig,
    ):
        self.index = index
        self.es_queries = es_queries
        self.query_words = query_words
        self.cfg = cfg
        self.catalog_ids = sorted(index.doc_ids)

    def is_suspect(self, product_id: str, query_id: str) -> bool:
        """商品在另一个相近 query（Jaccard ≥ 阈值）下被标为 E/S → 疑似对本 query 也相关。"""
        qw = self.query_words[query_id]
        for other in self.es_queries.get(product_id, ()):
            if other != query_id and jaccard(qw, self.query_words[other]) >= self.cfg.jaccard_threshold:
                return True
        return False

    def build(
        self, query_id: str, query: str, split: str, labeled: Mapping[str, str]
    ) -> Pool:
        cfg = self.cfg
        excluded = set(labeled)
        items = [PoolItem(pid, lab, GAIN[lab], "labeled") for pid, lab in sorted(labeled.items())]

        # 难干扰：取候选，按需要加大 k，直到凑够或候选耗尽
        chosen: list[str] = []
        k = max(4 * cfg.n_bm25_distractors, 200)
        tokens = tokenize(query)
        while True:
            cands = self.index.search(tokens, k)
            chosen = []
            for pid, _ in cands:
                if pid in excluded or self.is_suspect(pid, query_id):
                    continue
                chosen.append(pid)
                if len(chosen) == cfg.n_bm25_distractors:
                    break
            if len(chosen) == cfg.n_bm25_distractors or len(cands) < k:
                break
            k *= 4
        items += [PoolItem(pid, "U", 0.0, "bm25") for pid in chosen]
        excluded.update(chosen)

        # 随机干扰：每个 query 一个独立的 RNG，结果与处理顺序无关
        rng = random.Random(f"{cfg.seed}:{query_id}")
        n_cat = len(self.catalog_ids)
        picked: list[str] = []
        attempts = 0
        while len(picked) < cfg.n_random_distractors and attempts < 50 * cfg.n_random_distractors + n_cat:
            attempts += 1
            pid = self.catalog_ids[int(rng.random() * n_cat)]
            if pid in excluded or self.is_suspect(pid, query_id):
                continue
            picked.append(pid)
            excluded.add(pid)
        items += [PoolItem(pid, "U", 0.0, "random") for pid in picked]
        return Pool(query_id, query, split, tuple(items), cfg.data_version)


def check_pool_rules(
    pool: Pool,
    es_queries: Mapping[str, Sequence[str]],
    query_words: Mapping[str, frozenset[str]],
    cfg: DataConfig,
) -> None:
    """独立于构造过程的检查（data §9）：干扰商品与 L(q) 不相交，并满足 Jaccard 排除规则。"""
    labeled = {it.product_id for it in pool.items if it.source == "labeled"}
    for it in pool.items:
        if it.source == "labeled":
            if it.label not in ("E", "S", "C", "I"):
                raise AssertionError(f"{pool.query_id}: 标注商品 {it.product_id} 标签非法 {it.label}")
            continue
        if it.label != "U" or it.gain != 0.0:
            raise AssertionError(f"{pool.query_id}: 干扰商品 {it.product_id} 必须是 U")
        if it.product_id in labeled:
            raise AssertionError(f"{pool.query_id}: 干扰商品 {it.product_id} 与 L(q) 相交")
        for other in es_queries.get(it.product_id, ()):
            if other != pool.query_id and jaccard(query_words[pool.query_id], query_words[other]) >= cfg.jaccard_threshold:
                raise AssertionError(
                    f"{pool.query_id}: 干扰商品 {it.product_id} 违反 Jaccard 排除规则（相似 query {other}）"
                )
    n_b = sum(1 for it in pool.items if it.source == "bm25")
    n_r = sum(1 for it in pool.items if it.source == "random")
    if n_b > cfg.n_bm25_distractors or n_r > cfg.n_random_distractors:
        raise AssertionError(f"{pool.query_id}: 干扰商品数量超出配置")
