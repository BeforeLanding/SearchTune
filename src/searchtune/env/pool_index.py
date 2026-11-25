"""BM25 索引（倒排 + numpy 打分），结果确定：同分按 ``doc_id`` 排序。

同一份实现服务两处：单个商品池内的 ``search`` 工具（``PoolIndex``）和整个目录上的
难干扰商品检索（``data/pools.py``）。不用 ``rank-bm25``，原因见 DECISIONS.md DEC-015。
"""

from __future__ import annotations

import math
from array import array
from collections import Counter
from collections.abc import Iterable

import numpy as np

from searchtune.data.pool import Catalog, Pool
from searchtune.text import tokenize

K1 = 1.5
B = 0.75
_ROUND = 9  # 打分取整位数，避免浮点尾差破坏“同分按 id 排序”


class BM25Index:
    """文档 = (doc_id, 词列表)。"""

    def __init__(self, docs: Iterable[tuple[str, list[str]]], k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        lens: list[int] = []
        post_idx: dict[str, array] = {}
        post_tf: dict[str, array] = {}
        for i, (doc_id, tokens) in enumerate(docs):
            self.doc_ids.append(doc_id)
            lens.append(len(tokens))
            for term, tf in Counter(tokens).items():
                if term not in post_idx:
                    post_idx[term] = array("I")
                    post_tf[term] = array("H")
                post_idx[term].append(i)
                post_tf[term].append(min(tf, 65535))
        self.n = len(self.doc_ids)
        self.doc_len = np.asarray(lens, dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self._idx = {t: np.frombuffer(a, dtype=np.uint32) for t, a in post_idx.items()}
        self._tf = {t: np.frombuffer(a, dtype=np.uint16).astype(np.float64) for t, a in post_tf.items()}
        norm = 1.0 - b + b * self.doc_len / self.avgdl if self.avgdl > 0 else np.ones(self.n)
        self._len_norm = k1 * norm

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float64)
        for term in sorted(set(query_tokens)):  # 固定累加顺序，保证结果可复现
            idx = self._idx.get(term)
            if idx is None:
                continue
            tf = self._tf[term]
            df = len(idx)
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            s[idx] += idf * tf * (self.k1 + 1.0) / (tf + self._len_norm[idx])
        return np.round(s, _ROUND)

    def search(self, query_tokens: list[str], k: int) -> list[tuple[str, float]]:
        """返回前 k 个得分 > 0 的文档，按 (-score, doc_id) 排序；得分 0 的不返回。"""
        if self.n == 0 or not query_tokens or k <= 0:
            return []
        s = self.scores(query_tokens)
        cand = np.flatnonzero(s > 0)
        if len(cand) == 0:
            return []
        if len(cand) > k:
            kth = np.partition(s[cand], -k)[-k]
            cand = cand[s[cand] >= kth]  # 保留与第 k 名同分的全部文档，再按 id 决胜
        ranked = sorted(((-float(s[i]), self.doc_ids[i]) for i in cand))
        return [(doc_id, -neg) for neg, doc_id in ranked[:k]]


def product_text(title: str, brand: str | None, bullets: str | None) -> str:
    """池内 ``search`` 的索引文本：标题、品牌、卖点（specs/training.md §2.1）。"""
    return " ".join(x for x in (title, brand, bullets) if x)


class PoolIndex:
    """单个商品池上的 BM25。"""

    def __init__(self, pool: Pool, catalog: Catalog):
        docs = []
        for pid in pool.product_ids:
            p = catalog.get(pid)
            docs.append((pid, tokenize(product_text(p.title, p.brand, p.bullets))))
        self._index = BM25Index(docs)

    def search(self, query: str, k: int = 10) -> list[str]:
        return [pid for pid, _ in self._index.search(tokenize(query), k)]
