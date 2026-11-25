"""文本规范化与分词。数据管线、BM25、切片规则共用，避免各处口径不一致。"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_query(query: str) -> str:
    """规范化 query：小写、去首尾空白、合并连续空白（specs/data.md §6.4）。"""
    return " ".join(query.lower().split())


def tokenize(text: str) -> list[str]:
    """BM25 与 Jaccard 使用的分词：小写后取 [a-z0-9]+。"""
    return _TOKEN_RE.findall(text.lower())


def truncate_words(text: str, n: int) -> str:
    """按空白分隔的词截断到 n 个；被截断时末尾加 ' ...'。

    env_v1 不依赖具体分词器，这里的“token”就是空白分隔的词（specs/training.md §2.1）。
    """
    words = text.split()
    if len(words) <= n:
        return " ".join(words)
    return " ".join(words[:n]) + " ..."
