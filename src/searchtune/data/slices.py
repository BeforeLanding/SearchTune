"""难例切片的确定性打标（specs/eval.md §2.3）。一个 query 可以属于多个切片。

规则刻意保持简单、可读；它们只是分组依据，不追求语法上的完备（边界情况会有噪声，
各切片的规则与成员数写进 ``data/STATS.md``）。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping

from searchtune.text import tokenize

SLICES = ("negation", "quantity_spec", "compat_model", "brand", "long")

SLICE_RULES: dict[str, str] = {
    "negation": "query 词中含 without / no / free / non / not / excluding / except 之一（连字符拆词，如 sugar-free）",
    "quantity_spec": "含“数字 + 单位”（pack、count、oz、lb、inch、gb、v、w 等）、`数字x数字`，或含 pack / size / xl / xxl / small / medium / large",
    "compat_model": "含 for 且其后任一词含数字；或含字母数字混合的型号串（长度 ≥ 3，如 a15、ps5、x570）",
    "brand": "query 中连续 1–3 个词等于目录中某个品牌（该品牌至少对应 brand_min_count 个商品）；单词品牌可能与普通词重名，存在噪声",
    "long": "词数 ≥ long_query_words（默认 6）",
}

_NEGATION = {"without", "no", "free", "non", "not", "excluding", "except"}
_UNITS = (
    "pack|pk|ct|count|pcs|pc|piece|pieces|oz|ounce|ounces|lb|lbs|pound|pounds|kg|g|mg|ml|l|liter|liters|"
    "gal|gallon|gallons|inch|inches|in|cm|mm|ft|feet|foot|yd|gb|tb|mb|v|volt|volts|w|watt|watts|mah|ah|"
    "quart|qt|pint|pt|m"
)
_QTY_RE = re.compile(rf"(?<![a-z0-9])\d+(\.\d+)?\s*(?:{_UNITS})(?![a-z0-9])")
_DIM_RE = re.compile(r"(?<![a-z0-9])\d+(\.\d+)?\s*x\s*\d+")
_SIZE_WORDS = {"pack", "size", "xs", "xl", "xxl", "xxxl", "small", "medium", "large"}
_MODEL_RE = re.compile(r"(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]{3,}")
_BRAND_STOP = {"generic", "unknown", "other", "none", "na", "brand", "unbranded", "the", "and"}


def build_brand_set(brands: Iterable[str | None], min_count: int) -> set[tuple[str, ...]]:
    """目录中的品牌（分词后 1–3 个词）；出现次数不足 min_count 或属于泛词的丢弃。"""
    cnt: Counter[tuple[str, ...]] = Counter()
    for b in brands:
        if not b:
            continue
        toks = tuple(tokenize(b))
        if 1 <= len(toks) <= 3 and sum(map(len, toks)) >= 3 and not (len(toks) == 1 and toks[0] in _BRAND_STOP):
            cnt[toks] += 1
    return {b for b, c in cnt.items() if c >= min_count}


def assign_slices(query: str, brand_set: set[tuple[str, ...]], long_words: int = 6) -> list[str]:
    q = query.lower()
    toks = tokenize(q)
    out = []
    if _NEGATION & set(toks):
        out.append("negation")
    if _QTY_RE.search(q) or _DIM_RE.search(q) or _SIZE_WORDS & set(toks):
        out.append("quantity_spec")
    if ("for" in toks and any(any(c.isdigit() for c in t) for t in toks[toks.index("for") + 1 :])) or any(
        _MODEL_RE.fullmatch(t) for t in toks
    ):
        out.append("compat_model")
    if any(tuple(toks[i : i + n]) in brand_set for n in (1, 2, 3) for i in range(len(toks) - n + 1)):
        out.append("brand")
    if len(toks) >= long_words:
        out.append("long")
    return out


def build_slices(
    queries: Mapping[str, str], brand_set: set[tuple[str, ...]], long_words: int = 6
) -> dict[str, list[str]]:
    """query_id → query 文本 ⇒ 切片名 → 排序后的 query_id 列表（输出稳定，可进 git）。"""
    out: dict[str, list[str]] = {s: [] for s in SLICES}
    for qid, text in queries.items():
        for s in assign_slices(text, brand_set, long_words):
            out[s].append(qid)
    for s in out:
        out[s] = sorted(out[s], key=lambda x: (len(x), x))
    return out
