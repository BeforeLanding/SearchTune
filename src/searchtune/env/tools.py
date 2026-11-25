"""三个工具的返回文本格式（specs/training.md §2.1）。

这里只拼文本，**永远不接触 label / gain / source**：函数签名里根本没有这些参数。
"""

from __future__ import annotations

from collections.abc import Sequence

from searchtune.config import EnvConfig
from searchtune.data.pool import Product
from searchtune.text import truncate_words

NA = "N/A"

NO_RESULTS = "No results found."
BUDGET_NOTICE = "[Tool budget used up. You must call submit now.]"
OVER_BUDGET = "Tool budget exhausted. No more search or view calls are allowed."
FINISHED = "The episode has already finished."
EMPTY_QUERY = "Error: query must be a non-empty string."
UNKNOWN_PRODUCT = "Error: product_id not found in this product pool."


def _field(value: str | None) -> str:
    return value if value else NA


def format_search(products: Sequence[Product], cfg: EnvConfig) -> str:
    """``[product_id] 标题（截断）| 品牌``，每个结果一行。"""
    if not products:
        return NO_RESULTS
    return "\n".join(
        f"[{p.product_id}] {truncate_words(p.title, cfg.search_title_tokens)} | {_field(p.brand)}"
        for p in products
    )


def format_view(p: Product, cfg: EnvConfig) -> str:
    """标题、品牌、颜色、卖点（截断）；不含商品描述。"""
    bullets = truncate_words(p.bullets, cfg.view_bullet_tokens) if p.bullets else NA
    return "\n".join(
        [
            f"[{p.product_id}]",
            f"Title: {p.title}",
            f"Brand: {_field(p.brand)}",
            f"Color: {_field(p.color)}",
            f"Bullet points: {bullets}",
        ]
    )
