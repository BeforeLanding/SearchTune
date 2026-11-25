"""商品池与商品目录的数据结构及读写（specs/data.md §3、§5.3）。

``Pool`` 持有标签和增益；环境暴露给 Agent 的接口只返回商品文本，见 ``env/``。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

GAIN: dict[str, float] = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0, "U": 0.0}
LABELS = tuple(GAIN)
SOURCES = ("labeled", "bm25", "random")


@dataclass(frozen=True)
class PoolItem:
    product_id: str
    label: str
    gain: float
    source: str


@dataclass
class Pool:
    query_id: str
    query: str
    split: str
    items: tuple[PoolItem, ...]
    data_version: str

    def __post_init__(self) -> None:
        self._by_id = {it.product_id: it for it in self.items}
        if len(self._by_id) != len(self.items):
            raise ValueError(f"池 {self.query_id} 中有重复商品")

    @property
    def product_ids(self) -> list[str]:
        return [it.product_id for it in self.items]

    def __contains__(self, product_id: str) -> bool:
        return product_id in self._by_id

    def item(self, product_id: str) -> PoolItem:
        return self._by_id[product_id]

    def gain(self, product_id: str) -> float:
        it = self._by_id.get(product_id)
        return it.gain if it else 0.0

    def label(self, product_id: str) -> str | None:
        it = self._by_id.get(product_id)
        return it.label if it else None

    @property
    def labeled_ids(self) -> list[str]:
        return [it.product_id for it in self.items if it.source == "labeled"]

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "split": self.split,
            "pool": [
                {"product_id": it.product_id, "label": it.label, "gain": it.gain, "source": it.source}
                for it in self.items
            ],
            "data_version": self.data_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Pool:
        items = []
        for p in d["pool"]:
            label = p["label"]
            if label not in GAIN:
                raise ValueError(f"未知标签 {label!r}")
            if p["gain"] != GAIN[label]:
                raise ValueError(f"标签 {label} 的增益应为 {GAIN[label]}，实际 {p['gain']}")
            if p["source"] not in SOURCES:
                raise ValueError(f"未知来源 {p['source']!r}")
            items.append(PoolItem(p["product_id"], label, float(p["gain"]), p["source"]))
        return cls(str(d["query_id"]), d["query"], d["split"], tuple(items), d["data_version"])


def iter_pools(path: str | Path) -> Iterator[Pool]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Pool.from_dict(json.loads(line))


def read_pools(path: str | Path) -> list[Pool]:
    return list(iter_pools(path))


@dataclass(frozen=True)
class Product:
    product_id: str
    title: str
    brand: str | None
    color: str | None
    bullets: str | None
    description: str | None

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "color": self.color,
            "bullets": self.bullets,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Product:
        return cls(
            d["product_id"], d["title"], d.get("brand"), d.get("color"), d.get("bullets"), d.get("description")
        )


class Catalog:
    """商品目录：product_id → Product。"""

    def __init__(self, products: Iterable[Product]):
        self._p = {p.product_id: p for p in products}

    def __len__(self) -> int:
        return len(self._p)

    def __contains__(self, product_id: str) -> bool:
        return product_id in self._p

    def get(self, product_id: str) -> Product:
        return self._p[product_id]

    @classmethod
    def load(cls, path: str | Path, ids: set[str] | None = None) -> Catalog:
        """读取 catalog.jsonl；给定 ``ids`` 时只解析其中的商品（按行首的 product_id 预过滤）。"""
        prefix = '{"product_id": "'
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if ids is not None:
                    if not line.startswith(prefix):
                        raise ValueError("catalog.jsonl 行格式不符：product_id 必须是第一个字段")
                    pid = line[len(prefix) : line.index('"', len(prefix))]
                    if pid not in ids:
                        continue
                out.append(Product.from_dict(json.loads(line)))
        return cls(out)


def read_id_list(path: str | Path) -> list[str]:
    """读取 ``data/splits/*.txt``（每行一个 query_id）。"""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]
