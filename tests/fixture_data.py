"""合成的小型 ESCI parquet fixture：字段与真实文件一致（已用真实文件核对），规模很小。

只用于测试管线逻辑（切分、池规则、护栏、端到端接线）；不代表真实数据上的任何结论。
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from searchtune.config import DataConfig

CATEGORIES = {
    "shoes": ["running", "leather", "canvas", "hiking", "casual"],
    "battery": ["aaa", "aa", "lithium", "alkaline", "rechargeable"],
    "case": ["iphone", "silicone", "clear", "leather", "slim"],
    "lamp": ["desk", "led", "floor", "reading", "touch"],
    "mug": ["ceramic", "travel", "steel", "funny", "large"],
    "gummy": ["sugar", "free", "fruit", "bears", "vegan"],
    "cable": ["usb", "braided", "long", "fast", "type"],
    "bottle": ["water", "glass", "insulated", "kids", "sport"],
}
BRANDS = ["Nike", "Acme", "Zeta", "Orbit", None]
EXAMPLES_SCHEMA = pa.schema(
    [
        ("example_id", pa.int64()),
        ("query", pa.string()),
        ("query_id", pa.int64()),
        ("product_id", pa.string()),
        ("product_locale", pa.string()),
        ("esci_label", pa.string()),
        ("small_version", pa.int64()),
        ("large_version", pa.int64()),
        ("split", pa.string()),
    ]
)
N_PRODUCTS = 400
N_TRAIN_Q, N_TEST_Q = 50, 30
NOT_SMALL = set(range(51, 61)) | {9, 10}  # 非 small_version 的 query：train 里的 9、10 与 test 里的 51..60
ALL_I_QUERY = 80  # 一个只有 I 标注的 query：不建池
# test query 为 51..80，其中 61..80 是 small_version=1；去掉 ALL_I_QUERY 和“小版本不一致”的 70 号后，可进 test 的有 18 个
CROSS_SPLIT_QUERY = 7  # 同一 query_id 同时出现在 train 与 test（真实 ESCI 里有 1 个这样的 query）
MIXED_SMALL_QUERY = 70  # 各行 small_version 不一致（真实 ESCI 里有 7 个）


def fixture_config(**kw) -> DataConfig:
    base = dict(
        n_test=18, n_dev=8, n_dev_quick=3, n_rl_l1=10, n_sft_traj=8, n_rl_l2=8,
        n_bm25_distractors=5, n_random_distractors=5, bm25_rank_lo=3, bm25_rank_hi=60, brand_min_count=2,
    )
    base.update(kw)
    return DataConfig(**base)


def build_raw(raw_dir: Path) -> dict:
    """写出 examples / products 两个 parquet，返回一些供断言用的事实。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cats = list(CATEGORIES)
    prods = []
    for i in range(N_PRODUCTS):
        cat = cats[i % len(cats)]
        adj = CATEGORIES[cat][(i // len(cats)) % 5]
        brand = BRANDS[i % len(BRANDS)]
        prods.append(
            {
                "product_id": f"P{i:04d}",
                "product_title": f"{adj} {cat}  for daily use model {i}",
                "product_description": None if i % 7 == 0 else f"<p>A <b>{adj}</b> {cat}.</p>\n<br/>",
                "product_bullet_point": None if i % 5 == 0 else f"{adj} design\nlong lasting {cat}",
                "product_brand": brand,
                "product_color": None if i % 3 == 0 else "black",
                "product_locale": "us",
            }
        )
    prods.append({**prods[0], "product_locale": "es", "product_title": "otro"})  # 非 us，必须被过滤
    pt = pa.Table.from_pylist(prods)
    pq.write_table(pt, raw_dir / "shopping_queries_dataset_products.parquet")

    rows = []
    eid = 0

    def add(qid, qtext, pid, label, split, small, locale="us"):
        nonlocal eid
        rows.append(
            {"example_id": eid, "query": qtext, "query_id": qid, "product_id": pid, "product_locale": locale,
             "esci_label": label, "small_version": small, "large_version": 1, "split": split}
        )
        eid += 1

    labels_cycle = ["E", "E", "S", "C", "I", "I"]
    for qid in range(1, N_TRAIN_Q + N_TEST_Q + 1):
        cat = cats[qid % len(cats)]
        adj = CATEGORIES[cat][qid % 5]
        extra = ["", " 24 pack", " without sugar", " for iphone 13", " nike", " black large size"][qid % 6]
        text = f"{adj} {cat}{extra} q{qid}"  # q{qid} 保证规范化 query 互不相同
        split = "train" if qid <= N_TRAIN_Q else "test"
        small = 0 if qid in NOT_SMALL else 1
        base = qid * 3
        for j, lab in enumerate(labels_cycle):
            if qid == ALL_I_QUERY:
                lab = "I"
            add(qid, text, f"P{(base + j * 7) % N_PRODUCTS:04d}", lab, split, small)
    # 规范化后重名的 query：train 的 5 号与 test 的 65 号（65 是 small_version=1，会被选为 test）
    for r in rows:
        if r["query_id"] == 5:
            r["query"] = "Red  Shoes Men"
        if r["query_id"] == 65:
            r["query"] = "red shoes men"
    for r in rows:
        if r["query_id"] == MIXED_SMALL_QUERY:
            r["small_version"] = 0
            break
    qtext7 = next(r for r in rows if r["query_id"] == CROSS_SPLIT_QUERY)["query"]
    add(CROSS_SPLIT_QUERY, qtext7, "P0399", "E", "test", 1)
    # 清洗用例：完全重复、标签冲突、关联不上商品、非 us 样本
    r0 = next(r for r in rows if r["query_id"] == 2)
    add(r0["query_id"], r0["query"], r0["product_id"], r0["esci_label"], "train", 0)  # 重复，标签一致
    r1 = next(r for r in rows if r["query_id"] == 3)
    add(r1["query_id"], r1["query"], r1["product_id"], "I" if r1["esci_label"] != "I" else "E", "train", 0)  # 冲突
    add(4, next(r for r in rows if r["query_id"] == 4)["query"], "P_MISSING", "E", "train", 0)  # 无商品
    add(4, "x", "P0001", "E", "train", 0, locale="es")  # 非 us
    pq.write_table(pa.Table.from_pylist(rows, schema=EXAMPLES_SCHEMA), raw_dir / "shopping_queries_dataset_examples.parquet")
    return {
        "dup_query": 2, "conflict_query": 3, "conflict_pid": r1["product_id"],
        "overlap_train_q": "5", "overlap_test_q": "65", "all_i_query": str(ALL_I_QUERY),
        "cross_split_query": str(CROSS_SPLIT_QUERY), "mixed_small_query": str(MIXED_SMALL_QUERY),
    }
