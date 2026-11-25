"""raw → catalog、pools、splits、slices、MANIFEST、STATS（DESIGN.md §2；specs/data.md §4–§8）。

清洗顺序严格对应 specs/data.md §6；其中 §5.4 的“无 E/S/C 的 query 不建池”在抽样之前执行，
否则 dev/test 的规模会被事后过滤缩水。
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from searchtune.config import DataConfig
from searchtune.data.manifest import Layout, count_rows, sha256_file
from searchtune.data.pool import GAIN, Pool
from searchtune.data.pools import PoolBuilder, build_distractor_index, check_pool_rules
from searchtune.data.slices import SLICE_RULES, SLICES, build_brand_set, build_slices
from searchtune.runs import full_commit, is_dirty
from searchtune.text import normalize_query, tokenize

EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"
LICENSE_INFO = {
    "name": "Apache-2.0",
    "source": "https://github.com/amazon-science/esci-data/blob/main/LICENSE",
    "note": "数据集与代码均随 esci-data 仓库以 Apache License 2.0 发布",
}
_TAG_RE = re.compile(r"<[^>]+>")


def qid_key(qid: str) -> tuple[int, str]:
    """query_id 的稳定排序键（数字串按数值序）。"""
    return (len(qid), qid)


def _pick(ids: Iterable[str], n: int, seed: int, purpose: str) -> list[str]:
    """按 sha256(seed:purpose:id) 排序后取前 n 个：结果与输入顺序、平台、Python 版本无关。"""
    pool = sorted(ids, key=qid_key)
    if len(pool) < n:
        raise ValueError(f"{purpose}：候选 query 只有 {len(pool)} 个，不足 {n}")
    keyed = sorted(pool, key=lambda q: hashlib.sha256(f"{seed}:{purpose}:{q}".encode()).hexdigest())
    return sorted(keyed[:n], key=qid_key)


# ---- 文本清洗 -----------------------------------------------------------------
def _collapse(s: str | None) -> str | None:
    if s is None:
        return None
    s = " ".join(s.split())
    return s or None


def clean_html(s: str | None) -> str | None:
    if s is None:
        return None
    return _collapse(html.unescape(_TAG_RE.sub(" ", s)))


def clean_bullets(s: str | None) -> str | None:
    if s is None:
        return None
    parts = [clean_html(line) for line in s.splitlines()]
    out = "; ".join(p for p in parts if p)
    return out or None


# ---- 清洗与切分 -----------------------------------------------------------------
@dataclass
class QueryRec:
    query: str
    esci_split: str
    small: bool = True  # 该 query 的全部样本行 small_version 都是 1
    labels: dict[str, str] = field(default_factory=dict)  # product_id → E/S/C/I


def load_and_clean(examples_path: Path, product_ids: set[str], cfg: DataConfig, stats: dict) -> dict[str, QueryRec]:
    """specs/data.md §6 的第 1–3 步。"""
    t = pq.read_table(
        examples_path,
        columns=["query", "query_id", "product_id", "product_locale", "esci_label", "small_version", "split"],
    )
    stats["raw_example_rows"] = t.num_rows
    t = t.filter(pc.equal(t["product_locale"], cfg.locale))
    stats["dropped_non_locale"] = stats["raw_example_rows"] - t.num_rows
    cols = {c: t[c].to_pylist() for c in ("query", "query_id", "product_id", "esci_label", "small_version", "split")}
    n = t.num_rows
    stats["locale_example_rows"] = n

    recs: dict[str, QueryRec] = {}
    seen: dict[tuple[str, str], str] = {}
    conflicts: set[tuple[str, str]] = set()
    n_no_product = n_dup_same = 0
    text_mismatch = 0
    cross_split: set[str] = set()  # 同一 query_id 同时出现在 ESCI 的 train 与 test（真实数据里有）
    for i in range(n):
        pid = cols["product_id"][i]
        if pid not in product_ids:
            n_no_product += 1
            continue
        qid = str(cols["query_id"][i])
        lab = cols["esci_label"][i]
        if lab not in ("E", "S", "C", "I"):
            raise ValueError(f"未知的 esci_label {lab!r}")
        rec = recs.get(qid)
        if rec is None:
            rec = recs[qid] = QueryRec(cols["query"][i], cols["split"][i])
        else:
            text_mismatch += rec.query != cols["query"][i]
            if rec.esci_split != cols["split"][i]:
                cross_split.add(qid)
        rec.small = rec.small and cols["small_version"][i] == 1
        key = (qid, pid)
        if key in seen:
            if seen[key] == lab:
                n_dup_same += 1
            else:
                conflicts.add(key)
        else:
            seen[key] = lab
    stats["dropped_no_product"] = n_no_product
    stats["dropped_duplicate_same_label"] = n_dup_same
    stats["conflict_groups_dropped"] = len(conflicts)
    stats["query_text_mismatch"] = text_mismatch
    stats["dropped_cross_split_queries"] = len(cross_split)
    if text_mismatch:
        raise ValueError("同一 query_id 对应多个 query 文本，数据与假设不符，请先核对 specs/data.md")
    for (qid, pid), lab in seen.items():
        if (qid, pid) not in conflicts:
            recs[qid].labels[pid] = lab
    # 标签冲突整组删除后，可能留下没有任何标注的 query
    return {q: r for q, r in recs.items() if r.labels and q not in cross_split}


@dataclass
class Splits:
    test: list[str]
    dev: list[str]
    dev_quick: list[str]
    train: list[str]
    rl_l1: list[str]
    sft_traj: list[str]
    rl_l2: list[str]

    def as_files(self) -> dict[str, list[str]]:
        return {
            "test": self.test,
            "dev": self.dev,
            "dev-quick": self.dev_quick,
            "train": self.train,
            "rl_l1": self.rl_l1,
            "sft_traj": self.sft_traj,
            "rl_l2": self.rl_l2,
        }


def make_splits(recs: dict[str, QueryRec], cfg: DataConfig, stats: dict) -> Splits:
    """specs/data.md §4、§6.4、training.md §6。"""
    eligible = {q for q, r in recs.items() if set(r.labels.values()) & {"E", "S", "C"}}
    stats["dropped_no_esc_queries"] = len(recs) - len(eligible)
    norm = {q: normalize_query(recs[q].query) for q in eligible}

    test_pop = [q for q in eligible if recs[q].esci_split == "test" and recs[q].small]
    test = _pick(test_pop, cfg.n_test, cfg.seed, "test")
    test_norm = {norm[q] for q in test}

    dev_pop = [q for q in eligible if recs[q].esci_split == "train" and norm[q] not in test_norm]
    dev = _pick(dev_pop, cfg.n_dev, cfg.seed, "dev")
    dev_norm = {norm[q] for q in dev}
    stats["dev_excluded_by_test_overlap"] = sum(
        1 for q in eligible if recs[q].esci_split == "train" and norm[q] in test_norm
    )
    dev_quick = _pick(dev, cfg.n_dev_quick, cfg.seed, "dev-quick")

    heldout = test_norm | dev_norm
    dev_set = set(dev)
    train_all = [q for q in eligible if recs[q].esci_split == "train" and q not in dev_set]
    train = [q for q in train_all if norm[q] not in heldout]
    stats["dropped_train_query_overlap"] = len(train_all) - len(train)
    train = sorted(train, key=qid_key)

    rl_l1 = _pick(train, cfg.n_rl_l1, cfg.seed, "rl_l1")
    sft = _pick(train, cfg.n_sft_traj, cfg.seed, "sft_traj")
    sft_set = set(sft)
    rl_l2 = _pick([q for q in train if q not in sft_set], cfg.n_rl_l2, cfg.seed, "rl_l2")
    return Splits(test, dev, dev_quick, train, rl_l1, sft, rl_l2)


def check_splits(sp: Splits, recs: dict[str, QueryRec]) -> None:
    """data §6.5、§9 与 training §10 的自动检查；任何一项不满足都抛 AssertionError。"""
    train, dev, test = set(sp.train), set(sp.dev), set(sp.test)
    assert not (train & dev), "train 与 dev 的 query_id 相交"
    assert not (train & test), "train 与 test 的 query_id 相交"
    assert not (dev & test), "dev 与 test 的 query_id 相交"
    assert set(sp.dev_quick) <= dev, "dev-quick 不是 dev 的子集"
    norm = lambda s: {normalize_query(recs[q].query) for q in s}  # noqa: E731
    assert not (norm(train) & norm(dev)), "规范化后 train 与 dev 的 query 重叠"
    assert not (norm(train) & norm(test)), "规范化后 train 与 test 的 query 重叠"
    assert not (norm(dev) & norm(test)), "规范化后 dev 与 test 的 query 重叠"
    sft, rl2, rl1 = set(sp.sft_traj), set(sp.rl_l2), set(sp.rl_l1)
    assert not (sft & rl2), "sft_traj 与 rl_l2 相交"
    for name, s in (("rl_l1", rl1), ("sft_traj", sft), ("rl_l2", rl2)):
        assert s <= train, f"{name} 不是 train 的子集"
        assert not (s & dev) and not (s & test), f"{name} 与 dev/test 相交"


# ---- 主流程 -----------------------------------------------------------------
def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for ln in lines:
            f.write(ln + "\n")


def prepare(cfg: DataConfig, layout: Layout, raw_dir: Path | None = None, repo_root: Path | None = None) -> dict:
    """完整的数据准备流程；返回 stats。已存在的数据版本不覆盖（升版本号）。"""
    layout.version = cfg.data_version
    raw = raw_dir or layout.raw
    if layout.processed.exists() and any(layout.processed.iterdir()):
        raise FileExistsError(f"{layout.processed} 已存在；改动数据参数请升版本号，旧版本不覆盖（specs/data.md §7）")
    stats: dict = {"data_version": cfg.data_version}

    # 商品表
    prod = pq.read_table(raw / PRODUCTS_FILE)
    stats["raw_product_rows"] = prod.num_rows
    prod = prod.filter(pc.equal(prod["product_locale"], cfg.locale))
    prod = prod.take(pc.sort_indices(prod["product_id"]))
    pids = prod["product_id"].to_pylist()
    uniq = sorted(set(pids))
    stats["duplicate_product_rows"] = len(pids) - len(uniq)
    product_ids = set(uniq)
    stats["catalog_size"] = len(product_ids)

    # 清洗、切分
    recs = load_and_clean(raw / EXAMPLES_FILE, product_ids, cfg, stats)
    splits = make_splits(recs, cfg, stats)
    check_splits(splits, recs)

    # catalog.jsonl（按 product_id 排序，重复商品保留第一条）
    layout.processed.mkdir(parents=True, exist_ok=True)
    brands: list[str | None] = []
    distractor_src: list[tuple[str, str, str | None]] = []
    miss = Counter()
    n_cat = 0
    seen_pid: set[str] = set()

    def catalog_rows() -> Iterable[dict]:
        nonlocal n_cat
        for off in range(0, prod.num_rows, 50_000):
            chunk = prod.slice(off, 50_000).to_pylist()
            for r in chunk:
                pid = r["product_id"]
                if pid in seen_pid:
                    continue
                seen_pid.add(pid)
                title = _collapse(r["product_title"]) or ""
                row = {
                    "product_id": pid,
                    "title": title,
                    "brand": _collapse(r["product_brand"]),
                    "color": _collapse(r["product_color"]),
                    "bullets": clean_bullets(r["product_bullet_point"]),
                    "description": clean_html(r["product_description"]),
                }
                for k in ("brand", "color", "bullets", "description"):
                    miss[k] += row[k] is None
                brands.append(row["brand"])
                distractor_src.append((pid, title, row["brand"]))
                n_cat += 1
                yield row

    _write_jsonl(layout.catalog, catalog_rows())
    stats["field_missing_rate"] = {k: miss[k] / n_cat for k in ("brand", "color", "bullets", "description")}

    # 池构造所需的全局结构
    index = build_distractor_index(distractor_src)
    es_queries: dict[str, list[str]] = defaultdict(list)
    for qid, r in recs.items():
        for pid, lab in r.labels.items():
            if lab in ("E", "S"):
                es_queries[pid].append(qid)
    query_words = {q: frozenset(tokenize(r.query)) for q, r in recs.items()}
    builder = PoolBuilder(index, es_queries, query_words, cfg)

    train_union = sorted(set(splits.rl_l1) | set(splits.sft_traj) | set(splits.rl_l2), key=qid_key)
    to_build = {"test": splits.test, "dev": splits.dev, "train": train_union}
    built: dict[str, dict[str, Pool]] = {}
    for split, qids in to_build.items():
        built[split] = {}
        for q in qids:
            pool = builder.build(q, recs[q].query, split, recs[q].labels)
            check_pool_rules(pool, es_queries, query_words, cfg)
            built[split][q] = pool
    dq = {q: built["dev"][q] for q in splits.dev_quick}
    built["dev-quick"] = {
        q: Pool(p.query_id, p.query, "dev-quick", p.items, p.data_version) for q, p in dq.items()
    }
    for split in ("test", "dev", "dev-quick", "train"):
        _write_jsonl(layout.pools(split), (p.to_dict() for p in built[split].values()))

    # splits、slices
    for name, ids in splits.as_files().items():
        _write_lines(layout.split_list(name), ids)
    brand_set = build_brand_set(brands, cfg.brand_min_count)
    stats["brand_set_size"] = len(brand_set)
    slice_counts = {}
    for split, qids in (("dev", splits.dev), ("test", splits.test)):
        sl = build_slices({q: recs[q].query for q in qids}, brand_set, cfg.long_query_words)
        layout.slices(split).parent.mkdir(parents=True, exist_ok=True)
        layout.slices(split).write_text(
            json.dumps(sl, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        slice_counts[split] = {s: len(sl[s]) for s in SLICES}
    stats["slice_counts"] = slice_counts

    stats.update(_pool_stats(built, splits))
    layout.stats.write_text(render_stats(stats, cfg), encoding="utf-8", newline="\n")
    write_manifest(layout, cfg, raw, repo_root)
    return stats


def _pool_stats(built: dict[str, dict[str, Pool]], splits: Splits) -> dict:
    out: dict = {"n_queries": {k: len(v) for k, v in built.items()}}
    out["n_queries"]["train_total"] = len(splits.train)
    sizes, labels = {}, {}
    for split, pools in built.items():
        sz = sorted(len(p.items) for p in pools.values())
        sizes[split] = {
            "min": sz[0],
            "median": statistics.median(sz),
            "mean": round(statistics.mean(sz), 1),
            "p90": sz[int(0.9 * (len(sz) - 1))],
            "max": sz[-1],
        }
        c = Counter(it.label for p in pools.values() for it in p.items)
        tot = sum(c.values())
        labels[split] = {k: c.get(k, 0) / tot for k in GAIN}
    out["pool_size"] = sizes
    out["label_share"] = labels
    train_items = {it.product_id for p in built["train"].values() for it in p.items}
    test_items = [it.product_id for p in built["test"].values() for it in p.items]
    test_set = set(test_items)
    out["test_products_in_train_pools"] = len(test_set & train_items) / len(test_set)
    lab = {it.product_id for p in built["test"].values() for it in p.items if it.source == "labeled"}
    out["test_labeled_products_in_train_pools"] = len(lab & train_items) / len(lab) if lab else 0.0
    out["short_bm25_pools"] = {
        s: sum(1 for p in pools.values() if sum(1 for it in p.items if it.source == "bm25") < 50)
        for s, pools in built.items()
    }
    return out


def render_stats(s: dict, cfg: DataConfig) -> str:
    """data/STATS.md：不含时间戳，保证可复现。"""
    L = [f"# 数据统计（{s['data_version']}）", "", "> 由 `python -m searchtune prepare` 生成，请勿手改。", ""]
    L += ["## 清洗与过滤（按执行顺序）", "", "| 步骤 | 数量 |", "|---|---|"]
    rows = [
        ("原始样本行数（全部 locale）", s["raw_example_rows"]),
        (f"过滤 locale != {cfg.locale}（删除）", s["dropped_non_locale"]),
        (f"{cfg.locale} 样本行数", s["locale_example_rows"]),
        ("关联不上商品表（删除样本行）", s["dropped_no_product"]),
        ("(query_id, product_id) 重复且标签一致（删除多余行）", s["dropped_duplicate_same_label"]),
        ("(query_id, product_id) 标签冲突（整组删除，组数）", s["conflict_groups_dropped"]),
        ("query_id 同时出现在 ESCI 的 train 与 test（整个 query 删除）", s["dropped_cross_split_queries"]),
        ("无 E/S/C 的 query（不建池，抽样前过滤）", s["dropped_no_esc_queries"]),
        ("dev 候选中因与 test 规范化 query 相同而排除的 query", s["dev_excluded_by_test_overlap"]),
        ("train 中因与 dev/test 规范化 query 相同而删除的 query", s["dropped_train_query_overlap"]),
    ]
    L += [f"| {a} | {b:,} |" for a, b in rows]
    L += ["", f"商品表 {s['raw_product_rows']:,} 行；{cfg.locale} 商品目录 {s['catalog_size']:,} 个"
          f"（重复行 {s['duplicate_product_rows']:,}）。", ""]
    L += ["## 各 split 的 query 数", "", "| split | query 数 | 建池数 |", "|---|---|---|"]
    nq = s["n_queries"]
    for k in ("train", "dev", "dev-quick", "test"):
        total = nq["train_total"] if k == "train" else nq[k]
        L.append(f"| {k} | {total:,} | {nq[k]:,} |")
    L += ["", "train 只为 `rl_l1 ∪ sft_traj ∪ rl_l2` 中的 query 建池（DEC-015）。", ""]
    L += ["## 池大小分布", "", "| split | min | median | mean | p90 | max |", "|---|---|---|---|---|---|"]
    for k, v in s["pool_size"].items():
        L.append(f"| {k} | {v['min']} | {v['median']} | {v['mean']} | {v['p90']} | {v['max']} |")
    L += ["", "## 池内标签比例", "", "| split | E | S | C | I | U |", "|---|---|---|---|---|---|"]
    for k, v in s["label_share"].items():
        L.append(f"| {k} | " + " | ".join(f"{v[x]:.4f}" for x in ("E", "S", "C", "I", "U")) + " |")
    L += ["", "## 商品字段缺失率（目录）", "", "| 字段 | 缺失率 |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in s["field_missing_rate"].items()]
    L += ["", "## 泄漏相关", "",
          f"- test 池商品出现在 train 池中的比例：{s['test_products_in_train_pools']:.4f}"
          f"（仅标注商品：{s['test_labeled_products_in_train_pools']:.4f}）",
          f"- BM25 干扰不足 {cfg.n_bm25_distractors} 个的池数：" +
          "，".join(f"{k} {v}" for k, v in s["short_bm25_pools"].items()), ""]
    L += ["## 难例切片", "", "一个 query 可属于多个切片；成员列表见 `data/slices/<split>.json`。", "",
          "| 切片 | 规则 | dev | test |", "|---|---|---|---|"]
    for sl in SLICES:
        L.append(f"| {sl} | {SLICE_RULES[sl]} | {s['slice_counts']['dev'][sl]} | {s['slice_counts']['test'][sl]} |")
    L += ["", f"品牌集合大小（目录中至少 {cfg.brand_min_count} 个商品的品牌）：{s['brand_set_size']:,}。", ""]
    return "\n".join(L)


def write_manifest(layout: Layout, cfg: DataConfig, raw: Path, repo_root: Path | None) -> None:
    path = layout.manifest
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"versions": {}}
    manifest["license"] = LICENSE_INFO
    manifest["raw_files"] = {
        f: {"sha256": sha256_file(raw / f), "bytes": (raw / f).stat().st_size} for f in (EXAMPLES_FILE, PRODUCTS_FILE)
    }
    artifacts = {}
    files = [layout.catalog, *sorted(layout.processed.glob("pools/*.jsonl")), *sorted(layout.root.glob("splits/*.txt")),
             *sorted(layout.root.glob("slices/*.json"))]
    for p in files:
        artifacts[layout.rel(p)] = {
            "sha256": sha256_file(p),
            "rows": count_rows(p) if p.suffix in (".jsonl", ".txt") else None,
        }
    manifest["versions"][cfg.data_version] = {
        "code_commit": full_commit(repo_root),
        "code_dirty": is_dirty(repo_root),
        "seed": cfg.seed,
        "params": asdict(cfg),
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
