"""prepare 在合成 fixture 上的行为：清洗、切分、池规则、防泄漏断言、可复现、MANIFEST 护栏。"""

import json
import shutil

import pytest

from fixture_data import build_raw, fixture_config
from searchtune.data import prepare as prep
from searchtune.data.manifest import Layout, ManifestError, load_manifest, sha256_file, verify_artifacts
from searchtune.data.pool import Catalog, read_id_list, read_pools
from searchtune.data.pools import PoolBuilder, check_pool_rules, jaccard
from searchtune.data.prepare import QueryRec, Splits, check_splits, prepare
from searchtune.data.slices import assign_slices, build_brand_set
from searchtune.text import normalize_query, tokenize


def test_cleaning_counts(prepared):
    s = prepared["stats"]
    assert s["dropped_non_locale"] == 1  # examples 里的 1 行 es 样本
    assert s["dropped_no_product"] == 1
    assert s["dropped_duplicate_same_label"] == 1
    assert s["conflict_groups_dropped"] == 1
    assert s["dropped_no_esc_queries"] == 1  # 只有 I 的 query
    assert s["catalog_size"] == 400 and s["duplicate_product_rows"] == 0


def test_conflicting_pair_is_removed_from_pool(prepared):
    f, lay = prepared["facts"], prepared["layout"]
    pools = {p.query_id: p for p in read_pools(lay.pools("train")) + read_pools(lay.pools("dev"))}
    qid = str(f["conflict_query"])
    if qid in pools:
        assert f["conflict_pid"] not in {it.product_id for it in pools[qid].items if it.source == "labeled"}


def test_split_sizes_and_disjointness(prepared):
    lay, cfg = prepared["layout"], prepared["cfg"]
    ids = {n: read_id_list(lay.split_list(n)) for n in ("train", "dev", "dev-quick", "test", "rl_l1", "sft_traj", "rl_l2")}
    assert len(ids["test"]) == cfg.n_test and len(ids["dev"]) == cfg.n_dev and len(ids["dev-quick"]) == cfg.n_dev_quick
    assert (len(ids["rl_l1"]), len(ids["sft_traj"]), len(ids["rl_l2"])) == (cfg.n_rl_l1, cfg.n_sft_traj, cfg.n_rl_l2)
    assert not set(ids["train"]) & set(ids["dev"]) and not set(ids["train"]) & set(ids["test"])
    assert not set(ids["dev"]) & set(ids["test"])
    assert not set(ids["sft_traj"]) & set(ids["rl_l2"])
    assert set(ids["dev-quick"]) <= set(ids["dev"])
    assert prepared["facts"]["all_i_query"] not in sum(ids.values(), [])  # 无 E/S/C 的 query 不进任何 split


def test_test_split_only_small_version_queries(prepared):
    ids = set(read_id_list(prepared["layout"].split_list("test")))
    assert ids and all(int(q) > 60 for q in ids)  # fixture 里只有 61 号之后是 small_version=1


def test_normalized_query_overlap_is_removed(prepared):
    f, lay = prepared["facts"], prepared["layout"]
    test_ids = set(read_id_list(lay.split_list("test")))
    train_ids = set(read_id_list(lay.split_list("train")))
    assert f["overlap_test_q"] in test_ids
    assert f["overlap_train_q"] not in train_ids and f["overlap_train_q"] not in read_id_list(lay.split_list("dev"))
    assert prepared["stats"]["dropped_train_query_overlap"] == 1
    assert normalize_query("Red  Shoes Men") == normalize_query("red shoes men")


def test_pools_follow_the_rules(prepared):
    lay, cfg = prepared["layout"], prepared["cfg"]
    for split in ("dev", "test", "dev-quick", "train"):
        for p in read_pools(lay.pools(split)):
            assert p.split == split and p.data_version == cfg.data_version
            labeled = [it for it in p.items if it.source == "labeled"]
            assert labeled and {it.label for it in labeled} <= {"E", "S", "C", "I"}
            assert any(it.label in "ESC" for it in labeled)
            bm = [it for it in p.items if it.source == "bm25"]
            rnd = [it for it in p.items if it.source == "random"]
            assert len(bm) <= cfg.n_bm25_distractors and len(rnd) == cfg.n_random_distractors
            assert all(it.label == "U" and it.gain == 0.0 for it in bm + rnd)
            assert len({it.product_id for it in p.items}) == len(p.items)


def test_train_pools_cover_only_training_lists(prepared):
    lay = prepared["layout"]
    union = set(read_id_list(lay.split_list("rl_l1"))) | set(read_id_list(lay.split_list("sft_traj"))) | set(
        read_id_list(lay.split_list("rl_l2"))
    )
    assert {p.query_id for p in read_pools(lay.pools("train"))} == union


def test_jaccard_exclusion_rule_is_enforced():
    """构造一个必然触发规则的小例子：另一个 query 把商品 X 标为 E，两个 query 词集合 Jaccard ≥ 0.5。"""
    from searchtune.data.pools import build_distractor_index

    cfg = fixture_config(n_bm25_distractors=3, n_random_distractors=0)
    products = [("X", "red running shoes", None), ("Y", "red running shoes sale", None), ("Z", "red shoes", None),
                ("W", "blue running shoes", None)]
    index = build_distractor_index(products)
    qw = {"q": frozenset(tokenize("red running shoes")), "q2": frozenset(tokenize("red running shoes men"))}
    assert jaccard(qw["q"], qw["q2"]) == 0.75
    builder = PoolBuilder(index, {"X": ["q2"]}, qw, cfg)
    pool = builder.build("q", "red running shoes", "dev", {"Y": "E"})
    ids = {it.product_id for it in pool.items}
    assert "X" not in ids and "Y" in ids and {"Z", "W"} <= ids
    check_pool_rules(pool, {"X": ["q2"]}, qw, cfg)
    # 手动把 X 塞进去，独立检查必须报错
    from searchtune.data.pool import Pool, PoolItem

    bad = Pool("q", "red running shoes", "dev", pool.items + (PoolItem("X", "U", 0.0, "bm25"),), "v1")
    with pytest.raises(AssertionError, match="Jaccard"):
        check_pool_rules(bad, {"X": ["q2"]}, qw, cfg)


def test_distractor_overlapping_labeled_is_rejected():
    from searchtune.data.pool import Pool, PoolItem

    cfg = fixture_config()
    bad = Pool("q", "x", "dev", (PoolItem("A", "E", 1.0, "labeled"), PoolItem("B", "U", 0.0, "bm25"),
                                  PoolItem("C", "I", 0.0, "random")), "v1")
    with pytest.raises(AssertionError, match="必须是 U"):
        check_pool_rules(bad, {}, {"q": frozenset()}, cfg)


def _recs(spec):
    return {q: QueryRec(text, sp, True, {"p": "E"}) for q, (text, sp) in spec.items()}


def test_check_splits_catches_leaks():
    recs = _recs({"1": ("a b", "train"), "2": ("c d", "train"), "3": ("a  B", "test"), "4": ("e f", "test")})
    good = Splits(test=["4"], dev=["2"], dev_quick=["2"], train=["1"], rl_l1=["1"], sft_traj=["1"], rl_l2=[])
    check_splits(good, recs)
    with pytest.raises(AssertionError, match="query_id"):
        check_splits(Splits(["4"], ["2"], ["2"], ["1", "4"], [], [], []), recs)
    with pytest.raises(AssertionError, match="规范化"):
        check_splits(Splits(["3"], ["2"], ["2"], ["1"], [], [], []), recs)  # "a  B" 规范化后等于 train 的 "a b"
    with pytest.raises(AssertionError, match="sft_traj 与 rl_l2"):
        check_splits(Splits(["4"], ["2"], ["2"], ["1"], [], ["1"], ["1"]), recs)
    with pytest.raises(AssertionError, match="train 的子集"):
        check_splits(Splits(["4"], ["2"], ["2"], ["1"], ["2"], [], []), recs)


def test_catalog_cleaning(prepared):
    cat = Catalog.load(prepared["layout"].catalog)
    p = cat.get("P0001")
    assert p.description == "A aaa battery."  # HTML 标签去掉、空白合并
    assert p.bullets == "aaa design; long lasting battery"
    assert cat.get("P0000").description is None  # 缺失为 None，不是空串
    assert cat.get("P0005").bullets is None
    assert all(c.brand != "" and c.color != "" for c in (cat.get(f"P{i:04d}") for i in range(400)))


def test_catalog_partial_load_matches_full_load(prepared):
    lay = prepared["layout"]
    full = Catalog.load(lay.catalog)
    part = Catalog.load(lay.catalog, {"P0003", "P0100"})
    assert len(part) == 2 and part.get("P0100") == full.get("P0100")


def test_stats_contains_required_sections(prepared):
    text = prepared["layout"].stats.read_text(encoding="utf-8")
    for needle in ("各 split 的 query 数", "池大小分布", "池内标签比例", "缺失率", "清洗与过滤",
                   "test 池商品出现在 train 池中的比例", "难例切片"):
        assert needle in text


def test_prepare_is_reproducible(prepared, tmp_path):
    """data §9：从空目录一条命令复现，产物 sha256 与 MANIFEST 一致。"""
    root2 = tmp_path / "again"
    shutil.copytree(prepared["root"] / "raw", root2 / "raw")
    lay2 = Layout(root2, "v1")
    prepare(prepared["cfg"], lay2)
    a = load_manifest(prepared["layout"].manifest)["versions"]["v1"]["artifacts"]
    b = load_manifest(lay2.manifest)["versions"]["v1"]["artifacts"]
    assert a == b and len(a) >= 10
    assert prepared["layout"].stats.read_bytes() == lay2.stats.read_bytes()


def test_prepare_does_not_overwrite_existing_version(prepared):
    with pytest.raises(FileExistsError, match="升版本号"):
        prepare(prepared["cfg"], Layout(prepared["root"], "v1"))


def test_manifest_records_license_raw_hashes_and_params(prepared):
    m = load_manifest(prepared["layout"].manifest)
    assert m["license"]["name"] == "Apache-2.0"
    assert set(m["raw_files"]) == {"shopping_queries_dataset_examples.parquet", "shopping_queries_dataset_products.parquet"}
    v = m["versions"]["v1"]
    assert v["seed"] == 20261031 and v["params"]["n_dev"] == prepared["cfg"].n_dev and "code_commit" in v
    rel = prepared["layout"].rel(prepared["layout"].pools("dev"))
    assert v["artifacts"][rel]["rows"] == prepared["cfg"].n_dev


def test_manifest_detects_tampering(prepared, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(prepared["root"], root)
    lay = Layout(root, "v1")
    verify_artifacts(lay, [lay.pools("dev"), lay.catalog])
    with open(lay.pools("dev"), "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ManifestError, match="sha256"):
        verify_artifacts(lay, [lay.pools("dev")])
    with pytest.raises(ManifestError, match="MANIFEST 中没有"):
        verify_artifacts(lay, [root / "processed" / "v1" / "pools" / "nope.jsonl"])


def test_slices_rules():
    brands = build_brand_set(["Nike", "Nike", "Acme Corp", "Acme Corp", "Rare", None, "Generic", "Generic"], 2)
    assert brands == {("nike",), ("acme", "corp")}
    cases = {
        "sugar free gummy bears": {"negation"},
        "gummy bears without sugar": {"negation"},
        "aaa batteries 24 pack": {"quantity_spec"},
        "yoga mat 6mm": {"quantity_spec"},
        "case for iphone 13": {"compat_model"},
        "samsung a15 charger": {"compat_model"},
        "nike running shoes": {"brand"},
        "acme corp desk": {"brand"},
        "wireless noise cancelling over ear headphones for kids": {"long"},
        "red shoes": set(),
    }
    for q, expected in cases.items():
        assert expected <= set(assign_slices(q, brands)), q
    assert assign_slices("red shoes", brands) == []
    assert "negation" not in assign_slices("nominal shoes", brands)
    assert set(assign_slices("nike sugar free 24 pack case for iphone 13 black", brands)) >= {
        "negation", "quantity_spec", "compat_model", "brand", "long"
    }


def test_slice_files_written_and_sorted(prepared):
    lay = prepared["layout"]
    for split in ("dev", "test"):
        sl = json.loads(lay.slices(split).read_text(encoding="utf-8"))
        assert set(sl) == {"negation", "quantity_spec", "compat_model", "brand", "long"}
        split_ids = set(read_id_list(lay.split_list(split)))
        for ids in sl.values():
            assert set(ids) <= split_ids and ids == sorted(ids, key=lambda x: (len(x), x))


def test_sha256_helper(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_prep_module_helpers():
    assert prep.clean_html("<p>A  <b>b</b>&amp;c</p>") == "A b &c"
    assert prep.clean_bullets("one  \n\n two <i>x</i>") == "one; two x"
    assert prep.clean_html("   ") is None and prep.clean_html(None) is None
    a = prep._pick(["3", "1", "22", "2"], 2, 7, "p")
    assert a == prep._pick(["22", "2", "1", "3"], 2, 7, "p")  # 与输入顺序无关
    with pytest.raises(ValueError, match="不足"):
        prep._pick(["1"], 2, 7, "p")
