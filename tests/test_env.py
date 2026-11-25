import pytest

from searchtune.config import EnvConfig
from searchtune.data.pool import GAIN, Catalog, Pool, PoolItem, Product
from searchtune.env import tools
from searchtune.env.episode import Episode
from searchtune.env.pool_index import BM25Index, PoolIndex
from searchtune.text import tokenize, truncate_words


def make_world(labels=None):
    """8 个商品的小池；标题里故意含 'label' 'gain' 'source' 以外的普通词。"""
    labels = labels or ["E", "S", "C", "I", "U", "U", "E", "I"]
    prods = []
    for i, lab in enumerate(labels):
        prods.append(
            Product(
                f"p{i}",
                f"red running shoes model {i} " + "extra " * 40,
                None if i == 3 else f"Brand{i}",
                "black" if i % 2 else None,
                "light " * 300 if i == 0 else f"bullet {i}",
                "DESCRIPTION-SECRET " + str(i),
            )
        )
    items = tuple(PoolItem(p.product_id, lab, GAIN[lab], "labeled" if lab != "U" else "random") for p, lab in zip(prods, labels))
    return Pool("q1", "red shoes", "dev", items, "v1"), Catalog(prods)


def test_search_format_and_truncation():
    pool, cat = make_world()
    ep = Episode(pool, cat)
    out = ep.search("red shoes")
    lines = out.splitlines()
    assert len(lines) == 8  # 池里只有 8 个商品，全部匹配
    assert lines[0].startswith("[p") and " | " in lines[0]
    title = lines[0].split("] ", 1)[1].rsplit(" | ", 1)[0]
    assert title.endswith(" ...") and len(title.split()) == 32 + 1  # 32 个词 + 省略号
    assert any(line.endswith("| N/A") for line in lines)  # 缺失品牌显示 N/A


def test_search_returns_at_most_10():
    prods = [Product(f"p{i:02d}", f"widget {i}", None, None, None, None) for i in range(15)]
    pool = Pool("q", "widget", "dev", tuple(PoolItem(p.product_id, "U", 0.0, "random") for p in prods), "v1")
    assert len(Episode(pool, Catalog(prods)).search("widget").splitlines()) == 10


def test_search_no_results_and_empty_query():
    pool, cat = make_world()
    ep = Episode(pool, cat)
    assert ep.search("zzzzqqq") == tools.NO_RESULTS
    assert ep.search("   ").startswith("Error")
    assert ep.result.n_search == 2  # 无结果和空 query 都占用预算


def test_view_format_bullets_truncated_and_no_description():
    pool, cat = make_world()
    ep = Episode(pool, cat)
    out = ep.view("p0")
    assert out.splitlines()[0] == "[p0]" and "Title: " in out and "Brand: Brand0" in out
    assert "Color: N/A" in out
    bullets = out.split("Bullet points: ", 1)[1]
    assert len(bullets.split()) == 200 + 1 and bullets.endswith("...")
    assert "DESCRIPTION-SECRET" not in out  # 描述不暴露
    assert ep.view("nope").startswith("Error")


def test_tools_never_leak_label_gain_source():
    """eval §9：返回给 Agent 的内容里不含 label / gain / source。

    做法：把同一批商品配上两套完全不同的标签，所有工具的输出必须逐字相同；
    并且输出里不出现这三个字段名。
    """
    pool_a, cat = make_world(["E", "S", "C", "I", "U", "U", "E", "I"])
    pool_b, _ = make_world(["I", "U", "E", "E", "S", "C", "U", "U"])

    def transcript(pool):
        ep = Episode(pool, cat)
        outs = [ep.search("red shoes"), ep.search("model 3"), ep.view("p0"), ep.view("p3"), ep.view("p5")]
        ep.submit(["p0", "p1", "nope", "p0"])
        return outs, ep.last_results

    outs_a, last_a = transcript(pool_a)
    outs_b, last_b = transcript(pool_b)
    assert outs_a == outs_b and last_a == last_b
    blob = "\n".join(outs_a).lower()
    for word in ("label", "gain", "source"):
        assert word not in blob


def test_bm25_deterministic_and_ties_by_product_id():
    docs = [(pid, tokenize("blue pen")) for pid in ["b", "c", "a", "d"]]  # 完全同分
    idx = BM25Index(docs)
    r1 = idx.search(["blue"], 3)
    r2 = BM25Index(list(reversed(docs))).search(["blue"], 3)  # 输入顺序不同
    assert [d for d, _ in r1] == ["a", "b", "c"] == [d for d, _ in r2]
    assert r1 == idx.search(["blue"], 3)


def test_bm25_ranks_more_relevant_first_and_skips_zero_scores():
    docs = [("a", tokenize("red shoes")), ("b", tokenize("red hat")), ("c", tokenize("blue socks"))]
    res = BM25Index(docs).search(tokenize("red shoes"), 10)
    assert [d for d, _ in res] == ["a", "b"]  # c 得分为 0，不返回


def test_pool_index_search_is_repeatable_byte_for_byte():
    pool, cat = make_world()
    outs = {Episode(pool, cat).search("red running shoes") for _ in range(3)}
    assert len(outs) == 1


# ---- 预算与状态机 -------------------------------------------------------------
def test_submit_ok_dedupes_and_counts_dropped():
    pool, cat = make_world()
    ep = Episode(pool, cat)
    ep.search("red")
    ep.submit(["p1", "p1", "ghost", "p0"])
    r = ep.result
    assert r.status == "ok" and r.submitted == ["p1", "p0"] and r.n_dropped_ids == 2 and r.n_search == 1


def test_submit_keeps_first_ten_only():
    prods = [Product(f"p{i:02d}", f"w {i}", None, None, None, None) for i in range(14)]
    pool = Pool("q", "w", "dev", tuple(PoolItem(p.product_id, "U", 0.0, "random") for p in prods), "v1")
    ep = Episode(pool, Catalog(prods))
    ep.submit([p.product_id for p in prods])
    assert len(ep.result.submitted) == 10


def test_budget_notice_after_sixth_call_then_submit_is_ok():
    pool, cat = make_world()
    ep = Episode(pool, cat)
    outs = [ep.search("red") for _ in range(5)]
    assert all(tools.BUDGET_NOTICE not in o for o in outs) and not ep.must_submit
    sixth = ep.view("p0")
    assert sixth.endswith(tools.BUDGET_NOTICE) and ep.must_submit and not ep.done
    ep.submit(["p0"])
    assert ep.result.status == "ok" and ep.result.n_calls == 6


def test_extra_tool_call_after_budget_is_budget_exhausted():
    pool, cat = make_world()
    ep = Episode(pool, cat)
    for _ in range(6):
        ep.search("red")
    assert ep.search("red") == tools.OVER_BUDGET
    assert ep.result.status == "budget_exhausted" and ep.result.n_search == 6  # 第 7 次不计数
    ep.submit(["p0"])  # 已终止，不再改变状态
    assert ep.result.status == "budget_exhausted" and ep.result.submitted == []


def test_ending_without_submit():
    pool, cat = make_world()
    ep = Episode(pool, cat)
    ep.search("red")
    ep.end_without_submit()
    assert ep.result.status == "format_error"
    ep2 = Episode(pool, cat)
    for _ in range(6):
        ep2.search("red")
    ep2.end_without_submit()
    assert ep2.result.status == "budget_exhausted"


def test_format_error_and_truncated_are_terminal():
    pool, cat = make_world()
    ep = Episode(pool, cat)
    ep.submit("not a list")  # type: ignore[arg-type]
    assert ep.result.status == "format_error"
    ep.mark_truncated()
    assert ep.result.status == "format_error"  # 终态不变
    ep2 = Episode(pool, cat)
    ep2.search("red")
    ep2.mark_truncated()
    assert ep2.result.status == "truncated"
    assert ep2.search("red") == tools.FINISHED


def test_custom_budget_is_respected():
    pool, cat = make_world()
    ep = Episode(pool, cat, EnvConfig(max_tool_calls=2))
    ep.search("red")
    assert ep.view("p0").endswith(tools.BUDGET_NOTICE)


def test_truncate_words():
    assert truncate_words("a b c", 3) == "a b c"
    assert truncate_words("a b c d", 3) == "a b c ..."
    assert truncate_words("  a   b ", 5) == "a b"


def test_pool_index_class_exposed():
    pool, cat = make_world()
    assert PoolIndex(pool, cat).search("red shoes", 3) == PoolIndex(pool, cat).search("red shoes", 3)
