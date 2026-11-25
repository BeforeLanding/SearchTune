import pytest

from searchtune.eval.bootstrap import paired_bootstrap


def _vals(n, shift=0.0):
    return {str(i): ((i * 37) % 100) / 100 + shift for i in range(n)}


def test_deterministic_bit_for_bit():
    a, b = _vals(200, 0.05), _vals(200)
    r1 = paired_bootstrap(a, b, n=300, seed=7)
    r2 = paired_bootstrap(a, b, n=300, seed=7)
    assert r1 == r2 and (r1.lo, r1.hi) == (r2.lo, r2.hi)
    assert paired_bootstrap(a, b, n=300, seed=8) != r1  # 换 seed 结果会变


def test_constant_gain_gives_degenerate_significant_interval():
    r = paired_bootstrap(_vals(100, 0.1), _vals(100), n=100, seed=1)
    assert r.diff == pytest.approx(0.1) and r.lo == pytest.approx(0.1) and r.hi == pytest.approx(0.1)
    assert r.a_significantly_better and r.n_queries == 100


def test_no_difference_is_not_significant():
    r = paired_bootstrap(_vals(100), _vals(100), n=100, seed=1)
    assert r.diff == 0 and not r.a_significantly_better


def test_interval_contains_diff_and_is_ordered():
    a = {str(i): float(i % 3 == 0) for i in range(150)}
    b = {str(i): float(i % 4 == 0) for i in range(150)}
    r = paired_bootstrap(a, b, n=500, seed=3)
    assert r.lo <= r.diff <= r.hi


def test_mismatched_queries_rejected():
    with pytest.raises(ValueError, match="不一致"):
        paired_bootstrap({"1": 0.1}, {"2": 0.1})


def test_undefined_queries_are_skipped():
    r = paired_bootstrap({"1": 1.0, "2": None, "3": 0.0}, {"1": 0.0, "2": None, "3": 0.0}, n=50, seed=1)
    assert r.n_queries == 2
