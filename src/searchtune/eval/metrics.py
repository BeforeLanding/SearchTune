"""评测指标（specs/eval.md §2）。奖励函数直接复用这里的 ``ndcg_at_10``（DESIGN.md §1）。

约定：只有 ``status == "ok"`` 的 episode 才按其提交计分，其余一律按空提交计算，
并且**计入分母**。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from searchtune.data.pool import Pool

K = 10
OK = "ok"
STATUSES = ("ok", "format_error", "budget_exhausted", "truncated")


def normalize_submission(ids: Sequence[str], pool: Pool, k: int = K) -> tuple[list[str], int]:
    """去重（保留首次出现）、丢弃不在池中的 id，再取前 k 个。返回 (列表, 被丢弃的 id 数)。

    ``env.Episode.submit`` 与打分器共用这一个函数，两边的口径不可能不一致。
    """
    seen: set[str] = set()
    out: list[str] = []
    dropped = 0
    for pid in ids:
        if pid in seen or pid not in pool:
            dropped += 1
            continue
        seen.add(pid)
        out.append(pid)
    return out[:k], dropped


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_10(submitted: Sequence[str], pool: Pool) -> float:
    """NDCG@10，线性增益；提交不足 10 个时空位按 0；理想排序用池内全部商品的真实增益。"""
    ids, _ = normalize_submission(submitted, pool)
    ideal = sorted((it.gain for it in pool.items), reverse=True)[:K]
    idcg = _dcg(ideal)
    if idcg <= 0:
        return 0.0
    return _dcg([pool.gain(p) for p in ids]) / idcg


def e_recall_at_10(submitted: Sequence[str], pool: Pool) -> float | None:
    """前 10 个中 E 的数量 / min(池中 E 的数量, 10)。池中没有 E 时无定义，返回 None。"""
    n_e = sum(1 for it in pool.items if it.label == "E")
    if n_e == 0:
        return None
    ids, _ = normalize_submission(submitted, pool)
    return sum(1 for p in ids if pool.label(p) == "E") / min(n_e, K)


def bad_at_3(submitted: Sequence[str], pool: Pool) -> bool:
    """前 3 个中是否出现任何 I 商品（U 不算）。"""
    ids, _ = normalize_submission(submitted, pool)
    return any(pool.label(p) == "I" for p in ids[:3])


def n_bad_in_top(submitted: Sequence[str], pool: Pool, n: int = 3) -> int:
    ids, _ = normalize_submission(submitted, pool)
    return sum(1 for p in ids[:n] if pool.label(p) == "I")


def u_share_at_10(submitted: Sequence[str], pool: Pool) -> float | None:
    """前 10 个中干扰商品（U）的比例；空提交无定义，返回 None。"""
    ids, _ = normalize_submission(submitted, pool)
    if not ids:
        return None
    return sum(1 for p in ids if pool.label(p) == "U") / len(ids)


@dataclass(frozen=True)
class EpisodeMetrics:
    query_id: str
    status: str
    ndcg: float
    e_recall: float | None
    bad3: bool
    u_share: float | None


def episode_metrics(row: dict, pool: Pool) -> EpisodeMetrics:
    """单个 episode 的指标。非 ``ok`` 一律按空提交。"""
    status = row["status"]
    if status not in STATUSES:
        raise ValueError(f"未知 status {status!r}")
    submitted = row["submitted"] if status == OK else []
    return EpisodeMetrics(
        query_id=str(row["query_id"]),
        status=status,
        ndcg=ndcg_at_10(submitted, pool),
        e_recall=e_recall_at_10(submitted, pool),
        bad3=bad_at_3(submitted, pool),
        u_share=u_share_at_10(submitted, pool),
    )


def percentile(values: Sequence[float], q: float) -> float:
    """线性插值分位数，q ∈ [0, 100]。空序列返回 nan。"""
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(rows: Sequence[dict], pools: dict[str, Pool], slices: dict[str, set[str]] | None = None) -> dict:
    """汇总 specs/eval.md §2 的全部指标。``rows`` 与 ``pools`` 必须一一对应（由 score 保证）。"""
    per = [episode_metrics(r, pools[str(r["query_id"])]) for r in rows]
    n = len(per)
    out: dict = {"n_queries": n}
    out["ndcg_at_10"] = _mean([m.ndcg for m in per])
    er = [m.e_recall for m in per if m.e_recall is not None]
    out["e_recall_at_10"] = _mean(er)
    out["e_recall_n_defined"] = len(er)
    out["bad_rate_at_3"] = _mean([1.0 if m.bad3 else 0.0 for m in per])
    us = [m.u_share for m in per if m.u_share is not None]
    out["u_share_at_10"] = _mean(us)
    out["u_share_n_defined"] = len(us)
    for s in STATUSES[1:]:
        out[f"{s}_rate"] = sum(1 for m in per if m.status == s) / n if n else float("nan")
    out["ok_rate"] = sum(1 for m in per if m.status == OK) / n if n else float("nan")
    ns = [r.get("n_search", 0) for r in rows]
    nv = [r.get("n_view", 0) for r in rows]
    out["n_search_mean"] = _mean(ns)
    out["n_view_mean"] = _mean(nv)
    out["n_calls_mean"] = _mean([a + b for a, b in zip(ns, nv)])
    tok = [float(r.get("gen_tokens", 0)) for r in rows]
    wall = [float(r.get("wall_time_s", 0.0)) for r in rows]
    out["gen_tokens_mean"] = _mean(tok)
    out["gen_tokens_p90"] = percentile(tok, 90)
    out["wall_time_s_mean"] = _mean(wall)
    out["wall_time_s_p90"] = percentile(wall, 90)
    if slices:
        sl = {}
        for name, qids in sorted(slices.items()):
            sub = [m.ndcg for m in per if m.query_id in qids]
            sl[name] = {"n": len(sub), "ndcg_at_10": _mean(sub)}
        out["slices"] = sl
    return out


def per_query_values(rows: Sequence[dict], pools: dict[str, Pool], metric: str) -> dict[str, float | None]:
    """按 query 返回单个指标的取值，供配对 bootstrap 使用。"""
    pick = {
        "ndcg": lambda m: m.ndcg,
        "e_recall": lambda m: m.e_recall,
        "bad3": lambda m: 1.0 if m.bad3 else 0.0,
    }
    if metric not in pick:
        raise ValueError(f"不支持的指标 {metric!r}，可选 {sorted(pick)}")
    out: dict[str, float | None] = {}
    for r in rows:
        m = episode_metrics(r, pools[str(r["query_id"])])
        out[m.query_id] = pick[metric](m)
    return out
