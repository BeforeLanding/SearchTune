"""轨迹文件 + 商品池 → metrics.json + EXPERIMENTS.md 行（DESIGN.md §3）。

护栏：环境版本、数据版本、query 覆盖任一不一致，拒绝打分，而不是悄悄缩小分母。
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

from searchtune.config import ENV_VERSION
from searchtune.data.pool import Pool, read_pools
from searchtune.eval.bootstrap import CI, paired_bootstrap
from searchtune.eval.metrics import aggregate, per_query_values
from searchtune.eval.trajectory import read_trajectories
from searchtune.runs import full_commit


class ScoreError(RuntimeError):
    """轨迹文件与环境、数据版本或商品池不一致，拒绝打分。"""


def load_slices(path: str | Path | None) -> dict[str, set[str]] | None:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: {str(q) for q in ids} for name, ids in raw.items()}


def _load(traj_path: str | Path, pools_path: str | Path) -> tuple[dict, list[dict], dict[str, Pool]]:
    meta, rows = read_trajectories(traj_path)
    if meta["env_version"] != ENV_VERSION:
        raise ScoreError(f"轨迹的环境版本 {meta['env_version']!r} 与当前代码 {ENV_VERSION!r} 不一致，拒绝打分")
    pools_list = read_pools(pools_path)
    pools = {p.query_id: p for p in pools_list}
    versions = {p.data_version for p in pools_list}
    if versions != {meta["data_version"]}:
        raise ScoreError(f"轨迹的数据版本 {meta['data_version']!r} 与商品池 {sorted(versions)} 不一致，拒绝打分")
    splits = {p.split for p in pools_list}
    if splits != {meta["split"]}:
        raise ScoreError(f"轨迹的 split {meta['split']!r} 与商品池 {sorted(splits)} 不一致，拒绝打分")
    qids = [str(r["query_id"]) for r in rows]
    if len(set(qids)) != len(qids):
        raise ScoreError("轨迹文件中有重复的 query_id")
    extra = set(qids) - set(pools)
    missing = set(pools) - set(qids)
    if extra:
        raise ScoreError(f"轨迹文件含有商品池里没有的 query：{sorted(extra)[:5]}…")
    if missing:
        raise ScoreError(f"轨迹文件缺少 {len(missing)} 个 query（如 {sorted(missing)[:5]}），不能缩小分母")
    return meta, rows, pools


def score(traj_path: str | Path, pools_path: str | Path, slices_path: str | Path | None = None) -> dict:
    """返回 metrics 字典（specs/eval.md §2 的全部指标 + 元数据）。"""
    meta, rows, pools = _load(traj_path, pools_path)
    out = aggregate(rows, pools, load_slices(slices_path))
    out.update(
        {
            "run_id": meta.get("run_id"),
            "system": meta.get("system"),
            "split": meta["split"],
            "data_version": meta["data_version"],
            "env_version": meta["env_version"],
            "prompt_version": meta.get("prompt_version"),
            "git_commit": meta.get("git_commit") or full_commit(),
            "hardware": meta.get("hardware") or platform.processor() or platform.machine(),
        }
    )
    return out


def compare(
    traj_a: str | Path,
    traj_b: str | Path,
    pools_path: str | Path,
    metric: str = "ndcg",
    n: int = 1000,
    seed: int = 20261031,
) -> CI:
    meta_a, rows_a, pools = _load(traj_a, pools_path)
    meta_b, rows_b, _ = _load(traj_b, pools_path)
    if meta_a["split"] != meta_b["split"]:
        raise ScoreError("两个轨迹文件的 split 不同")
    return paired_bootstrap(
        per_query_values(rows_a, pools, metric),
        per_query_values(rows_b, pools, metric),
        metric=metric,
        n=n,
        seed=seed,
    )


EXPERIMENTS_HEADER = (
    "| run_id | 系统 | split | 平台 | 数据 | 环境 | prompt | NDCG@10 | E-Recall@10 | BadRate@3 | "
    "U 占比 | 调用/ep | 非 ok 率 | 结论 |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def experiments_row(m: dict, platform_name: str, note: str = "") -> str:
    """可直接粘贴进 EXPERIMENTS.md 的一行（eval §9）。"""
    non_ok = 1.0 - m["ok_rate"]
    split = m["split"] + ("（test）" if m["split"] == "test" else "")
    return (
        f"| {m.get('run_id') or '-'} | {m.get('system') or '-'} | {split} | {platform_name} | "
        f"{m['data_version']} | {m['env_version']} | {m.get('prompt_version') or '-'} | "
        f"{m['ndcg_at_10']:.4f} | {m['e_recall_at_10']:.4f} | {m['bad_rate_at_3']:.4f} | "
        f"{m['u_share_at_10']:.4f} | {m['n_calls_mean']:.2f} | {non_ok:.4f} | {note} |"
    )


def append_experiments(path: str | Path, row: str) -> None:
    p = Path(path)
    if not p.exists():
        p.write_text(
            "# 实验记录\n\n每次训练或评测一行，失败的也记。B0 等非 LLM 基线与硬件无关，平台记 `local-cpu`；"
            "涉及 GPU 的数字以云端为准（AGENTS.md）。\n\n" + EXPERIMENTS_HEADER,
            encoding="utf-8",
            newline="\n",
        )
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(row + "\n")
