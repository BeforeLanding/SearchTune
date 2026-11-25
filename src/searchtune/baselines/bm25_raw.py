"""B0：用原始 query 调一次 ``search``，直接提交前 10（specs/eval.md §5）。

非 LLM 基线，同样只产出轨迹文件，指标交给统一的打分器。
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from searchtune.config import ENV_VERSION, EnvConfig
from searchtune.data.manifest import Layout, verify_artifacts
from searchtune.data.pool import Catalog, Pool, read_pools
from searchtune.env.episode import Episode
from searchtune.eval.trajectory import write_trajectories
from searchtune.runs import full_commit, guard_test_split, make_run_id, run_dir, write_run_config


def run_b0(pools: Iterable[Pool], catalog: Catalog, env_cfg: EnvConfig | None = None) -> list[dict]:
    rows = []
    for pool in pools:
        t0 = time.perf_counter()
        ep = Episode(pool, catalog, env_cfg)
        ep.search(pool.query)
        ep.submit(ep.last_results)
        res = ep.result
        rows.append(
            {
                "query_id": pool.query_id,
                "submitted": res.submitted,
                "status": res.status,
                "turns": [],
                "n_search": res.n_search,
                "n_view": res.n_view,
                "gen_tokens": 0,
                "wall_time_s": round(time.perf_counter() - t0, 6),
            }
        )
    return rows


def rollout_b0(
    split: str,
    layout: Layout,
    allow_test: bool = False,
    repo_root: Path | str | None = None,
    out_root: Path | str = ".",
    verify: bool = True,
) -> Path:
    """读 split 的商品池，跑 B0，写 ``runs/<run_id>/traj/<split>.jsonl``，返回轨迹路径。"""
    guard_test_split(split, allow_test, repo_root)
    pools_path = layout.pools(split)
    if verify:
        verify_artifacts(layout, [pools_path, layout.catalog])
    pools = read_pools(pools_path)
    catalog = Catalog.load(layout.catalog, {p for pool in pools for p in pool.product_ids})
    env_cfg = EnvConfig()
    run_id = make_run_id("B0", layout.version, repo_root)
    rdir = run_dir(out_root, run_id)
    write_run_config(rdir, {"system": "b0", "split": split, "env": env_cfg.__dict__})
    meta = {
        "run_id": run_id,
        "system": "B0",
        "split": split,
        "data_version": layout.version,
        "env_version": ENV_VERSION,
        "prompt_version": None,
        "git_commit": full_commit(repo_root),
        "hardware": "cpu",
    }
    path = rdir / "traj" / f"{split}.jsonl"
    write_trajectories(path, meta, run_b0(pools, catalog, env_cfg))
    return path
