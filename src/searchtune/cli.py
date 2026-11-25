"""统一入口：``python -m searchtune <子命令>``（DESIGN.md §5）。

共 7 个子命令。``rollout`` 目前只实现了非 LLM 基线 B0；LLM rollout、``gen-traj``、
``train-ce``、``train`` 依赖云端 GPU 环境（T0）与 ``gpu`` / ``rerank`` extras，
在对应任务完成前会明确报“未实现”，不用假实现冒充。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from searchtune.config import DataConfig, load_config
from searchtune.data.manifest import Layout, verify_artifacts
from searchtune.runs import GuardError

SUBCOMMANDS = ("prepare", "rollout", "score", "compare", "gen-traj", "train-ce", "train")


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", default="data", help="数据根目录（默认 data）")
    p.add_argument("--data-version", default="v1")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="searchtune", description="SearchTune：商品搜索 Agent 的 RL 后训练")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="<子命令>")

    p = sub.add_parser("prepare", help="raw → catalog、pools、splits、slices、MANIFEST、STATS")
    _add_data_args(p)
    p.add_argument("--config", help="DataConfig 的 YAML；缺省用默认值（即 specs/data.md 的参数）")
    p.add_argument("--raw-dir", help="原始 parquet 目录（默认 <data-root>/raw）")
    p.add_argument("--verify-only", action="store_true", help="只校验现有产物的 sha256 是否与 MANIFEST 一致")

    p = sub.add_parser("rollout", help="跑一个系统，写轨迹文件（目前只支持 --system b0）")
    _add_data_args(p)
    p.add_argument("--system", choices=["b0", "llm"], required=True)
    p.add_argument("--split", choices=["dev-quick", "dev", "test"], required=True)
    p.add_argument("--allow-test", action="store_true", help="在 test 上运行必须显式加此参数，且工作区必须干净")
    p.add_argument("--no-verify", action="store_true", help="跳过 sha256 校验（仅调试）")

    p = sub.add_parser("score", help="轨迹文件 + 商品池 → 指标")
    _add_data_args(p)
    p.add_argument("--traj", required=True)
    p.add_argument("--pools", help="商品池文件；缺省按轨迹元数据里的 split 推断")
    p.add_argument("--slices", help="切片成员 JSON；缺省按 split 推断（有则用）")
    p.add_argument("--out", help="metrics.json 输出路径；缺省写到 run 目录的 metrics/<split>.json")
    p.add_argument("--experiments", default="EXPERIMENTS.md", help="追加记录行的文件；'-' 表示不写")
    p.add_argument("--platform", default="local-cpu", help="记录行里的运行平台")
    p.add_argument("--note", default="")

    p = sub.add_parser("compare", help="两个轨迹文件的配对 bootstrap")
    _add_data_args(p)
    p.add_argument("--traj-a", required=True)
    p.add_argument("--traj-b", required=True)
    p.add_argument("--pools")
    p.add_argument("--metric", default="ndcg", choices=["ndcg", "e_recall", "bad3"])
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261031)

    for name, hint in (
        ("gen-traj", "B4 采样 + 拒绝采样 → SFT 数据（T11，需要云端 GPU）"),
        ("train-ce", "训练 B3 的 DeBERTa 精排模型（T7，需要 rerank extras）"),
        ("train", "转调 verl：L1 / L2 / SFT（T9、T11、T12，需要云端 GPU）"),
    ):
        sub.add_parser(name, help=hint)
    return ap


def _traj_defaults(traj: str, layout: Layout, pools: str | None, slices: str | None) -> tuple[Path, Path | None, dict]:
    from searchtune.eval.trajectory import read_trajectories

    meta, _ = read_trajectories(traj)
    split = meta["split"]
    pools_path = Path(pools) if pools else layout.pools(split)
    slices_path = Path(slices) if slices else layout.slices(split)
    return pools_path, (slices_path if slices_path.exists() else None), meta


def cmd_prepare(a: argparse.Namespace) -> int:
    layout = Layout(a.data_root, a.data_version)
    if a.verify_only:
        from searchtune.data.manifest import load_manifest

        arts = load_manifest(layout.manifest)["versions"][a.data_version]["artifacts"]
        verify_artifacts(layout, [layout.root / rel for rel in arts])
        print(f"{len(arts)} 个产物的 sha256 与 MANIFEST 一致")
        return 0
    from searchtune.data.prepare import prepare

    cfg = load_config(a.config, DataConfig) if a.config else DataConfig(data_version=a.data_version)
    if cfg.data_version != a.data_version:
        raise SystemExit(f"配置的 data_version={cfg.data_version!r} 与 --data-version={a.data_version!r} 不一致")
    stats = prepare(cfg, layout, Path(a.raw_dir) if a.raw_dir else None, Path.cwd())
    print(f"完成：{layout.processed}；统计见 {layout.stats}")
    print(json.dumps(stats["n_queries"], ensure_ascii=False))
    return 0


def cmd_rollout(a: argparse.Namespace) -> int:
    if a.system == "llm":
        raise SystemExit("LLM rollout 尚未实现：依赖 T0（云端 GPU 环境与模型检查）和 T6，见 TASKS.md")
    from searchtune.baselines.bm25_raw import rollout_b0

    layout = Layout(a.data_root, a.data_version)
    path = rollout_b0(a.split, layout, allow_test=a.allow_test, repo_root=Path.cwd(), verify=not a.no_verify)
    print(path)
    return 0


def cmd_score(a: argparse.Namespace) -> int:
    from searchtune.eval.score import append_experiments, experiments_row, score

    layout = Layout(a.data_root, a.data_version)
    pools, slices, meta = _traj_defaults(a.traj, layout, a.pools, a.slices)
    m = score(a.traj, pools, slices)
    out = Path(a.out) if a.out else Path(a.traj).parent.parent / "metrics" / f"{meta['split']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    row = experiments_row(m, a.platform, a.note)
    print(json.dumps({k: m[k] for k in ("ndcg_at_10", "e_recall_at_10", "bad_rate_at_3", "u_share_at_10", "ok_rate")}))
    print(row)
    if a.experiments != "-":
        append_experiments(a.experiments, row)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    from searchtune.eval.score import compare

    layout = Layout(a.data_root, a.data_version)
    pools, _, _ = _traj_defaults(a.traj_a, layout, a.pools, None)
    ci = compare(a.traj_a, a.traj_b, pools, a.metric, a.n, a.seed)
    print(json.dumps(ci.to_dict(), ensure_ascii=False, indent=1))
    return 0


def cmd_not_ready(a: argparse.Namespace) -> int:
    raise SystemExit(f"`{a.cmd}` 尚未实现：属于云端 GPU 阶段的任务，依赖 T0；见 TASKS.md")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    handlers = {
        "prepare": cmd_prepare,
        "rollout": cmd_rollout,
        "score": cmd_score,
        "compare": cmd_compare,
    }
    try:
        return handlers.get(args.cmd, cmd_not_ready)(args)
    except GuardError as e:
        print(f"护栏拒绝：{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
