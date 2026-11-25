"""run_id、run 目录与 git 状态检查、测试集护栏（DESIGN.md §5、§6）。"""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Any

import yaml


class GuardError(RuntimeError):
    """护栏拒绝执行（测试集保护等）。"""


def _git(args: list[str], cwd: Path | str | None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout


def short_commit(cwd: Path | str | None = None) -> str:
    out = _git(["rev-parse", "--short=8", "HEAD"], cwd)
    return out.strip() if out else "nogit"


def full_commit(cwd: Path | str | None = None) -> str:
    out = _git(["rev-parse", "HEAD"], cwd)
    return out.strip() if out else "nogit"


def is_dirty(cwd: Path | str | None = None) -> bool:
    """工作区有未提交改动（含未被 gitignore 的未跟踪文件）。非 git 目录视为 dirty。"""
    out = _git(["status", "--porcelain"], cwd)
    if out is None:
        return True
    return bool(out.strip())


def make_run_id(exp_id: str, data_version: str, cwd: Path | str | None = None) -> str:
    """``<实验ID>-<data_version>-<短 commit>``；工作区有改动时加 ``-dirty``。"""
    rid = f"{exp_id}-{data_version}-{short_commit(cwd)}"
    if is_dirty(cwd):
        rid += "-dirty"
    return rid


def run_dir(root: Path | str, run_id: str) -> Path:
    return Path(root) / "runs" / run_id


def write_run_config(directory: Path | str, cfg: Any) -> Path:
    """把最终生效的配置写入 ``<run_dir>/config.yaml``。"""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    path = d / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=True), encoding="utf-8")
    return path


def guard_test_split(split: str, allow_test: bool, cwd: Path | str | None = None) -> None:
    """eval §1：``--split test`` 必须显式 ``--allow-test``，且工作区干净。"""
    if split != "test":
        return
    if not allow_test:
        raise GuardError("--split test 必须显式加 --allow-test（specs/eval.md §1：测试集只读）")
    if is_dirty(cwd):
        raise GuardError("工作区有未提交改动，拒绝在 test 上运行；请先提交（DESIGN.md §6）")
