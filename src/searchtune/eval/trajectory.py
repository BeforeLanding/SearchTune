"""轨迹文件读写（specs/eval.md §3）。

文件第一行是元数据 ``{"_meta": {...}}``，其后每行一个 episode。元数据必须包含
``env_version``、``data_version``、``split``。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

REQUIRED_META = ("env_version", "data_version", "split")
REQUIRED_ROW = ("query_id", "submitted", "status")


class TrajectoryError(ValueError):
    pass


def write_trajectories(path: str | Path, meta: dict, rows: Iterable[dict]) -> int:
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise TrajectoryError(f"元数据缺少 {missing}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False, sort_keys=True) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_trajectories(path: str | Path) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    if not lines:
        raise TrajectoryError(f"{path} 是空文件")
    head = json.loads(lines[0])
    if "_meta" not in head:
        raise TrajectoryError(f"{path} 第一行不是 _meta 元数据")
    meta = head["_meta"]
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise TrajectoryError(f"元数据缺少 {missing}")
    rows = [json.loads(ln) for ln in lines[1:]]
    for i, r in enumerate(rows):
        miss = [k for k in REQUIRED_ROW if k not in r]
        if miss:
            raise TrajectoryError(f"第 {i + 2} 行缺少字段 {miss}")
    return meta, rows
