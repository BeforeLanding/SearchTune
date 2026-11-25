"""数据目录布局与 MANIFEST.json 的校验（specs/data.md §7、§8）。

``MANIFEST.json`` 结构::

    {"license": {...}, "raw_files": {name: {sha256, bytes}},
     "versions": {"v1": {"code_commit", "seed", "params", "artifacts": {relpath: {sha256, rows}}}}}

``relpath`` 相对 ``data_root``，用正斜杠。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ManifestError(RuntimeError):
    """文件的 sha256 与 MANIFEST 不一致，或 MANIFEST 中没有该文件。"""


def sha256_file(path: str | Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def count_rows(path: str | Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for ln in f if ln.strip())


class Layout:
    """``data_root`` 下的目录约定。"""

    def __init__(self, data_root: str | Path = "data", version: str = "v2"):
        self.root = Path(data_root)
        self.version = version

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed" / self.version

    @property
    def catalog(self) -> Path:
        return self.processed / "catalog.jsonl"

    def pools(self, split: str) -> Path:
        return self.processed / "pools" / f"{split}.jsonl"

    def split_list(self, name: str) -> Path:
        return self.root / "splits" / self.version / f"{name}.txt"

    def slices(self, split: str) -> Path:
        return self.root / "slices" / self.version / f"{split}.json"

    @property
    def manifest(self) -> Path:
        return self.root / "MANIFEST.json"

    @property
    def stats(self) -> Path:
        return self.root / "STATS.md"

    def rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


def load_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"找不到 {p}；请先运行 prepare")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_artifacts(layout: Layout, paths: list[Path]) -> None:
    """逐个校验文件的 sha256 与 MANIFEST 一致（训练、rollout、打分读取数据前调用）。"""
    manifest = load_manifest(layout.manifest)
    ver = manifest.get("versions", {}).get(layout.version)
    if ver is None:
        raise ManifestError(f"MANIFEST 中没有数据版本 {layout.version!r}")
    for p in paths:
        rel = layout.rel(p)
        entry = ver["artifacts"].get(rel)
        if entry is None:
            raise ManifestError(f"MANIFEST 中没有 {rel}")
        if not p.exists():
            raise ManifestError(f"{rel} 不存在")
        actual = sha256_file(p)
        if actual != entry["sha256"]:
            raise ManifestError(f"{rel} 的 sha256 与 MANIFEST 不一致（文件被改动或版本不匹配）")
