"""下载 ESCI 原始 parquet 到 data/raw/（specs/data.md §1；T2）。

只用标准库。文件在 Git LFS 上，走 media.githubusercontent.com。支持断点续传，
下载完成后核对大小并打印 sha256；sha256 与许可证由 `prepare` 写入 data/MANIFEST.json。

    python scripts/download_esci.py [--dest data/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/"
FILES = ["shopping_queries_dataset_examples.parquet", "shopping_queries_dataset_products.parquet"]


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download(url: str, dest: Path) -> None:
    total = remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if have == total:
        print(f"已存在且大小一致：{dest.name}（{total:,} 字节）")
        return
    if have > total:
        dest.unlink()
        have = 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab" if have else "wb") as f:
        if have and r.status != 206:  # 服务器不支持续传，重来
            f.close()
            dest.unlink()
            return download(url, dest)
        while block := r.read(1 << 20):
            f.write(block)
    if dest.stat().st_size != total:
        sys.exit(f"{dest.name} 下载不完整：{dest.stat().st_size:,} / {total:,}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(BASE + name, args.dest / name)
        print(f"{name}  sha256={sha256(args.dest / name)}")


if __name__ == "__main__":
    main()
