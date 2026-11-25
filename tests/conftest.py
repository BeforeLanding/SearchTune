from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixture_data import build_raw, fixture_config  # noqa: E402

from searchtune.data.manifest import Layout  # noqa: E402
from searchtune.data.prepare import prepare  # noqa: E402


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    """在合成 fixture 上跑一遍完整的 prepare，供多个测试共用（只读）。"""
    root = tmp_path_factory.mktemp("data")
    facts = build_raw(root / "raw")
    cfg = fixture_config()
    layout = Layout(root, cfg.data_version)
    stats = prepare(cfg, layout)
    return {"root": root, "layout": layout, "cfg": cfg, "stats": stats, "facts": facts}
