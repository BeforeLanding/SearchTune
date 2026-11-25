"""合成 fixture 上的端到端：prepare → B0 rollout → score → compare，以及各项护栏。

这是接线与护栏的集成测试，里面的数字不代表真实数据上的任何结论。
"""

import json
import subprocess
import sys

import pytest

from searchtune.baselines.bm25_raw import rollout_b0
from searchtune.cli import main
from searchtune.config import ENV_VERSION
from searchtune.data.manifest import Layout, ManifestError
from searchtune.data.pool import read_pools
from searchtune.eval.score import ScoreError, compare, experiments_row, score
from searchtune.eval.trajectory import TrajectoryError, read_trajectories, write_trajectories
from searchtune.runs import GuardError


@pytest.fixture(scope="module")
def traj(prepared, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    path = rollout_b0("dev", prepared["layout"], repo_root=out, out_root=out)
    return path


def test_b0_trajectory_protocol(prepared, traj):
    meta, rows = read_trajectories(traj)
    assert meta["env_version"] == ENV_VERSION and meta["split"] == "dev" and meta["system"] == "B0"
    pools = {p.query_id: p for p in read_pools(prepared["layout"].pools("dev"))}
    assert {r["query_id"] for r in rows} == set(pools)
    for r in rows:
        assert r["status"] == "ok" and r["turns"] == [] and r["n_search"] == 1 and r["n_view"] == 0
        assert r["gen_tokens"] == 0 and len(r["submitted"]) <= 10
        assert set(r["submitted"]) <= set(pools[r["query_id"]].product_ids)


def test_b0_is_deterministic_apart_from_timing(prepared, tmp_path):
    a = rollout_b0("dev", prepared["layout"], repo_root=tmp_path, out_root=tmp_path / "a")
    b = rollout_b0("dev", prepared["layout"], repo_root=tmp_path, out_root=tmp_path / "b")

    def strip(p):
        meta, rows = read_trajectories(p)
        return [{k: v for k, v in r.items() if k != "wall_time_s"} for r in rows]

    assert strip(a) == strip(b)


def test_score_full_metrics(prepared, traj):
    lay = prepared["layout"]
    m = score(traj, lay.pools("dev"), lay.slices("dev"))
    for k in ("ndcg_at_10", "e_recall_at_10", "bad_rate_at_3", "u_share_at_10", "n_search_mean", "n_view_mean",
              "gen_tokens_mean", "wall_time_s_p90", "format_error_rate", "budget_exhausted_rate", "truncated_rate", "slices"):
        assert k in m
    assert 0.0 <= m["ndcg_at_10"] <= 1.0 and m["n_queries"] == prepared["cfg"].n_dev and m["n_calls_mean"] == 1.0
    assert m["env_version"] == ENV_VERSION and m["data_version"] == "v1" and m["split"] == "dev"
    row = experiments_row(m, "local-cpu")
    assert row.startswith("| B0-v1-") and "local-cpu" in row and row.count("|") == 15


def test_score_refuses_env_version_mismatch(prepared, traj, tmp_path):
    meta, rows = read_trajectories(traj)
    bad = tmp_path / "bad.jsonl"
    write_trajectories(bad, {**meta, "env_version": "env_v0"}, rows)
    with pytest.raises(ScoreError, match="环境版本"):
        score(bad, prepared["layout"].pools("dev"))


def test_score_refuses_data_version_or_split_mismatch(prepared, traj, tmp_path):
    meta, rows = read_trajectories(traj)
    p1 = tmp_path / "v.jsonl"
    write_trajectories(p1, {**meta, "data_version": "v9"}, rows)
    with pytest.raises(ScoreError, match="数据版本"):
        score(p1, prepared["layout"].pools("dev"))
    p2 = tmp_path / "s.jsonl"
    write_trajectories(p2, {**meta, "split": "test"}, rows)
    with pytest.raises(ScoreError, match="split"):
        score(p2, prepared["layout"].pools("dev"))


def test_score_refuses_missing_extra_and_duplicate_queries(prepared, traj, tmp_path):
    meta, rows = read_trajectories(traj)
    pools = prepared["layout"].pools("dev")
    p = tmp_path / "missing.jsonl"
    write_trajectories(p, meta, rows[:-1])
    with pytest.raises(ScoreError, match="缺少"):
        score(p, pools)
    p = tmp_path / "extra.jsonl"
    write_trajectories(p, meta, rows + [{**rows[0], "query_id": "999999"}])
    with pytest.raises(ScoreError, match="没有的 query"):
        score(p, pools)
    p = tmp_path / "dup.jsonl"
    write_trajectories(p, meta, rows + [rows[0]])
    with pytest.raises(ScoreError, match="重复"):
        score(p, pools)


def test_failed_episodes_lower_the_mean(prepared, traj, tmp_path):
    meta, rows = read_trajectories(traj)
    pools = prepared["layout"].pools("dev")
    base = score(traj, pools)["ndcg_at_10"]
    broken = [{**r, "status": "format_error"} if i == 0 else r for i, r in enumerate(rows)]
    p = tmp_path / "broken.jsonl"
    write_trajectories(p, meta, broken)
    m = score(p, pools)
    assert m["n_queries"] == len(rows) and m["format_error_rate"] == pytest.approx(1 / len(rows))
    assert m["ndcg_at_10"] <= base


def test_trajectory_reader_rejects_missing_meta(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"query_id": "1", "submitted": [], "status": "ok"}) + "\n", encoding="utf-8")
    with pytest.raises(TrajectoryError):
        read_trajectories(p)


def test_compare_self_and_degraded(prepared, traj, tmp_path):
    meta, rows = read_trajectories(traj)
    pools = prepared["layout"].pools("dev")
    same = compare(traj, traj, pools, n=100)
    assert same.diff == 0 and not same.a_significantly_better and same.n_queries == prepared["cfg"].n_dev
    worse = tmp_path / "worse.jsonl"
    write_trajectories(worse, meta, [{**r, "submitted": [], "status": "format_error"} for r in rows])
    ci = compare(traj, worse, pools, n=200)
    assert ci.diff > 0 and ci.lo <= ci.diff <= ci.hi
    assert compare(traj, worse, pools, n=200) == ci  # 同 seed 逐位一致


def test_rollout_test_split_guards(prepared, tmp_path):
    lay = prepared["layout"]
    with pytest.raises(GuardError, match="--allow-test"):
        rollout_b0("test", lay, allow_test=False, repo_root=tmp_path, out_root=tmp_path)
    # tmp_path 不是 git 仓库 → 视为 dirty，即使加了 --allow-test 也拒绝
    with pytest.raises(GuardError, match="未提交"):
        rollout_b0("test", lay, allow_test=True, repo_root=tmp_path, out_root=tmp_path)


def test_rollout_test_split_runs_when_clean_and_flagged(prepared, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for cmd in (["init", "-q"], ["config", "user.email", "t@e.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True, capture_output=True)
    (repo / "f").write_text("x")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "i"], cwd=repo, check=True, capture_output=True)
    (repo / ".gitignore").write_text("runs/\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "ignore"], cwd=repo, check=True, capture_output=True)
    path = rollout_b0("test", prepared["layout"], allow_test=True, repo_root=repo, out_root=repo)
    meta, rows = read_trajectories(path)
    assert meta["split"] == "test" and not meta["run_id"].endswith("-dirty") and len(rows) == prepared["cfg"].n_test


def test_rollout_refuses_tampered_data(prepared, tmp_path):
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(prepared["root"], root)
    lay = Layout(root, "v1")
    with open(lay.pools("dev"), "a", encoding="utf-8") as f:
        f.write("\n\n")
    with pytest.raises(ManifestError, match="sha256"):
        rollout_b0("dev", lay, repo_root=tmp_path, out_root=tmp_path)


def test_cli_end_to_end(prepared, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = str(prepared["root"])
    assert main(["rollout", "--system", "b0", "--split", "dev-quick", "--data-root", root]) == 0
    path = capsys.readouterr().out.strip().splitlines()[-1]
    assert main(["score", "--traj", path, "--data-root", root, "--experiments", "EXP.md"]) == 0
    assert "NDCG@10" in (tmp_path / "EXP.md").read_text(encoding="utf-8")
    assert (tmp_path / "runs").exists() and list((tmp_path / "runs").glob("*/metrics/dev-quick.json"))
    assert main(["compare", "--traj-a", path, "--traj-b", path, "--data-root", root, "--n", "50"]) == 0
    assert main(["prepare", "--verify-only", "--data-root", root]) == 0
    assert main(["rollout", "--system", "b0", "--split", "test", "--data-root", root]) == 2  # 缺 --allow-test


def test_cli_help_lists_all_seven_subcommands():
    out = subprocess.run([sys.executable, "-m", "searchtune", "--help"], capture_output=True, text=True, encoding="utf-8").stdout
    for name in ("prepare", "rollout", "score", "compare", "gen-traj", "train-ce", "train"):
        assert name in out


def test_unready_subcommands_fail_loudly():
    for name in ("gen-traj", "train-ce", "train"):
        with pytest.raises(SystemExit, match="尚未实现"):
            main([name])
    with pytest.raises(SystemExit, match="尚未实现"):
        main(["rollout", "--system", "llm", "--split", "dev"])
