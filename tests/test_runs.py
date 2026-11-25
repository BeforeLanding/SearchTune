import subprocess

import pytest

from searchtune.runs import GuardError, guard_test_split, is_dirty, make_run_id, short_commit


def _git(cwd, *a):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.txt").write_text("x")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def test_run_id_clean_and_dirty(repo):
    rid = make_run_id("E1", "v1", repo)
    assert rid == f"E1-v1-{short_commit(repo)}" and not is_dirty(repo)
    (repo / "a.txt").write_text("changed")
    assert make_run_id("E1", "v1", repo) == rid + "-dirty"


def test_untracked_file_counts_as_dirty(repo):
    (repo / "new.txt").write_text("x")
    assert is_dirty(repo)


def test_non_git_dir_is_dirty(tmp_path):
    assert is_dirty(tmp_path) and make_run_id("E1", "v1", tmp_path).endswith("-dirty")


def test_test_split_requires_flag(repo):
    with pytest.raises(GuardError, match="--allow-test"):
        guard_test_split("test", False, repo)


def test_test_split_refuses_dirty_tree(repo):
    (repo / "a.txt").write_text("changed")
    with pytest.raises(GuardError, match="未提交"):
        guard_test_split("test", True, repo)


def test_test_split_allowed_when_clean_and_flagged(repo):
    guard_test_split("test", True, repo)
    guard_test_split("dev", False, repo)  # 非 test 不受限，脏工作区也行
