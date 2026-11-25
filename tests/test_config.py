import pytest

from searchtune.config import ENV_VERSION, ConfigError, DataConfig, EnvConfig, RewardConfig, from_dict, load_config


def test_unknown_field_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data_version: v1\nn_tset: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="未知字段"):
        load_config(p, DataConfig)


def test_type_error_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("n_dev: many\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="类型不符"):
        load_config(p, DataConfig)
    with pytest.raises(ConfigError):
        from_dict(DataConfig, {"n_dev": True})  # bool 不是 int


def test_defaults_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("n_dev: 7\njaccard_threshold: 1\n", encoding="utf-8")
    cfg = load_config(p, DataConfig)
    assert cfg.n_dev == 7 and cfg.jaccard_threshold == 1.0 and cfg.n_test == 1000
    p.write_text("", encoding="utf-8")
    assert load_config(p, DataConfig) == DataConfig()


def test_env_v1_is_frozen():
    """改动 env_v1 的任何参数都必须同时升 ENV_VERSION 并改这条测试（specs/training.md §2）。"""
    assert ENV_VERSION == "env_v1"
    assert EnvConfig().__dict__ == {
        "max_tool_calls": 6, "search_top_k": 10, "search_title_tokens": 32, "view_bullet_tokens": 200,
        "submit_top_k": 10, "max_gen_tokens_per_turn": 512, "max_context_tokens": 6144,
    }
    assert RewardConfig().__dict__ == {
        "non_ok_reward": -0.2, "bad_weight": 0.3, "bad_top_n": 3, "free_calls": 4, "call_cost": 0.02,
    }


def test_data_defaults_match_spec():
    c = DataConfig()
    assert (c.seed, c.n_test, c.n_dev, c.n_dev_quick) == (20261031, 1000, 500, 200)
    assert (c.n_rl_l1, c.n_sft_traj, c.n_rl_l2) == (8000, 4000, 3000)
    assert (c.n_bm25_distractors, c.n_random_distractors, c.jaccard_threshold) == (50, 50, 0.5)
