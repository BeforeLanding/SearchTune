# 设计

> 本文只定代码结构和模块之间的约定。数据、评测、训练的“做什么”分别以 `specs/data.md`、`specs/eval.md`、`specs/training.md` 为准；本文与它们冲突时改本文。

## 1. 数据流

```
data/raw ──prepare──▶ catalog.jsonl + pools/<split>.jsonl + splits/*.txt
                                   │
                                   ▼
                        env（商品池 + 工具 + 奖励）
                     ┌─────────────┼──────────────────────┐
                rollout（评测）  verl 训练（L1 / L2）   gen-traj（B4 → 冷启动 SFT 数据）
                     │
                     ▼
        runs/<run_id>/traj/<split>.jsonl ──score / compare──▶ metrics/<split>.json → EXPERIMENTS.md
```

三条贯穿全局的约定：

1. **环境只有一份实现**：评测 rollout、verl 训练、冷启动轨迹生成都调用同一个 `env` 包。工具的返回格式、预算和奖励不会在训练和评测之间出现偏差。
2. **rollout 和打分分开**：所有系统（包括非 LLM 的 B0、B3）都只负责产出 `specs/eval.md` §3 格式的轨迹文件，指标只由一个打分器计算。
3. **奖励和指标共用实现**：训练奖励里的 NDCG@10 直接调用 `eval/metrics.py`，不另写一份。

## 2. 目录结构

```
src/searchtune/
  cli.py                # 统一入口：python -m searchtune <子命令>
  config.py             # YAML → dataclass，未知字段直接报错
  runs.py               # run_id 生成、run 目录、git 状态检查
  data/
    prepare.py          # raw → catalog、pools、splits、MANIFEST、STATS
    pools.py            # 商品池构造（specs/data.md §5）
    slices.py           # 难例切片规则（specs/eval.md §2.3）
  env/
    pool_index.py       # 单个商品池上的 BM25 索引，结果确定
    tools.py            # search / view / submit 的实现与返回格式（training §2）
    episode.py          # 一个 episode 的状态：预算、已展示商品、状态码
    reward.py           # L1 / L2 奖励（training §4），调用 eval/metrics.py
  agent/
    rollout.py          # 评测用 rollout：模型 + env → 轨迹文件
    verl_tools.py       # 把 env 的工具注册成 verl 的工具
    verl_agent_loop.py  # L1 的单轮 AgentLoop；L2 复用 verl 的 ToolAgentLoop
  baselines/
    bm25_raw.py         # B0
    rerank.py           # B3：BM25 前 30 + DeBERTa 重排
  eval/
    metrics.py          # NDCG@10、E-Recall@10、BadRate@3 等
    bootstrap.py        # 按 query 配对 bootstrap
    score.py            # 轨迹文件 + 商品池 → metrics.json + EXPERIMENTS 行
  train/
    gen_traj.py         # B4 采样 + 拒绝采样 → SFT 数据
    cross_encoder.py    # B3 的 DeBERTa 训练
configs/                # 每个实验一个 YAML（含 verl 配置），随代码提交
prompts/                # agent_v<N>.txt、rewrite_v<N>.txt
data/                   # 见 specs/data.md §8
runs/                   # 运行产物，gitignore
tests/
```

## 3. 关键接口

```python
Label = Literal["E", "S", "C", "I", "U"]
Status = Literal["ok", "format_error", "budget_exhausted", "truncated"]

class Episode:
    def __init__(self, pool: Pool, catalog: Catalog, env_cfg: EnvConfig): ...
    def search(self, query: str) -> str          # 返回给 Agent 的文本
    def view(self, product_id: str) -> str
    def submit(self, product_ids: list[str]) -> None
    @property
    def result(self) -> EpisodeResult            # submitted、status、调用计数

def reward_l1(result: EpisodeResult, pool: Pool) -> float
def reward_l2(result: EpisodeResult, pool: Pool, cfg: RewardConfig) -> float
def ndcg_at_10(submitted: list[str], pool: Pool) -> float
def score(traj_path: Path, pools_path: Path) -> Metrics
def paired_bootstrap(traj_a: Path, traj_b: Path, pools_path: Path,
                     metric: str, n: int, seed: int) -> CI
```

- `Pool` 持有标签和增益，但 `Episode` 暴露给 Agent 的方法只返回商品文本。
- `metrics.json`：`specs/eval.md` §2 的全部指标，再加 `run_id`、`split`、`data_version`、`env_version`、`prompt_version`、`git_commit`、`hardware`。

## 4. 与 verl 的集成

- L2：用 verl 的 ToolAgentLoop（`rollout.mode = async`，`multi_turn.enable = true`，`format = hermes`），工具由 `verl_tools.py` 注册。每个 rollout 开始时按 query_id 创建一个 `Episode`，episode 结束后由 `reward_l2` 计算奖励。
- L1：实现一个单轮 AgentLoop：生成改写 → 调 `Episode.search` → 自动提交 → `reward_l1`。
- 工具返回的 token 在训练时不计 loss（verl 的 response mask）。
- 评测不经过 verl：`agent/rollout.py` 用 vLLM 直接推理并调用同一个 `Episode`，保证 greedy 评测可以复现。两条路径的工具文本有一致性测试（§8）。

## 5. 配置与运行

- 每次运行由一个 YAML 配置驱动，命令行只接收配置路径和少量覆盖项。
- `run_id = <实验ID>-<data_version>-<短 commit hash>`；工作区有未提交改动时，在 run_id 后加 `-dirty`。
- 运行开始时把最终生效的配置写入 `runs/<run_id>/config.yaml`。
- 子命令：`prepare`、`rollout`、`score`、`compare`、`gen-traj`、`train-ce`、`train`（转调 verl，L1 / L2 / SFT 由配置区分）。

## 6. 护栏

把 spec 里“不许做”的规则落成代码检查，不靠自觉：

| 规则来源 | 检查 |
|---|---|
| eval §1 测试集只读 | `rollout --split test` 必须显式加 `--allow-test`；工作区有未提交改动时拒绝运行 |
| eval §9 不泄漏标签 | 单元测试：工具返回的文本中不出现 `label`、`gain`、`source` 字段的值 |
| data §6 防泄漏 | `prepare` 结束时断言各 split 的 query 两两不相交，规范化后的 query 不重叠 |
| data §5 商品池规则 | `prepare` 断言干扰商品与 L(q) 不相交，并满足 Jaccard 排除规则 |
| training §6 数据隔离 | `prepare` 断言 `sft_traj` 与 `rl_l2` 不相交，三份训练列表都与 dev、test 不相交 |
| data §7 版本化 | 训练、rollout、打分读取数据时校验 sha256 与 MANIFEST 一致 |
| eval §3 环境版本 | 轨迹文件的环境版本与当前代码不一致时，`score` 拒绝打分 |

## 7. 环境

- Python 3.11，用 `uv` 管理依赖，`uv.lock` 进 git。
- 主要依赖：`verl`、`vllm`（或 `sglang`）、`torch`、`transformers`、`peft`、`rank-bm25`（或同类实现）、`pyarrow`、`pyyaml`、`pytest`。verl、vLLM 和 PyTorch 的版本一起锁定，以 verl 官方推荐的组合为准。

### 7.1 运行平台（DEC-009）

| 平台 | 硬件 | 负责 |
|---|---|---|
| 本地 WSL2 Ubuntu | RTX 5060 Laptop 8GB，24GB 内存 | 写代码、数据管线、环境、打分器、单元测试；用 Qwen3.5-0.8B 在 `dev-quick` 上冒烟 |
| 云端 AutoDL | RTX 4090 24GB；L2 不够时换 48–80GB 单卡 | 所有正式结果：baseline、轨迹生成、全部训练与评测 |

**判断标准**：结果会进入比较、checkpoint 选择或验收判定的，一律在云端跑。本地跑出的数字只用来确认代码能跑通，不写进 `EXPERIMENTS.md` 的结果栏。

### 7.2 本地 WSL2 的注意事项

- RTX 5060 是 Blackwell 架构，PyTorch 必须安装 CUDA 12.8 或更高版本的构建。
- WSL 内不装 Linux 版 NVIDIA 驱动，直接使用 Windows 驱动。
- 在 `%UserProfile%\.wslconfig` 中把 WSL 可用内存调到 16GB 左右（默认只有宿主机内存的一半）。
- 仓库、数据和 checkpoint 放在 WSL 自己的文件系统里（`~/`），不放在 `/mnt/c` 或 `/mnt/d`，后者 I/O 慢很多。

### 7.3 云端约定

- 环境由 `scripts/setup_cloud.sh` 一次装好；镜像名称、CUDA 版本和驱动版本记入 `DECISIONS.md`。先用“无卡模式”开机装环境，装好后再用 GPU 开机。
- 代码通过 git 同步；模型从 ModelScope 下载；数据和 checkpoint 放在数据盘 `/root/autodl-tmp`。
- 训练在 `tmux` 中运行，SSH 断开不会中断训练。
- 回传：`metrics/*.json` 和 `EXPERIMENTS.md` 的记录行在云端直接提交推送；LoRA adapter 用 `scp` 拉回本地；完整 checkpoint 留在云端，不需要的及时删除。
- 跑完立即关机。按小时计费，实例开着不跑也在扣钱。

## 8. 测试

- 单元测试覆盖：BM25 结果的确定性、三个工具的返回格式和截断、预算与状态码、奖励函数的各种情况、全部指标、bootstrap 的确定性、各项护栏检查。
- 一致性测试：同一个 episode，经由 verl 工具注册和经由 `agent/rollout.py` 调用，工具返回的文本逐字一致；奖励函数与打分器的 NDCG 结果一致。
- 冒烟测试：用 Qwen3.5-0.8B 在 `dev-quick` 的 10 个 query 上跑通 `rollout → score`，以及 L1 的 2 个训练步。
