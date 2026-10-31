# 设计

> 本文只定代码结构和模块之间的约定。数据、评测、训练的“做什么”分别以 `specs/data.md`、`specs/eval.md`、`specs/training.md` 为准；本文与它们冲突时改本文。

## 1. 数据流

```
data/raw ──prepare──▶ data/processed/<ver>/*.jsonl ─┬─train-sft / train-grpo / train-ce──▶ runs/<run_id>/ckpt
                    └▶ data/splits/*.txt            │
                                                    └─infer──▶ runs/<run_id>/pred/<split>.jsonl
                                                                         │
                                                                     score / compare
                                                                         ▼
                                                     runs/<run_id>/metrics/<split>.json  →  EXPERIMENTS.md
```

两条贯穿全局的约定：

1. **推理和打分分开**：所有模型（B0、B1、B3、SFT、GRPO）都只负责产出 `specs/eval.md` §3 格式的预测文件，指标只由一个打分器计算。改指标不重跑推理，各模型之间也不会出现口径不一致。
2. **训练和推理共用同一套输入构造与输出解析**：prompt 渲染、截断和档位解析各只有一份实现，SFT、GRPO 奖励和评测推理都调用它，避免训练和评测的口径偏差。

## 2. 目录结构

```
src/searchtune/
  cli.py              # 统一入口：python -m searchtune <子命令>
  config.py           # YAML → dataclass，未知字段直接报错
  runs.py             # run_id 生成、run 目录、git 状态检查
  data/
    schema.py         # Sample 数据结构与校验
    prepare.py        # raw → processed，清洗、切分、MANIFEST、STATS
  text/
    render.py         # Sample → chat messages（含截断，见 training §3）
    parse.py          # 模型输出 → 档位或 INVALID
  infer/
    llm.py            # HF transformers + PEFT，直答模式取答案位置 logits
    llm_vllm.py       # 仅推理模式（E8）使用，可选
    baselines.py      # B0 多数类
  eval/
    metrics.py        # P/R/F1、混淆矩阵、PR-AUC
    ndcg.py           # specs/eval.md §2.3
    bootstrap.py      # 按 query 配对 bootstrap
    score.py          # 预测文件 + gold → metrics.json + EXPERIMENTS 行
  train/
    sft.py            # TRL SFTTrainer + LoRA
    grpo.py           # TRL GRPOTrainer
    reward.py         # 代价矩阵奖励，复用 text/parse.py
    cross_encoder.py  # B3
configs/              # 每个实验一个 YAML，随代码提交
prompts/              # relevance_v<N>.txt、sft_input_v<N>.txt
data/                 # 见 specs/data.md §7
runs/                 # 运行产物，gitignore
tests/
```

## 3. 关键接口

```python
Label = Literal["E", "S", "C", "I"]

def render(sample: Sample, template: str, max_product_tokens: int, tokenizer) -> list[dict]
def parse_label(text: str) -> Label | Literal["INVALID"]
def reward(gold: Label, completion: str, matrix: RewardMatrix) -> float   # 内部调用 parse_label
def score(pred_path: Path, gold_path: Path) -> Metrics
def paired_bootstrap(pred_a: Path, pred_b: Path, gold_path: Path,
                     metric: str, n: int, seed: int) -> CI
```

数据结构：

- `Sample`：字段与 `specs/data.md` §3 一致
- 预测行：字段与 `specs/eval.md` §3 一致
- `metrics.json`：`specs/eval.md` §2 的全部指标，再加 `run_id`、`split`、`data_version`、`prompt_version`、`git_commit`、`hardware`（GPU 型号）

## 4. 直答模式的推理实现

- 用 HF `generate`（greedy，`max_new_tokens = 8`，`output_scores=True`），一次拿到生成文本和每一步的 logits。
- `pred` 来自对生成文本的 `parse_label`；`probs` 取第一个档位 token 出现位置的 logits，只在 E/S/C/I 四个 token 上做 softmax。
- 不用“只在四个 token 里取 argmax”的约束解码，因为那会把格式错误藏起来，而 `specs/eval.md` 要求如实报告解析失败率。

## 5. 配置与运行

- 每次运行由一个 YAML 配置驱动，命令行只接收配置路径和少量覆盖项。
- `run_id = <实验ID>-<data_version>-<短 commit hash>`；工作区有未提交改动时，在 run_id 后加 `-dirty`。
- 运行开始时把最终生效的配置写入 `runs/<run_id>/config.yaml`。
- 子命令：`prepare`、`infer`、`score`、`compare`、`train-sft`、`train-grpo`、`train-ce`。

## 6. 护栏

把 spec 里“不许做”的规则落成代码检查，不靠自觉：

| 规则来源 | 检查 |
|---|---|
| eval §1 测试集只读 | `infer --split test` 必须显式加 `--allow-test`；工作区有未提交改动时拒绝运行 |
| data §5 防泄漏 | `prepare` 结束时断言各 split 的 query 两两不相交，规范化后的 query 不重叠 |
| training §9 数据隔离 | `prepare` 断言 `sft_main` 与 `grpo_main` 的 query 不相交 |
| data §6 版本化 | 训练和推理读取数据时校验文件 sha256 与 MANIFEST 一致 |
| eval §3 解析失败计入分母 | `score` 遇到 `INVALID` 按错误计算，有单元测试 |

## 7. 环境

- Python 3.11，用 `uv` 管理依赖，`uv.lock` 进 git。
- 主要依赖：`torch`、`transformers`、`peft`、`trl`、`datasets`、`pyarrow`、`pyyaml`、`pytest`；`scikit-learn` 只在测试里用作参照；`vllm` 是可选依赖，只有 E8 需要。
### 7.1 运行平台（DEC-009）

| 平台 | 硬件 | 负责 |
|---|---|---|
| 本地 WSL2 Ubuntu | RTX 5060 Laptop 8GB，24GB 内存 | 写代码、数据管线、打分器、单元测试、`dev-quick` 冒烟 |
| 云端 AutoDL | RTX 4090 24GB | 所有正式结果：B1、B3 的正式结果，E1–E8 的训练与评测 |

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

- 单元测试覆盖：`parse_label`、`render` 的截断、`reward` 的 16 个格子加解析失败、全部指标、bootstrap 的确定性、各项护栏检查。
- 冒烟测试：用一个极小模型在 `dev-quick` 的 10 条样本上跑通 `infer → score`，在 CI 或本地均可运行，不需要 GPU。
