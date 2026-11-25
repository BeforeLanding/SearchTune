# AGENTS.md — SearchTune 仓库不变量

> 开发会话自动加载。上位文档：`CONSTITUTION.md`；改目标、范围、任务依赖或训练配置，先在 `DECISIONS.md` 记一条，同一次提交里同步 spec / 设计文档。**代码与文档不许脱节。**

## 工作方式

- Spec 驱动：开工前读 `CONSTITUTION.md`、`DECISIONS.md`、`DESIGN.md`、`TASKS.md` 与相关 `specs/*.md`。spec 与现实不符时先停下确认、先改 spec 再改代码。
- 能验证的结果必须用真实命令和 `pytest` 验证。本地无法验证的项（需要云端大显存 GPU）明确列出；**不伪造实验数字，不用 mock 冒充真实数据管线的结论**。
- 优先级：P0 是 2 周 MVP 主线，P2（E2/E7/E8/E9、漏标抽检）不阻塞验收（DEC-015）。

## 不变量（违反即 bug）

1. **单一 `env` 实现**：评测 rollout、verl 训练、冷启动轨迹生成都调用 `src/searchtune/env/` 同一份。工具文本、预算、奖励不得在别处复制。
2. **rollout 与打分分离**：所有系统（含非 LLM 的 B0、B3）只产出轨迹文件（`specs/eval.md` §3）；指标只由 `eval/` 的打分器计算。
3. **奖励与评测共用 `ndcg_at_10`**：`env/reward.py` 直接 import `eval/metrics.py`，不另写 NDCG。
4. **Agent 看不到标签**：工具返回里不得出现 `label`、`gain`、`source` 的值（有测试守护）。
5. **冻结测试集**：`--split test` 必须显式 `--allow-test`，且 git 工作区干净，否则拒绝运行。test 不参与调参、选 checkpoint、改 prompt。每次 test 运行都记入 `EXPERIMENTS.md` 并计数。
6. **split 零泄漏断言**：`prepare` 必须断言 train/dev/test 的 `query_id` 两两不相交、规范化 query 跨 train 与 dev/test 零重叠、`sft_traj` 与 `rl_l2` 不相交、三份训练列表与 dev/test 不相交；干扰商品与 `L(q)` 不相交且满足 Jaccard 排除规则。
7. **数据与环境版本**：读数据时校验 `data/MANIFEST.json` 的 sha256；轨迹文件的环境版本与代码不一致时 `score` 拒绝打分。已冻结的 `env_v1` 与锁定后的 eval 内容不得改动。
8. **确定性**：BM25 同分按 `product_id` 排序；所有抽样固定 `seed = 20261031`；bootstrap 固定 seed。
9. **非 `ok` 状态按 0 分计入分母**，不得从分母剔除。
10. **每次运行写 `EXPERIMENTS.md`**，包括失败的；`run_id = <实验ID>-<data_version>-<短 commit>`，工作区有改动时加 `-dirty`。

## 平台职责（DEC-009、DEC-015）

- **本地（Windows 或 WSL2，纯 CPU 即可）**：T1–T5、T5b（B0）的开发与验收、单元测试、数据管线。不依赖 T0。
- **云端 AutoDL（RTX 4090 24GB，必要时 48–80GB）**：T0、LLM rollout（B1/B2/B4）、训练、所有会进入比较或验收的正式结果。本地 GPU 数字只用于确认代码能跑，不写进 `EXPERIMENTS.md` 的结果栏。
- 核心包只依赖基础依赖；`torch/transformers/peft`（extra `rerank`）与 `vllm/verl`（extra `gpu`，仅 Linux）不得在核心模块顶层 import。
- T0 止损红线：Qwen3.5-2B 框架兼容出现任何阻力，半天内切换 Qwen3 / Qwen2.5-Instruct 纯文本底座。

## 仓库约定

- 原始数据、`data/processed/`、`runs/`、checkpoint 不进 git；`data/splits/<version>/`、`data/slices/<version>/`、`data/MANIFEST.json`、`data/STATS*.md` 进 git。
- 命令：`uv sync` 后 `uv run pytest`；入口 `python -m searchtune --help`。
- 文本文件统一 LF 换行、UTF-8。
