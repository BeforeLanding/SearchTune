# SearchTune

**基于 ESCI 可验证奖励的电商商品搜索 Agent 强化学习后训练**

> 状态：**规划完成，M0（环境与评测）开发中**。目前还没有任何实验结果，本页不包含效果数字。范围已按 2 周 MVP 收敛，见 [DEC-015](DECISIONS.md)。

## 这个项目做什么

训练一个小参数量的 LLM 搜索 Agent：给定一个真实的电商搜索 query，Agent 在商品库中多轮调用工具（检索、查看商品详情），最后提交一个排好序的商品列表。

训练信号来自 [Amazon Shopping Queries Dataset（ESCI）](https://github.com/amazon-science/esci-data) 的**人工分档相关性标签**：每个被提交商品的增益都是已知的，所以奖励可以直接计算、可以验证，**不依赖奖励模型或 LLM 裁判**。

## 核心问题

**受限预算与低延迟约束下的小模型（≤4B）Agent 多轮工具能力内化闭环：拒绝采样冷启动 SFT + 可验证奖励多轮 GRPO。**

小模型做复杂搜索约束时有三类典型失败：**改写偏离**（改写丢掉硬约束）、**未核验盲交**（没 `view` 就提交，前 3 名出现不相关商品，即 BadRate@3）、**长轨迹的格式与模板坍塌**。本项目用可验证的奖励把这三类行为训进 2B 模型，并量化三者之间的边界：2B 后训练 Agent、9B 零样本 Agent（B4），以及同查看预算的传统 `BM25 + DeBERTa 精排`（B3），在**效果、延迟、成本**上各自的位置。

## 为什么这么设计

- **可验证的环境**：每个 query 对应一个封闭的商品池，由全部已标注商品、BM25 检索出的难干扰商品和随机干扰商品组成。工具全部是确定性的，评测可以复现。
- **两级训练**：
  - **L1** 单步改写 query + GRPO，是最小可行版本，先把环境和训练管线跑通。
  - **L2** 多轮工具调用 Agent：用更强的开源模型生成轨迹，经拒绝采样做冷启动 SFT，再做多轮 GRPO。
- **贴近线上的奖励**：NDCG@10，加上前 3 名出现不相关商品的坏例惩罚，以及超额工具调用的成本。
- **和传统做法比**：与“BM25 + 精排模型”流水线在相同查看预算下对比，回答“为什么要用 Agent”。这个对比如实报告，也可能显示传统流水线更强。

## 评测纪律

- 测试集冻结，只有每个里程碑最终选定的模型才在上面跑一次，总运行次数公开报告。
- 验收阈值在 baseline 跑完之后、第一次训练之前锁定，并单独提交。
- 显著性用按 query 配对的 bootstrap 判断。
- 除总体指标外，还按难例切片（否定、数量规格、兼容型号、指定品牌、长 query）分别报告，并对失败轨迹做分类分析。

## 进度

范围分两层（DEC-015）：**P0 是 2 周 MVP 必交主线**，**P2 是延后可选、不阻塞验收**。P2 没做就如实标“未做”。

| 里程碑 | 内容 | 优先级 | 状态 |
|---|---|---|---|
| M0 | 数据管线、商品池、`env_v1`、打分器、B0–B4、评测锁定 | P0 | 进行中（本地 CPU 部分先行，不依赖云端 GPU） |
| M1 | E1：L1 单步改写 + GRPO | P0 | 未开始 |
| M2 核心 | E3 冷启动 SFT、E4 多轮 GRPO、E5 奖励消融、E6 无冷启动消融 | P0 | 未开始 |
| M3 核心 | 难例切片与轨迹误差归因；推理效率基准（P50/P95 延迟、生成 token、显存） | P0 | 未开始 |
| 扩展 | E2 动态过滤、E7 GiGPO、E8 环境变体稳健性、E9 4B 对照、漏标抽检 | P2 | 未做 |

## 技术栈（暂定）

- 策略模型：Qwen3.5-2B（LoRA）；冷启动轨迹由 Qwen3.5-9B 生成
- 训练框架：[verl](https://github.com/verl-project/verl)（多轮工具调用的 AgentLoop）
- 传统流水线基线：BM25 + DeBERTa-v3-base 精排
- 数据与环境的核心代码是跨平台纯 Python，本地 CPU 即可开发与测试；正式训练与评测在云端单卡 GPU
- 若 Qwen3.5-2B 在 T0 出现框架兼容阻力，半天内切换到 Qwen3 / Qwen2.5-Instruct 纯文本底座（DEC-015）

模型和框架的最终选择以 T0 环境检查的结论为准，见 [DECISIONS.md](DECISIONS.md)。

## 文档

本项目采用 Spec 驱动开发：先写 spec 并锁定评测，再实现。

| 文档 | 内容 |
|---|---|
| [CONSTITUTION.md](CONSTITUTION.md) | 目标、范围、约束、原则、里程碑 |
| [specs/data.md](specs/data.md) | 数据来源、商品池构造、切分与版本 |
| [specs/eval.md](specs/eval.md) | 指标、baseline、验收阈值、锁定流程、误差分析 |
| [specs/training.md](specs/training.md) | 环境与工具、奖励、训练配置、实验矩阵 |
| [specs/labeling-guideline.md](specs/labeling-guideline.md) | 相关性档位定义（硬约束与软约束） |
| [DESIGN.md](DESIGN.md) | 代码结构、接口、护栏 |
| [AGENTS.md](AGENTS.md) | 仓库不变量，开发会话自动加载 |
| [TASKS.md](TASKS.md) | 任务拆分与验收清单 |
| [DECISIONS.md](DECISIONS.md) | 关键决策及其取舍 |

## 主要参考

- Reddy et al. [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://arxiv.org/abs/2206.06588), 2022
- Jin et al. [Search-R1](https://github.com/PeterGriffinJin/Search-R1)：用 RL 训练调用搜索引擎的推理模型
- Tongyi DeepResearch Team. [Tongyi DeepResearch Technical Report](https://arxiv.org/abs/2510.24701)
- [ShoppingBench: A Real-World Intent-Grounded Shopping Benchmark for LLM-based Agents](https://arxiv.org/abs/2508.04266)
- [Shopping Companion: Benchmarking and Training LLM Agents for Long-Horizon Preference-Grounded E-Commerce Tasks](https://arxiv.org/abs/2603.14864)
- [RAGEN-2: Reasoning Collapse in Agentic RL](https://arxiv.org/abs/2604.06268)
- [TaoSR1: The Thinking Model for E-commerce Relevance Search](https://arxiv.org/abs/2508.12365)
