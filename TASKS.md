# 任务

> 每个任务都可以单独交给执行者。执行者只读任务里引用的 spec 章节，按“验收”逐条自查后再交付。
>
> 通用规则：
> - spec 没写清楚或与现实不符时，**停下来提问**，不要自行决定。确认后先改 spec，再改代码。
> - 不修改 `specs/eval.md` 中已锁定的内容，也不修改已冻结的环境版本。
> - 每次训练或评测都在 `EXPERIMENTS.md` 记一行。
> - 能验证的结果必须用真实命令和自动化测试验证；本地无法验证的项（需要云端 GPU）明确列出，不伪造数字，不用 mock 冒充真实数据管线的结论。
>
> 状态：`待办` / `进行中` / `完成` / `砍掉`
> 优先级：`P0` 为 2 周 MVP 必交主线；`P2` 为延后可选，不阻塞 MVP 验收（DEC-015）。

## 依赖总览（DEC-015：本地 CPU 研发与云端 GPU 解耦）

```
本地纯 CPU 链：  T1 ─┬─ T2 ─ T3 ──┐
                    └─ T4 ───────┴─ T5 ─ T5b(B0)
云端 GPU 链：    T0 ──▶ T6（LLM rollout，B1/B2/B4）、T9/T11（训练）
                T5 + T0 ─▶ T6          T7（B3）依赖 T3、T4，需要 torch extras
```

T1–T5、T5b 全部在本地纯 CPU 环境独立开发与验收，**不被 T0 阻塞**。T0 只作为 T6 与 T9/T11 的前置。

## M0 环境与评测（P0）

### T0 运行环境与模型检查 · 待办 · P0（云端，仅阻塞 T6、T9、T11）
- 依赖：无（不阻塞 T1–T5b）
- 依据：`specs/training.md` §3.1，`DESIGN.md` §7，`DECISIONS.md` DEC-009、DEC-012、DEC-015
- 产出：`scripts/setup_cloud.sh`；云端 extras 的 `uv.lock`；模型检查脚本及其输出；`DECISIONS.md` 中的检查结论
- 验收：
  - [ ] 本地 WSL2（可选）：按 DESIGN §7.2 配好，`torch.cuda.is_available()` 为真，能识别 RTX 5060
  - [ ] 云端：`setup_cloud.sh` 能在新实例上一次装好环境（含 verl 和 vLLM，通过 extras 安装）；镜像名称、CUDA 和驱动版本已记入 `DECISIONS.md`
  - [ ] training §3.1 的 5 项在云端逐项有结论；**出现任何框架兼容阻力就在半天内切换纯文本备选底座**，并记入 `DECISIONS.md`
  - [ ] 在 4090 上用 verl 跑一个玩具多轮工具任务的若干训练步，记录显存峰值和每步耗时，据此决定 L2 是否需要换更大的卡

### T1 项目骨架 · 完成 · P0（本地）
- 依赖：无
- 依据：`DESIGN.md` §2、§5、§7
- 产出：`pyproject.toml`（分层依赖）、`src/searchtune/` 的目录结构、`config.py`、`runs.py`、`cli.py`、`.gitignore`、pytest 配置
- 验收：
  - [x] `python -m searchtune --help` 列出全部 7 个子命令
  - [x] 配置文件里有未知字段时报错（有单元测试）
  - [x] 工作区有未提交改动时，run_id 带 `-dirty`（有单元测试）
  - [x] `data/raw/`、`data/processed/`、`runs/` 已加入 gitignore
  - [x] 仅基础依赖时 `uv sync` 能装好并跑通全部核心单元测试（`torch/transformers/peft`、`vllm/verl` 在 optional extras 中）

### T2 数据下载与核对 · 完成 · P0（本地）
- 依赖：T1
- 依据：`specs/data.md` §1
- 产出：数据下载脚本；`data/MANIFEST.json` 中原始文件的 sha256 和许可证
- 验收：
  - [x] 实际字段与 data §1 逐项核对；不一致的地方**先改 spec**，并在交付说明中列出
  - [x] 许可证已核对并记录

### T3 数据管线与商品池 · 完成 · P0（本地）
- 依赖：T2
- 依据：`specs/data.md` §3–§9，`specs/eval.md` §2.3，`specs/training.md` §6，`DESIGN.md` §6
- 产出：`prepare` 子命令；`catalog.jsonl`；三个评测 split 和三份训练列表；每个 split 的商品池；难例切片成员列表；`data/STATS.md`
- 验收：
  - [x] data §9 的 5 项全部满足
  - [x] training §10 中关于列表不相交的检查已实现
  - [x] STATS 中写明难例切片的规则和各切片的 query 数

### T4 打分器 · 完成 · P0（本地）
- 依赖：T1（可与 T2、T3 并行）
- 依据：`specs/eval.md` §2、§3、§6、§9，`DESIGN.md` §3
- 产出：`eval/` 模块；`score` 与 `compare` 子命令
- 验收：
  - [x] eval §9 中与打分器相关的 4 项满足（指标手算一致、bootstrap 确定性、非 ok 计入分母、一条命令出全部指标）
  - [x] 环境版本与当前代码不一致的轨迹文件被拒绝打分（有测试）

### T5 环境与工具 · 完成 · P0（本地）
- 依赖：T3、T4
- 依据：`specs/training.md` §2、§4，`DESIGN.md` §3、§6、§8
- 产出：`env/` 包；`env_v1`
- 验收：
  - [x] 三个工具的返回格式、截断、预算和状态码有单元测试
  - [x] 标签不泄漏测试通过（eval §9）
  - [x] 奖励函数满足 training §10 中的两项测试
  - [x] BM25 结果确定：同一输入两次调用逐字一致

### T5b B0 原始 query BM25 基线 · 完成 · P0（本地）
- 依赖：T3、T4、T5
- 依据：`specs/eval.md` §3、§5，`DESIGN.md` §6
- 产出：`baselines/bm25_raw.py`；B0 在 dev-quick、dev（以及锁定时的 test）上的轨迹文件与指标
- 验收：
  - [x] `--split test` 不加 `--allow-test` 时拒绝运行；工作区有未提交改动时也拒绝（有测试）
  - [x] 轨迹文件符合 eval §3，能被 `score` 直接打分
  - [x] 结果写入 `EXPERIMENTS.md`（dev-quick、dev 已跑；test 上的 B0 按 eval §7 在锁定时一并跑，尚未跑）

### T6 rollout、B1、B2、B4 与 prompt 迭代 · 待办 · P0（云端）
- 依赖：T0、T5
- 依据：`specs/eval.md` §3、§4、§5，`DESIGN.md` §4
- 产出：`agent/rollout.py`；`prompts/agent_v1.txt`、`prompts/rewrite_v1.txt`；B1、B2、B4 在 dev 和 test 上的结果
- 验收：
  - [ ] greedy 评测两次的轨迹逐字一致（eval §9）
  - [ ] prompt 迭代只用了 dev，每个版本的结果都记入 `EXPERIMENTS.md`
  - [ ] 结果已填入 eval §7 的表格

### T7 B3 检索 + 精排流水线 · 待办 · P0（需 `rerank` extras；CPU 可冒烟，正式结果在云端）
- 依赖：T3、T4
- 依据：`specs/eval.md` §5，`specs/training.md` §3
- 产出：`train-ce` 子命令；`baselines/rerank.py`；B3 在 dev 和 test 上的结果
- 验收：
  - [ ] DeBERTa 只用 train split 的标注训练，没有用到 dev、test 的 query
  - [ ] 结果已填入 eval §7 的表格

### T8 锁定评测 · 待办 · P0
- 依赖：T5b、T6、T7
- 依据：`specs/eval.md` §7
- 验收：
  - [ ] Δ₁ 已填写，eval 顶部的锁定状态改为“已锁定”，并以 `lock eval v1` 单独提交
  - [ ] commit hash 已回填到 eval 顶部，并在 `DECISIONS.md` 中记录

## M1 L1 单步改写（P0）

### T9 L1 训练管线与 E1 · 待办 · P0
- 依赖：T0、T8
- 依据：`specs/training.md` §2.3、§4.1、§5.1、§9，`DESIGN.md` §4
- 产出：L1 的 AgentLoop；`configs/e1.yaml`；E1 的结果
- 验收：
  - [ ] 本地用 Qwen3.5-0.8B 跑通 2 个训练步（冒烟）
  - [ ] 用相同配置跑两次，dev 主指标相差 < 1 个点
  - [ ] training §8 的 L1 实测耗时已回填

### T10 选定 M1 · 待办 · P0
- 依赖：T9
- 内容：M1 候选只有 E1（E2 为 P2，有余力再做）
- 验收：
  - [ ] 在 test 上只跑一次 E1，写出验收判定并附 bootstrap 置信区间
  - [ ] 在 dev 上完成轨迹误差分析（eval §8.1）

## M2 L2 多轮 Agent · 核心（P0）

### T11 冷启动轨迹与 SFT（E3） · 待办 · P0
- 依赖：T0、T8（可与 T9、T10 并行）
- 依据：`specs/training.md` §5.2
- 产出：`gen-traj` 子命令；SFT 数据集；E3 的结果
- 验收：
  - [ ] 轨迹保留率已记入 `EXPERIMENTS.md`
  - [ ] SFT 的 loss 只作用于 assistant token（有测试）

### T12 L2 GRPO（E4） · 待办 · P0
- 依赖：T11
- 依据：`specs/training.md` §4.2、§5.3、§5.4，`DESIGN.md` §4
- 产出：L2 的 verl 配置与工具注册；E4 的结果；训练监控曲线
- 验收：
  - [ ] verl 工具与 rollout 工具的一致性测试通过（DESIGN §8）
  - [ ] 训练监控包含 training §5.4 的全部指标
  - [ ] training §8 的 L2 实测耗时已回填

### T13 L2 对照实验并选定 M2 · 待办 · P0
- 依赖：T12
- 内容：**E5、E6 必做**；E7、E9 为 P2；E8 为 P2，只评测
- 验收：
  - [ ] 按 dev 在 E4–E6 中选出 M2，在 test 上只跑一次，写出验收判定（未达标也要写清楚原因分析）
  - [ ] 在 dev 上完成轨迹误差分析，并与 M1 对比

## M3 收尾 · 核心（P0）

### T14 难例切片、轨迹误差归因与推理效率基准 · 待办 · P0
- 依赖：T13
- 依据：`specs/eval.md` §2.3、§8.1、§10
- 验收：
  - [ ] 在 test 上完成 M1、M2 的轨迹误差分布对比，以及难例切片对比
  - [ ] 完成 eval §10 的推理效率基准：2B 后训练模型、9B 零样本（B4）、B3 的单 episode P50/P95 延迟、生成 token 消耗、显存占用，硬件与批大小写明

### T15 README 与复现 · 待办 · P0
- 依赖：T14
- 验收：
  - [ ] README 包含主结果表（含与 B3、B4 的对比和效率基准）、复现命令、test 总运行次数；P2 项没做的如实标注“未做”
  - [ ] 按 README 从空环境复现 B0 和 E1 的数据、环境与评测部分
  - [ ] `CONSTITUTION.md` 文档地图中所有文档的状态都是最新的

## 延后可选扩展（P2，不阻塞 MVP 验收）

### T16 漏标抽检与敏感性分析 · 待办 · P2
- 依赖：T14
- 依据：`specs/eval.md` §8.2
- 验收：
  - [ ] 完成 eval §8.2 的漏标抽检和敏感性分析

### T17 P2 实验：E2、E7、E8、E9 · 待办 · P2
- 依赖：E2 依赖 T9；E7、E8、E9 依赖 T12
- 依据：`specs/training.md` §7
- 验收：每个实验独立验收，没做就在 README 标“未做”
