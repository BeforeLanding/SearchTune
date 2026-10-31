# 任务

> 每个任务都可以单独交给执行者。执行者只读任务里引用的 spec 章节，按“验收”逐条自查后再交付。
>
> 通用规则：
> - spec 没写清楚或与现实不符时，**停下来提问**，不要自行决定。确认后先改 spec，再改代码。
> - 不修改 `specs/eval.md` 中已锁定的内容。
> - 每次训练或评测都在 `EXPERIMENTS.md` 记一行。
>
> 状态：`待办` / `进行中` / `完成` / `砍掉`

## M0 数据与评测

### T0 环境与模型检查 · 待办
- 依赖：无
- 依据：`specs/training.md` §2.1，`DESIGN.md` §7
- 产出：`pyproject.toml`、`uv.lock`；模型检查脚本及其输出；`DECISIONS.md` 中的检查结论
- 验收：
  - [ ] training §2.1 的 5 项逐项有结论；不通过就切换备选模型，并记入 `DECISIONS.md`
  - [ ] 确定训练和推理的运行平台（WSL2 或云端），记入 `DECISIONS.md`

### T1 项目骨架 · 待办
- 依赖：T0
- 依据：`DESIGN.md` §2、§5
- 产出：`src/searchtune/` 的目录结构、`config.py`、`runs.py`、`cli.py`、`.gitignore`、pytest 配置
- 验收：
  - [ ] `python -m searchtune --help` 列出全部 7 个子命令
  - [ ] 配置文件里有未知字段时报错（有单元测试）
  - [ ] 工作区有未提交改动时，run_id 带 `-dirty`（有单元测试）
  - [ ] `data/raw/`、`data/processed/`、`runs/` 已加入 gitignore

### T2 数据下载与核对 · 待办
- 依赖：T1
- 依据：`specs/data.md` §1
- 产出：数据下载脚本；`data/MANIFEST.json` 中原始文件的 sha256 和许可证
- 验收：
  - [ ] 实际字段与 data §1 逐项核对；不一致的地方**先改 spec**，并在交付说明中列出
  - [ ] 许可证已核对并记录

### T3 数据管线 · 待办
- 依赖：T2
- 依据：`specs/data.md` §3–§7，`specs/training.md` §4.1、§5.1，`DESIGN.md` §6
- 产出：`prepare` 子命令；`data/processed/v1/`；`data/splits/` 下 dev、dev-quick、test、sft_main、grpo_main 五份列表；`data/STATS.md`
- 验收：
  - [ ] data §8 的 4 项全部满足
  - [ ] `sft_main` 与 `grpo_main` 的 query 不相交（有断言）
  - [ ] STATS 中包含输入 token 长度分布，并给出超过 512 的比例（training §3）

### T4 打分器 · 待办
- 依赖：T1（可与 T2、T3 并行）
- 依据：`specs/eval.md` §2、§6、§9，`DESIGN.md` §3
- 产出：`eval/` 模块；`score` 与 `compare` 子命令
- 验收：
  - [ ] eval §9 的 5 项全部满足

### T5 prompt 与推理 · 待办
- 依赖：T3、T4
- 依据：`specs/eval.md` §3、§4，`specs/training.md` §3，`DESIGN.md` §4、§6
- 产出：`text/`、`infer/` 模块；`prompts/relevance_v1.txt`；`infer` 子命令
- 验收：
  - [ ] `parse_label` 和 `render` 的截断逻辑有单元测试
  - [ ] B1 在 `dev-quick` 上跑通，预测文件格式符合 eval §3
  - [ ] `--split test` 不加 `--allow-test` 时拒绝运行；工作区有未提交改动时也拒绝（有测试）
  - [ ] DESIGN §8 的冒烟测试能跑通

### T6 B0、B1 与 prompt 迭代 · 待办
- 依赖：T5
- 依据：`specs/eval.md` §4、§5
- 产出：在 `dev` 上迭代后选定的 prompt 版本；B0、B1 在 dev 和 test 上的结果
- 验收：
  - [ ] prompt 迭代过程只用了 dev，每个版本的结果都记入 `EXPERIMENTS.md`
  - [ ] B0、B1 的 dev 和 test 结果已填入 eval §7 的表格

### T7 B3 cross-encoder · 待办
- 依赖：T3、T4
- 依据：`specs/eval.md` §5，`specs/training.md` §2
- 产出：`train-ce` 子命令；B3 在 dev 和 test 上的结果
- 验收：
  - [ ] 训练数据使用 `sft_main.txt`
  - [ ] 结果已填入 eval §7 的表格

### T8 锁定评测 · 待办
- 依赖：T6、T7
- 依据：`specs/eval.md` §7
- 验收：
  - [ ] Δ₁ 已填写，eval 顶部的锁定状态改为“已锁定”，并以 `lock eval v1` 单独提交
  - [ ] commit hash 已回填到 eval 顶部，并在 `DECISIONS.md` 中记录

## M1 SFT

### T9 SFT 与 E1 · 待办
- 依赖：T8
- 依据：`specs/training.md` §3、§4、§8、§9
- 产出：`train-sft` 子命令；`configs/e1.yaml`；E1 的结果
- 验收：
  - [ ] 用相同配置跑两次，dev 主指标相差 < 0.5 个点
  - [ ] training §7 的 SFT 实测耗时已回填

### T10 SFT 对照实验并选定 M1 · 待办
- 依赖：T9
- 依据：`specs/training.md` §6，`specs/eval.md` §6
- 内容：E3 必做；E2、E4、E5 视时间决定
- 验收：
  - [ ] 按 dev 主指标选出 M1，在 test 上只跑一次
  - [ ] 写出 M1 是否满足验收条件的判定，附 bootstrap 置信区间
  - [ ] 在 dev 上完成误差分析（eval §8）

## M2 GRPO

### T11 奖励函数、GRPO 与 E6 · 待办
- 依赖：T10
- 依据：`specs/training.md` §5，`DESIGN.md` §3
- 产出：`train/reward.py`；`train-grpo` 子命令；E6 的结果
- 验收：
  - [ ] 奖励函数的单元测试覆盖 16 个格子和解析失败
  - [ ] training §7 的 GRPO 实测耗时已回填

### T12 GRPO 对照实验并选定 M2 · 待办
- 依赖：T11
- 内容：E7 必做；E8 视时间决定
- 验收：
  - [ ] 按 dev 选出 M2，在 test 上只跑一次，写出验收判定（未达标也要写清楚原因分析）
  - [ ] 在 dev 上完成误差分析，并与 M1 对比

## M3 收尾

### T13 最终误差分析 · 待办
- 依赖：T12
- 依据：`specs/eval.md` §8
- 验收：
  - [ ] 在 test 上完成 M1、M2 的 T1–T9 分布对比；T9 占比单独报告

### T14 README 与复现 · 待办
- 依赖：T13
- 验收：
  - [ ] README 包含主结果表、复现命令、test 总运行次数
  - [ ] 按 README 从空环境复现 E1 的数据和评测部分
  - [ ] `CONSTITUTION.md` 文档地图中所有文档的状态都是最新的
