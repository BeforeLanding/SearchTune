# 评测 Spec

> 上位约束：`CONSTITUTION.md` 原则 1、2。评测集来源见 `specs/data.md` §4，档位定义见 `specs/labeling-guideline.md`。
>
> **锁定状态：未锁定**（锁定流程见 §7；锁定后在此填 commit hash）

## 1. 评测集与使用规则

| 集合 | 规模 | 允许的用途 | 禁止的用途 |
|---|---|---|---|
| `dev-quick` | 200 query | 开发期冒烟、改代码后的回归检查 | — |
| `dev` | 500 query | 调参、选 checkpoint、迭代 prompt、指导迭代的误差分析 | — |
| `test` | 1000 query | 每个里程碑**最终选定**的模型跑一次；baseline 跑一次 | 调参、选 checkpoint、改 prompt、指导迭代的误差分析 |

- 每次在 `test` 上评测都要记入 `EXPERIMENTS.md`，并标注“test”。test 的总运行次数是要公开报告的数字。
- checkpoint 选择一律按 `dev` 上的主指标。
- 在 test 上看错例只用于最终报告，看完不能回头改模型或 prompt。

## 2. 指标

### 2.1 主指标

**四档 macro-F1**（E/S/C/I 各类 F1 的算术平均）。

选它是因为数据中 E 占多数，accuracy 和 micro-F1 会被 E 主导；macro-F1 逼迫模型在 S/C 这类难点上也做好。代价是 C 类样本少、方差大，所以必须配合 §6 的置信区间来判断。

### 2.2 次要指标（都要报告，不作为验收门槛，除非 §6 另有说明）

| 指标 | 说明 |
|---|---|
| 各类 P / R / F1 | 定位短板 |
| micro-F1（= accuracy） | 便于和 KDD Cup 2022 Task 2 的口径对照 |
| 混淆矩阵 | 4×4，行是真实标签，列是预测 |
| NDCG@10 | 见 §2.3，衡量用作排序信号时的效果 |
| I 类 PR-AUC | 以 P(I) 为分数，衡量“过滤不相关商品”的能力 |
| 解析失败率 | 输出无法解析成合法档位的比例 |
| 截断率 | 生成达到 `max_new_tokens` 上限的比例 |
| 吞吐 | 在 D3 指定硬件上的 pair/s 和峰值显存 |

### 2.3 NDCG@10 的计算口径

- 每个商品的排序分 = Σ P(l) · gain(l)，gain 见 `specs/data.md` §2。没有概率输出的模型（如解析出的硬标签）用硬标签的 gain。
- 同分时按 `pair_id` 字典序打破，保证确定性。
- 理想排序按真实标签的 gain。
- 所有商品真实 gain 都为 0 的 query（IDCG = 0）不计入平均，单独报告数量。
- 最终值为所有 query 的 NDCG@10 的算术平均。

## 3. 模型输出协议

所有被评测的模型（包括 baseline）都先产出**预测文件**，再由独立的打分脚本计算指标。推理和打分分开，改指标不需要重跑推理。

预测文件每行：

```json
{"pair_id": "...", "pred": "E", "probs": {"E": 0.7, "S": 0.2, "C": 0.05, "I": 0.05},
 "raw_output": "...", "parse_ok": true, "truncated": false}
```

规则：

- **解码**：greedy（temperature = 0），不做采样。
- **两种输出模式**，实验记录中要注明用的是哪种：
  - 直答模式：只输出一个档位字母，`max_new_tokens = 8`
  - 推理模式：先推理，最后一行写 `Label: X`，`max_new_tokens = 512`
- **概率**：在输出档位的位置，取 E/S/C/I 四个 token 的 logits，只在这四个上做 softmax。拿不到 logits 的模型 `probs` 置为 `null`。
- **解析失败**（包括截断导致没有输出档位）：`pred = "INVALID"`，计入分母并按错误处理，不能剔除，也不能重试。
- 非 LLM 模型（B0、B3）同样输出这个格式，`raw_output` 置为 `null`。

## 4. Prompt 版本

- prompt 模板存放在 `prompts/relevance_v<N>.txt`，档位定义从 `specs/labeling-guideline.md` 提炼。
- 实验记录必须写明 prompt 版本。
- prompt 只能在 `dev` 上迭代。锁定时选定的 prompt 版本用于所有 LLM baseline；SFT 的输入模板可以不同，但要另行记录。

## 5. Baseline

| ID | 内容 | 作用 | 是否必须 |
|---|---|---|---|
| B0 | 全部预测为训练集多数类 | 下限，检查打分脚本 | 必须 |
| B1 | 待训练的基座模型，零样本，锁定的 prompt | 衡量后训练带来的提升 | 必须 |
| B2 | B1 + 固定的 k 个 few-shot 示例（从 train 抽取，seed 固定） | 区分“提示能解决多少”和“训练能解决多少” | 可选 |
| B3 | 在同一份训练数据上微调的 cross-encoder（小型 encoder + 四分类头） | 工业界常用的非 LLM 强基线 | 必须 |
| B4 | 大模型 API 零样本 | 参照上限，不作为验收门槛 | 可选；会产生费用，运行前需确认 |

B1 和 B3 的具体模型在 `specs/training.md` 中确定；B3 和 SFT 用的训练数据子集必须相同。

## 6. 统计判定与验收

**显著性**：在 test 上按 **query 做配对 bootstrap**（1000 次重采样，seed = 20261031），计算两个模型主指标之差的 95% 置信区间。**区间下界 > 0** 才算“显著优于”。

| 里程碑 | 验收条件（全部满足） |
|---|---|
| M1 SFT | ① macro-F1 显著优于 B1 和 B3；② 相对 B1 的提升 ≥ Δ₁（锁定时填写）；③ 解析失败率 < 0.5% |
| M2 后训练 | ① macro-F1 显著优于 M1；② 四类中任一类的 F1 比 M1 下降不超过 2 个点；③ 解析失败率不高于 M1 |

M2 没有达标不代表项目失败：按宪章要求如实记录结论和原因分析即可。

## 7. 锁定流程

在**第一次 SFT 训练之前**完成：

1. 数据管线产出 `v1`，满足 `specs/data.md` §8
2. 打分脚本满足本文 §9
3. 在 dev 上迭代 prompt，选定版本
4. 跑完 B0、B1、B3（以及选做的 B2）在 dev 和 test 上的结果，填入下表
5. 根据 baseline 结果填写 Δ₁
6. 将本文顶部的锁定状态改为“已锁定”，单独提交（提交信息 `lock eval v1`），并把 commit hash 填到顶部，同时在 `DECISIONS.md` 记录

| 模型 | dev macro-F1 | test macro-F1 | test micro-F1 | test NDCG@10 |
|---|---|---|---|---|
| B0 | | | | |
| B1 | | | | |
| B2 | | | | |
| B3 | | | | |

Δ₁ = ＿＿（锁定时填写）

**锁定后不可变**：test 的 query 列表、指标定义、输出协议、baseline 用的 prompt、验收阈值。确实需要改的话，开 `eval v2`：重新跑全部 baseline，`v1` 的结果保留并继续报告，并在 `DECISIONS.md` 说明原因。

## 8. 误差分析

- 每个里程碑：在 **dev** 上随机抽 100 个错例（seed 固定），按 `specs/labeling-guideline.md` §5 打 T1–T9 标签，报告分布，并与上一阶段对比。这部分可以用来指导下一步迭代。
- 最终报告：在 test 上做同样的分析，只用于报告。
- T9（疑似标注噪声）的占比单独报告，用来估计这份测试集上可达到的上限。

## 9. 评测管线的验收

- [ ] macro-F1、micro-F1 和各类 P/R/F1 在玩具数据上与 sklearn 的结果一致（单元测试）
- [ ] NDCG@10 在手算的小例子上结果一致，包括同分和 IDCG = 0 的情况（单元测试）
- [ ] 同一个预测文件、同一个 seed，bootstrap 结果逐位一致
- [ ] `INVALID` 预测计入分母（单元测试）
- [ ] 一条命令能从预测文件输出 §2 的全部指标，并生成可直接粘贴进 `EXPERIMENTS.md` 的一行
