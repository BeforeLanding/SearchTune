# 数据 Spec

> 上位约束：`CONSTITUTION.md` D1（数据源）、D2（任务形式）。

## 1. 数据来源

| 项 | 内容 |
|---|---|
| 数据集 | Amazon Shopping Queries Dataset（ESCI，KDD Cup 2022） |
| 获取 | `github.com/amazon-science/esci-data`，下载到 `data/raw/`，不进 git |
| 许可证 | 下载时核对仓库 LICENSE 并记入 `data/MANIFEST.json` |
| 使用范围 | 只用 `product_locale == "us"` 的数据 |

预期的原始文件与字段如下（**下载后以实际文件为准**；不一致就改本节，不在代码里迁就）：

- `shopping_queries_dataset_examples.parquet`：`example_id, query, query_id, product_id, product_locale, esci_label, small_version, large_version, split`
- `shopping_queries_dataset_products.parquet`：`product_id, product_title, product_description, product_bullet_point, product_brand, product_color, product_locale`

两表按 `(product_id, product_locale)` 关联。

> 本 spec 不写数据规模和标签分布的具体数字。下载后由数据管线统计，写入 `data/STATS.md`。

## 2. 标签与增益

| 标签 | 含义（简述，详见 `specs/labeling-guideline.md`） | 增益 gain |
|---|---|---|
| `E` Exact | 完全满足 query | 1.0 |
| `S` Substitute | 不完全满足，但可以替代 | 0.1 |
| `C` Complement | 不满足，但与目标商品搭配使用 | 0.01 |
| `I` Irrelevant | 不相关，或违反核心约束 | 0.0 |
| `U` Unlabeled | 干扰商品，没有该 query 下的人工标签 | 0.0（按不相关处理，见 §5.3） |

E/S/C/I 的增益沿用 KDD Cup 2022 Task 1 的设定。

规则：**测试集标签永远用原始人工标签**，不做任何重标或清洗。

## 3. 商品目录

所有商品统一存成 `data/processed/<version>/catalog.jsonl`，每行一个商品：

```json
{"product_id": "...", "title": "...", "brand": "...", "color": "...",
 "bullets": "...", "description": "..."}
```

- 字段缺失时为 `null`，不填空字符串。
- `description` 去 HTML 标签、合并空白。
- 不在这一层截断文本；截断属于工具返回值的格式，在 `specs/training.md` §2 中定义。

## 4. 切分

**一律按 query 切分**，同一个 query 只出现在一个 split 里。

| split | 来源 | 规模 | 用途 |
|---|---|---|---|
| `train` | ESCI `split == "train"`，扣除 dev 的 query | 剩余全部 | 训练；L1、L2 各自使用的子集在 `specs/training.md` §6 中定义 |
| `dev` | 从 ESCI train 中按 query 抽 500 个 | 500 query | 调参、选 checkpoint、迭代 prompt |
| `dev-quick` | 从 dev 中抽 200 个 query | 200 query | 开发期快速冒烟评测 |
| `test` | 从 ESCI `split == "test"` 且 `small_version == 1` 中按 query 抽 1000 个 | 1000 query | 冻结测试集，规则见 `specs/eval.md` |

- 所有抽样 `seed = 20261031`。
- 每个 split 的 query_id 列表写入 `data/splits/<split>.txt`（**进 git**，体积小，是切分的唯一事实来源）。
- 选 `small_version == 1` 是因为这部分 query 更难，区分度更高。

## 5. 商品池（环境的核心）

每个 query 对应一个**封闭的商品池**，Agent 只能在池里搜索和挑选。这样每个被提交的商品都有确定的增益，奖励完全可以验证。

### 5.1 构造规则

对每个 query q：

1. **标注商品** L(q)：ESCI 中 q 的全部已标注商品，带 E/S/C/I 标签。
2. **难干扰商品**：用 q 在商品目录（标题 + 品牌）上做 BM25 检索，排除 L(q) 和 §5.2 中的“疑似相关”商品后，取前 50 个，标为 `U`。
3. **随机干扰商品**：从商品目录中随机抽 50 个（排除同样的集合），标为 `U`。

池的大小约为 |L(q)| + 100。数量参数写进数据版本，修改就升版本号。

### 5.2 降低“漏标的相关商品”

BM25 检索出的干扰商品里，可能混有对 q 真正相关、只是 ESCI 没有标注的商品。为降低这种风险：

- 若某商品在另一个 query q′ 下被标为 E 或 S，并且 q 与 q′ 规范化后的词集合 Jaccard 相似度 ≥ 0.5，就不把它选作 q 的干扰商品。
- 残余风险由 `specs/eval.md` §8 的抽检来量化。

### 5.3 记录格式

`data/processed/<version>/pools/<split>.jsonl`，每行一个 query：

```json
{"query_id": "...", "query": "aaa batteries 24 pack", "split": "test",
 "pool": [{"product_id": "...", "label": "E", "gain": 1.0, "source": "labeled"},
          {"product_id": "...", "label": "U", "gain": 0.0, "source": "bm25"}],
 "data_version": "v1"}
```

- `source` 取值：`labeled` / `bm25` / `random`。
- **Agent 永远看不到 `label`、`gain`、`source`**，环境只把商品文本暴露给 Agent（`DESIGN.md` §6 有护栏）。

### 5.4 过滤

- 标注商品中没有任何 E/S/C 的 query（理想 NDCG 为 0）不建池，删除数量计入 STATS。

## 6. 清洗与防泄漏

按顺序执行，每一步的删除数量写入 `data/STATS.md`：

1. 过滤 `product_locale != "us"`
2. 关联商品表；关联不上的样本删除并计数
3. 完全重复的 (query_id, product_id) 只保留一条；若重复项标签冲突，整组删除并计数
4. **跨 split query 去重**：query 规范化（小写、去首尾空白、合并连续空白）后，若 train 中的 query 与 dev/test 中的相同，从 train 删除该 query
5. 检查 train/dev/test 的 query_id 交集为空，否则管线报错

商品在不同 split 之间重叠是允许的（标签是按 (query, 商品) 对给的，商品本身不泄漏标签），但要在 `data/STATS.md` 中报告 test 池中有多少比例的商品出现在 train 池里。

## 7. 版本与可复现

- 每次处理产出一个数据版本 `vN`；切分、清洗规则、商品池参数任一项变化就升版本号，旧版本不覆盖。
- `data/MANIFEST.json`（进 git）记录：原始文件 sha256、处理脚本的 git commit、每个产物文件的 sha256 和行数、seed、池参数、许可证。
- 实验记录必须写明 `data_version`。

## 8. 目录约定

```
data/
  raw/                       # 原始下载，gitignore
  processed/<version>/       # 处理结果，gitignore
    catalog.jsonl
    pools/<split>.jsonl
  splits/*.txt               # query_id 列表，进 git
  MANIFEST.json              # 进 git
  STATS.md                   # 进 git，由管线生成
```

## 9. 验收

数据管线算完成，需要满足以下全部：

- [ ] 从空目录执行一条命令能复现 `v1`，且产物 sha256 与 MANIFEST 一致
- [ ] `data/STATS.md` 包含：各 split 的 query 数；池大小分布；池内 E/S/C/I/U 的比例；字段缺失率；每一步清洗和过滤的删除数；test 池商品在 train 池中的出现率
- [ ] train/dev/test 的 query_id 两两交集为空（有自动检查）
- [ ] 规范化后的 query 在 train 与 dev/test 之间无重叠（有自动检查）
- [ ] 每个池中，干扰商品与 L(q) 不相交，且满足 §5.2 的排除规则（有自动检查）
