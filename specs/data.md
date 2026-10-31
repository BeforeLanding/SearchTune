# 数据 Spec

> 上位约束：`CONSTITUTION.md` D1（数据源）、D2（任务形式）。D1 已确认为 ESCI 英文。

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

> 本 spec 不写数据规模和标签分布的具体数字。下载后由数据管线统计，写入 `data/STATS.md`。已知 E 类占多数、C 类很少，具体比例以实测为准。

## 2. 标签

沿用 ESCI 四档，档位语义见 `specs/labeling-guideline.md`：

| 标签 | 含义（简述） | 排序增益 gain |
|---|---|---|
| `E` Exact | 完全满足 query | 1.0 |
| `S` Substitute | 不完全满足，但可以替代 | 0.1 |
| `C` Complement | 不满足，但与目标商品搭配使用 | 0.01 |
| `I` Irrelevant | 不相关，或违反核心约束 | 0.0 |

增益沿用 KDD Cup 2022 Task 1 的设定，供 `specs/eval.md` 计算 NDCG。

规则：**测试集标签永远用原始人工标签**，不做任何重标或清洗。训练集若要处理标注噪声，产出新的数据版本，原版本保留。

## 3. 统一 schema

处理后的每条样本是一行 JSON（`data/processed/<version>/<split>.jsonl`）：

```json
{
  "pair_id": "esci-us-<example_id>",
  "query_id": "<query_id>",
  "query": "aaa batteries 24 pack",
  "product_id": "<product_id>",
  "product": {
    "title": "...",
    "brand": "...",
    "color": "...",
    "bullets": "...",
    "description": "..."
  },
  "label": "E",
  "split": "train",
  "data_version": "v1"
}
```

字段规则：

- 商品字段缺失时为 `null`，不填空字符串，也不丢弃该样本（缺失本身是误差分析的一个维度）。
- `description` 去 HTML 标签、合并空白。
- 不在 schema 层截断文本；截断属于 prompt 构造，在 `DESIGN.md` 定义，并记录截断长度。
- `label` 只能是 `E/S/C/I`；出现其他值时管线直接报错。

## 4. 切分

**一律按 query 切分**，同一个 query 的所有商品只出现在一个 split 里。

| split | 来源 | 规模 | 用途 |
|---|---|---|---|
| `train` | ESCI `split == "train"`，扣除 dev 的 query | 剩余全部 | 训练；实际训练用的子集在 `specs/training.md` 中定义 |
| `dev` | 从 ESCI train 中按 query 抽 500 个 | 500 query | 调参、选 checkpoint、迭代 prompt |
| `dev-quick` | 从 dev 中抽 200 个 query | 200 query | 开发期快速冒烟评测 |
| `test` | 从 ESCI `split == "test"` 且 `small_version == 1` 中按 query 抽 1000 个 | 1000 query | 冻结测试集，规则见 `specs/eval.md` |

- 所有抽样 `seed = 20261031`。
- 每个 split 的 query_id 列表写入 `data/splits/<split>.txt`（**进 git**，体积小，是切分的唯一事实来源）。
- 选 `small_version == 1` 是因为这部分 query 更难，区分度更高。

## 5. 清洗与防泄漏

按顺序执行，每一步的删除数量写入 `data/STATS.md`：

1. 过滤 `product_locale != "us"`
2. 关联商品表；关联不上的样本删除并计数
3. 完全重复的 (query_id, product_id) 只保留一条；若重复项标签冲突，整组删除并计数
4. **跨 split query 去重**：query 规范化（小写、去首尾空白、合并连续空白）后，若 train 中的 query 与 dev/test 中的相同，从 train 删除该 query 的全部样本
5. 检查 train/dev/test 的 query_id 交集为空，否则管线报错

商品在不同 split 间重叠是允许的（真实搜索场景就是如此），但要在 `data/STATS.md` 中报告 test 里有多少比例的商品在 train 中出现过。

## 6. 版本与可复现

- 每次处理产出一个数据版本 `vN`；只要切分、清洗规则或字段处理任一项变化就升版本号，旧版本不覆盖。
- `data/MANIFEST.json`（进 git）记录：每个版本的原始文件 sha256、处理脚本的 git commit、每个 split 文件的 sha256 和行数、seed、许可证。
- 实验记录必须写明 `data_version`。

## 7. 目录约定

```
data/
  raw/                 # 原始下载，gitignore
  processed/<version>/ # 处理结果，gitignore
  splits/*.txt         # query_id 列表，进 git
  MANIFEST.json        # 进 git
  STATS.md             # 进 git，由管线生成
```

## 8. 验收

数据管线算完成，需要满足以下全部：

- [ ] 从空目录执行一条命令能复现 `v1`，且产物 sha256 与 MANIFEST 一致
- [ ] `data/STATS.md` 包含各 split 的 query 数、pair 数、标签分布、字段缺失率、每一步清洗的删除数、test 商品在 train 中的出现率
- [ ] train/dev/test 的 query_id 两两交集为空（有自动检查）
- [ ] 规范化后的 query 在 train 与 dev/test 之间无重叠（有自动检查）
