# 数据统计（v1，已被 v2 取代，仅留档；见 DECISIONS.md DEC-017）

> 由 `python -m searchtune prepare` 生成，请勿手改。

## 清洗与过滤（按执行顺序）

| 步骤 | 数量 |
|---|---|
| 原始样本行数（全部 locale） | 2,621,288 |
| 过滤 locale != us（删除） | 802,463 |
| us 样本行数 | 1,818,825 |
| 关联不上商品表（删除样本行） | 0 |
| (query_id, product_id) 重复且标签一致（删除多余行） | 0 |
| (query_id, product_id) 标签冲突（整组删除，组数） | 0 |
| query_id 同时出现在 ESCI 的 train 与 test（整个 query 删除） | 1 |
| 无 E/S/C 的 query（不建池，抽样前过滤） | 0 |
| dev 候选中因与 test 规范化 query 相同而排除的 query | 0 |
| train 中因与 dev/test 规范化 query 相同而删除的 query | 0 |

商品表 1,814,924 行；us 商品目录 1,215,854 个（重复行 0）。

## 各 split 的 query 数

| split | query 数 | 建池数 |
|---|---|---|
| train | 74,387 | 14,273 |
| dev | 500 | 500 |
| dev-quick | 200 | 200 |
| test | 1,000 | 1,000 |

train 只为 `rl_l1 ∪ sft_traj ∪ rl_l2` 中的 query 建池（DEC-015）。

## 池大小分布

| split | min | median | mean | p90 | max |
|---|---|---|---|---|---|
| test | 66 | 116.0 | 119.3 | 140 | 153 |
| dev | 61 | 116.0 | 117.6 | 131 | 159 |
| train | 60 | 116 | 117.2 | 130 | 212 |
| dev-quick | 66 | 116.0 | 117.7 | 130 | 146 |

## 池内标签比例

| split | E | S | C | I | U |
|---|---|---|---|---|---|
| test | 0.0763 | 0.0600 | 0.0074 | 0.0295 | 0.8269 |
| dev | 0.1124 | 0.0306 | 0.0037 | 0.0123 | 0.8409 |
| train | 0.1107 | 0.0311 | 0.0033 | 0.0136 | 0.8413 |
| dev-quick | 0.1100 | 0.0334 | 0.0027 | 0.0110 | 0.8429 |

## 商品字段缺失率（目录）

| 字段 | 缺失率 |
|---|---|
| brand | 0.0595 |
| color | 0.3340 |
| bullets | 0.1470 |
| description | 0.4687 |

## 泄漏相关

- test 池商品出现在 train 池中的比例：0.7486（仅标注商品：0.7050）
- BM25 干扰不足 50 个的池数：test 33，dev 14，train 504，dev-quick 4

## 难例切片

一个 query 可属于多个切片；成员列表见 `data/slices/<split>.json`。

| 切片 | 规则 | dev | test |
|---|---|---|---|
| negation | query 词中含 without / no / free / non / not / excluding / except 之一（连字符拆词，如 sugar-free） | 15 | 94 |
| quantity_spec | 含“数字 + 单位”（pack、count、oz、lb、inch、gb、v、w 等）、`数字x数字`，或含 pack / size / xl / xxl / small / medium / large | 35 | 90 |
| compat_model | 含 for 且其后任一词含数字；或含字母数字混合的型号串（长度 ≥ 3，如 a15、ps5、x570） | 18 | 65 |
| brand | query 中连续 1–3 个词等于目录中某个品牌（该品牌至少对应 brand_min_count 个商品）；单词品牌可能与普通词重名，存在噪声 | 73 | 163 |
| long | 词数 ≥ long_query_words（默认 6） | 60 | 138 |

品牌集合大小（目录中至少 50 个商品的品牌）：2,590。
