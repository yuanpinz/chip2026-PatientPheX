# PatientPheX solver

CHIP2026 PatientPheX 的无本地 GPU 混合方案。Python 环境由 `uv` 管理，模型推理只调用赛题提供的 API，输出为赛题要求的 UTF-8 JSONL。

## 合规边界

当前默认模型是 `modelE6-9-local`（Qwen3.5-9B），备用模型是 `modelK5`（Qwen3-8B），均不超过赛题规定的 10B 参数上限。

仓库历史曾实验 `modelH`、`modelS5_6S`、`modelK1-instruct-2507` 等超过 10B 或参数规模不符合要求的模型；它们产生的缓存和预测不属于当前方案，不得用于正式提交。曾包含目标文章标签信息的 replay/intersection 实验同样无效。当前结果从严格留一实体候选重新生成，不读取这些历史预测。

训练数据、API 指南、PhenoTagger 发布包、模型缓存和 `outputs/` 均被 `.gitignore` 排除。

## 方法

- 从训练集表面统计和 HPO 2026-06-23 本体构造离线词典，保留全局 offset、复合 HPO ID 和否定标记。
- 复用已生成的 PhenoTagger CNN 原始候选，只加入 `score >= 0.9997`、文本长度至少 6、不与现有 span 重叠的候选；每篇文章的同一 HPO ID 最多补充 10 次。
- 由 9B API 在 occurrence 层面逐患者选择候选，不允许模型发明实体或 HPO ID。
- 使用方向性文章结构窗口过滤患者串线：向前 4000 字符、向后 0 字符；显式的 `both/all patients` 句式再做保守传播。
- 所有 API 响应按完整请求 SHA-256 缓存到 `cache/llm/`，中断后可继续运行。

## 环境

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

API 端点默认使用指南中的地址，也可以覆盖：

```bash
export PATIENTPHEX_API_ENDPOINT='https://test.huihaohealth.com/ai-center/x/server/api/v1/big_model/chat'
```

## A 集流程

先生成不调用模型的词典基线：

```bash
uv run patientphex-solver predict \
  --data-dir PatientPheX-V1-A \
  --split a \
  --association proximity \
  --output outputs/pred_a_compliant_base.jsonl
```

融合已经生成的 PhenoTagger CNN 原始结果：

```bash
uv run patientphex-solver fuse-cnn-entities \
  --base outputs/pred_a_compliant_base.jsonl \
  --cnn outputs/phenotagger_cnn_a_raw.jsonl \
  --min-score 0.9997 \
  --min-text-length 6 \
  --max-per-identifier 10 \
  --output outputs/pred_a_compliant_cnn.jsonl
```

调用合规 9B 模型完成患者关联：

```bash
uv run patientphex-solver judge-associations \
  --data-dir PatientPheX-V1-A \
  --split a \
  --candidates outputs/pred_a_compliant_cnn.jsonl \
  --occurrence-level \
  --model modelE6-9-local \
  --structure-previous-distance 4000 \
  --structure-next-distance 0 \
  --propagate-explicit-groups \
  --output outputs/pred_a_compliant_e69_occ.jsonl
```

提交前校验：

```bash
uv run patientphex-solver validate \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --predicted outputs/pred_a_compliant_e69_occ.jsonl
```

当前 A 集文件含 20 篇文章、1487 个实体、53 个患者和 478 个患者-表型值。平台未知标签不能用于本地评分，最终成绩以天池返回值为准。

## 严格留一结果

每篇训练文章的词典候选只使用另外 79 篇文章构造，CNN 阈值和关联窗口均在全部 80 篇上核验，并同时检查前后两个不重叠的 40 篇子集。

| 路线 | Mention F1 | Document F1 | Association micro F1 | Association macro F1 | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 严格留一词典 | 0.65590 | 0.72560 | 0.52208 | 0.43888 | 0.58561 |
| + 高置信 CNN | 0.68053 | 0.76447 | 0.52208 | 0.43888 | 0.60149 |
| + 9B occurrence 关联 | 0.68053 | 0.76447 | 0.65982 | 0.59490 | 0.67493 |

9B 自动实体补漏的严格留一结果为 `mention F1 0.6783 / document F1 0.7627`，低于不补漏方案，因此正式流程不启用 `discover-entities`。8B 简单投票和联合 occurrence 关联也未在全量留出上改善，未进入最终流程。

重现训练集关联评估：

```bash
uv run patientphex-solver judge-associations \
  --split train \
  --candidates outputs/train_compliant_loo_cnn.jsonl \
  --occurrence-level \
  --model modelE6-9-local \
  --structure-previous-distance 4000 \
  --structure-next-distance 0 \
  --propagate-explicit-groups \
  --output outputs/train_compliant_loo_e69_occ_4000.jsonl

uv run patientphex-solver evaluate \
  --gold PatientPheX-V1-A/PatientPheX-train.jsonl \
  --predicted outputs/train_compliant_loo_e69_occ_4000.jsonl
```

## 其他命令

`--joint` 可以和 `--occurrence-level` 组合，让全部患者共同竞争 occurrence；`--no-structure-filter` 可导出模型原始选择用于离线审计。两者是诊断选项，不是当前推荐参数。

```bash
uv run patientphex-solver evaluate \
  --gold PatientPheX-V1-A/PatientPheX-train.jsonl \
  --predicted outputs/pred_train.jsonl
```
