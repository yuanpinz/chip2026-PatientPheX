# PatientPheX solver

CHIP2026 PatientPheX 的无本地 GPU 混合方案。Python 环境由 `uv` 管理，模型推理只调用赛题提供的 API，输出为赛题要求的 UTF-8 JSONL。

## 合规边界

正式模型使用 `modelE6-9-local`（9B），不使用超过 10B 的历史实验模型。训练集留一验证时，目标文章的标签不进入词典、表面校准、实体审核校准样本或关联 prompt。

训练数据、API 指南、PhenoTagger 发布包、模型缓存和 `outputs/` 均被 `.gitignore` 排除。

## 方法

- 从训练集和 HPO 本体构造词典候选，保留全局 offset、复合 HPO ID 和否定标记。
- 复用 PhenoTagger CNN 的高置信候选：`score >= 0.9997`、文本长度至少 6、每篇文章同一 HPO 最多 20 个 occurrence，并用训练集表面精度 `0.5` 做校准。
- 对 CNN 新增 occurrence 调用 9B API 做实体审核，仅在 `CASE`、`METHODS` 段且文本长度至少 9 的范围应用审核结果；词典实体和门控范围外的候选不受 API 拒绝影响。
- 用 9B API 在 occurrence 层面进行患者关联，不允许模型发明实体或 HPO ID。逐患者结果作为 primary，联合患者结果作为 secondary。
- 患者数为 2 到 4 时合并两路关联，范围外保留 primary。关联裁剪使用最终审核后的实体集合，避免被拒绝的实体重新带回关联。
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

先生成不调用模型的词典基线，并关闭 exact alias recovery：

```bash
uv run patientphex-solver predict \
  --data-dir PatientPheX-V1-A \
  --split a \
  --train-min-precision 0.5 \
  --no-recover-exact-train-aliases \
  --association proximity \
  --output outputs/pred_a_p05_norecover_base.jsonl
```

融合高置信 CNN 候选，并按训练集表面精度校准：

```bash
uv run patientphex-solver fuse-cnn-entities \
  --base outputs/pred_a_p05_norecover_base.jsonl \
  --cnn outputs/phenotagger_cnn_a_raw.jsonl \
  --surface-gold PatientPheX-V1-A/PatientPheX-train.jsonl \
  --surface-min-precision 0.5 \
  --surface-min-count 1 \
  --surface-calibration-max-per-identifier 1000000 \
  --min-score 0.9997 \
  --min-text-length 6 \
  --max-per-identifier 20 \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_cnn.jsonl \
  --additions-output outputs/pred_a_p05_surfacewide_p01_cap20_additions.jsonl
```

审核 CNN 新增实体。该步骤只对 `CASE METHODS` 且文本长度至少 9 的新增 occurrence 应用审核结果：

```bash
uv run patientphex-solver judge-entities \
  --data-dir PatientPheX-V1-A \
  --split a \
  --candidates outputs/pred_a_p05_surfacewide_p01_cap20_additions.jsonl \
  --model modelE6-9-local \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_additions_judged_e69.jsonl

uv run patientphex-solver filter-judged-entities \
  --data-dir PatientPheX-V1-A \
  --split a \
  --candidates outputs/pred_a_p05_surfacewide_p01_cap20_cnn.jsonl \
  --additions outputs/pred_a_p05_surfacewide_p01_cap20_additions.jsonl \
  --judged outputs/pred_a_p05_surfacewide_p01_cap20_additions_judged_e69.jsonl \
  --sections CASE METHODS \
  --min-text-length 9 \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_judged_gate_case_methods_len9_entities.jsonl
```

对新增 occurrence 调用合规 9B API。已有稳定关联可以复用；下面两路分别生成 primary 和 secondary：

```bash
uv run patientphex-solver judge-associations \
  --data-dir PatientPheX-V1-A \
  --split a \
  --candidates outputs/pred_a_p05_surfacewide_p01_cap20_additions.jsonl \
  --occurrence-level \
  --model modelE6-9-local \
  --structure-previous-distance 2500 \
  --structure-next-distance 750 \
  --propagate-explicit-groups \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_additions_e69_occ.jsonl

uv run patientphex-solver judge-associations \
  --data-dir PatientPheX-V1-A \
  --split a \
  --candidates outputs/pred_a_p05_surfacewide_p01_cap20_additions.jsonl \
  --occurrence-level \
  --joint \
  --model modelE6-9-local \
  --structure-previous-distance 2500 \
  --structure-next-distance 750 \
  --propagate-explicit-groups \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_additions_e69_joint.jsonl
```

用已有合规 9B 关联作为稳定基线，只为新增 occurrence 接受上面两路 API 判断：

```bash
uv run patientphex-solver stabilize-associations \
  --data-dir PatientPheX-V1-A \
  --split a \
  --base outputs/pred_a_compliant_e69_occ.jsonl \
  --entities outputs/pred_a_p05_surfacewide_p01_cap20_cnn.jsonl \
  --additions outputs/pred_a_p05_surfacewide_p01_cap20_additions.jsonl \
  --addition-associations outputs/pred_a_p05_surfacewide_p01_cap20_additions_e69_occ.jsonl \
  --sections CASE METHODS RESULTS \
  --final-structure-filter \
  --structure-previous-distance 2500 \
  --structure-next-distance 750 \
  --preserve-explicit-groups \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_matched_stabilized.jsonl

uv run patientphex-solver stabilize-associations \
  --data-dir PatientPheX-V1-A \
  --split a \
  --base outputs/pred_a_compliant_e69_occ.jsonl \
  --entities outputs/pred_a_p05_surfacewide_p01_cap20_cnn.jsonl \
  --additions outputs/pred_a_p05_surfacewide_p01_cap20_additions.jsonl \
  --addition-associations outputs/pred_a_p05_surfacewide_p01_cap20_additions_e69_joint.jsonl \
  --sections CASE METHODS RESULTS \
  --final-structure-filter \
  --structure-previous-distance 2500 \
  --structure-next-distance 750 \
  --preserve-explicit-groups \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_joint_stabilized.jsonl
```

在融合前，必须使用最终审核实体重新裁剪两路关联：

```bash
uv run patientphex-solver clip-associations \
  --input outputs/pred_a_p05_surfacewide_p01_cap20_matched_stabilized.jsonl \
  --entities outputs/pred_a_p05_surfacewide_p01_cap20_judged_gate_case_methods_len9_entities.jsonl \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_primary_final_clipped.jsonl

uv run patientphex-solver clip-associations \
  --input outputs/pred_a_p05_surfacewide_p01_cap20_joint_stabilized.jsonl \
  --entities outputs/pred_a_p05_surfacewide_p01_cap20_judged_gate_case_methods_len9_entities.jsonl \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_secondary_final_clipped.jsonl

uv run patientphex-solver fuse-associations \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --base outputs/pred_a_p05_surfacewide_p01_cap20_judged_gate_case_methods_len9_entities.jsonl \
  --primary outputs/pred_a_p05_surfacewide_p01_cap20_primary_final_clipped.jsonl \
  --secondary outputs/pred_a_p05_surfacewide_p01_cap20_secondary_final_clipped.jsonl \
  --union-patient-count-range 2 4 \
  --range-outside-primary \
  --output outputs/pred_a_p05_surfacewide_p01_cap20_judged_gate_case_methods_len9_primary_outside_joint_2to4.jsonl
```

两路稳定关联均使用 `CASE METHODS RESULTS`、前向 2500、后向 750、最终结构过滤和显式群组保留；如从头运行，可直接执行上面的 `stabilize-associations` 命令。

提交前校验：

```bash
uv run patientphex-solver validate \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --predicted outputs/pred_a_p05_surfacewide_p01_cap20_judged_gate_case_methods_len9_primary_outside_joint_2to4.jsonl
```

当前正式 A 集文件含 20 篇文章、1427 个实体和 470 个患者-表型关联值。平台未知标签不能用于本地评分，最终成绩以天池返回值为准。

## 严格留一结果

每篇训练文章的词典候选和 CNN 表面校准只使用另外 79 篇文章。当前工作树可复现的实体审核留一结果如下：

| 路线 | Mention F1 | Document F1 | Association micro F1 | Association macro F1 | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 严格留一词典 | 0.65590 | 0.72560 | 0.52208 | 0.43888 | 0.58561 |
| + 高置信 CNN | 0.68053 | 0.76447 | 0.52208 | 0.43888 | 0.60149 |
| + 9B occurrence 关联 | 0.68053 | 0.76447 | 0.65982 | 0.59490 | 0.67493 |
| + 稳定关联融合 | 0.68820 | 0.76813 | 0.66422 | 0.59746 | 0.67950 |
| + 表面精度校准 | 0.69071 | 0.77179 | 0.66568 | 0.59869 | 0.68172 |
| + 实体审核门控 | 0.69320 | 0.77318 | 0.67177 | 0.60098 | 0.68478 |

9B 自动实体补漏和 8B 简单投票未在全量留出上改善，因此不进入正式流程。最终线上分数仍需以天池提交结果为准。

## 其他命令

`--joint` 可以和 `--occurrence-level` 组合，让全部患者共同竞争 occurrence；`--no-structure-filter` 可导出模型原始选择用于离线审计。

```bash
uv run patientphex-solver evaluate \
  --gold PatientPheX-V1-A/PatientPheX-train.jsonl \
  --predicted outputs/pred_train.jsonl
```
