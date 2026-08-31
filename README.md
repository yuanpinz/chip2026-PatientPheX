# PatientPheX solver

这是 CHIP2026 PatientPheX 的可复现混合系统。项目使用 `uv` 管理 Python 3.12 环境和锁文件，预测结果输出为赛题要求的 UTF-8 JSONL。模型推理全部通过赛题提供的 API 完成，本地不需要 GPU。

## 组成

- `patientphex_solver.ontology`：解析赛题提供的 HPO 2026-06-23 OBO，并限制在 `HP:0000118` 分支。
- `patientphex_solver.entities`：用训练集表面统计与 HPO 词典做离线实体候选抽取，保留原文全局 offset、复合 ID 和否定标记；旧版 PhenoTagger 词典仅作为显式实验选项。
- `patientphex_solver.llm_entities`：可选调用 API 让 9B 模型补充词典遗漏实体，之后由本地 HPO 链接器确定 ID。
- `patientphex_solver.association`：使用实体候选索引作为受限动作空间，由大模型完成患者-表型关联，并用患者局部文章结构过滤多患者串线；也提供无 API 的距离基线。
- `patientphex_solver.evaluation`：实现赛题中的 mention/document、micro/macro F1 与总分。

`PhenoTagger-master/` 是用户提供的官方代码包，不纳入 Git；它可用于额外实验，但发布包不含模型权重。官方 PhenoTagger API 是异步远程服务，未作为默认路径，以保证预测可缓存且可复现。

## 环境

```bash
uv sync
```

API 端点可以用环境变量覆盖：

```bash
export PATIENTPHEX_API_ENDPOINT='https://test.huihaohealth.com/ai-center/x/server/api/v1/big_model/chat'
```

## 常用命令

先生成离线基线：

```bash
uv run patientphex-solver predict \
  --data-dir PatientPheX-V1-A \
  --split a \
  --association proximity \
  --output outputs/pred_baseline.jsonl
```

生成推荐候选（患者关联使用可缓存的大模型 API；实体抽取默认采用经交叉验证的离线词典）：

```bash
uv run patientphex-solver predict \
  --data-dir PatientPheX-V1-A \
  --split a \
  --use-llm \
  --entity-batch article \
  --association consensus-structured \
  --model modelS5_6S \
  --cache-dir cache/llm \
  --output outputs/pred_a.jsonl
```

`consensus-structured` 对 per-patient 与 joint 两种实体 occurrence 选择分别施加患者局部结构约束，
再按患者合并；训练集金标准实体实验中，关联 micro/macro F1 为 `0.8441/0.7895`。

患者关联也可以使用更强的模型。文章级实体发现是保守实验模式，只接受模型输出文本本身为
HPO 精确别名的新增实体，避免模糊链接引入大量误报：

```bash
uv run patientphex-solver predict \
  --data-dir PatientPheX-V1-A \
  --split a \
  --use-llm \
  --entity-batch article \
  --association patient-structured \
  --model modelS5_6S \
  --cache-dir cache/llm \
  --output outputs/pred_a_s56.jsonl
```

提交前校验：

```bash
uv run patientphex-solver validate \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --predicted outputs/pred_a.jsonl
```

训练集有答案时可评分：

```bash
uv run patientphex-solver evaluate \
  --gold PatientPheX-V1-A/PatientPheX-train.jsonl \
  --predicted outputs/pred_train.jsonl
```

纯 API 校准工具：先准备包含 `pmc_id` 和 `entities` 的候选 JSONL，再让 API 按训练集标注风格筛选实体或关联患者。训练集模式会排除当前文章的校准样例，便于做留出验证：

```bash
uv run patientphex-solver judge-entities \
  --split train \
  --candidates outputs/train_candidates.jsonl \
  --model modelS5_6S \
  --output outputs/train_judged.jsonl

uv run patientphex-solver judge-associations \
  --split a \
  --candidates outputs/pred_a_entities.jsonl \
  --model modelS5_6S \
  --output outputs/pred_a_final.jsonl
```

联合校准关联会让多个患者在同一请求中竞争候选表型；`--include-uncertain` 会保留模型标记为边界情况的结果：

```bash
uv run patientphex-solver judge-associations \
  --split a \
  --candidates outputs/pred_a_v7_base.jsonl \
  --joint \
  --include-uncertain \
  --model modelS5_6S \
  --output outputs/pred_a_calibrated.jsonl
```

如果已经用官方 PhenoTagger CNN 生成了原始 JSONL，可以用下面的保守路径补充词典遗漏的实体。
默认规则由严格留一验证选择：只保留高置信、长度至少 6 的候选，且不与已有实体 span 重叠；同一 HPO
ID 在一篇文章中最多补充 10 次。`--additions-output` 用于把新增项单独送入 API，避免 API 重判已有词典实体。

```bash
uv run patientphex-solver fuse-cnn-entities \
  --base outputs/pred_a_v7_entity_consensus.jsonl \
  --cnn outputs/phenotagger_cnn_a_raw.jsonl \
  --output outputs/pred_a_cnn_fused.jsonl \
  --additions-output outputs/pred_a_cnn_additions.jsonl

uv run patientphex-solver judge-entities \
  --split a \
  --candidates outputs/pred_a_cnn_additions.jsonl \
  --model modelH \
  --include-uncertain \
  --output outputs/pred_a_cnn_additions_judged_h_uncertain.jsonl

uv run patientphex-solver merge-entities \
  --base outputs/pred_a_v7_entity_consensus.jsonl \
  --additions outputs/pred_a_cnn_additions_judged_h_uncertain.jsonl \
  --output outputs/pred_a_cnn_judged_entities.jsonl
```

在新增实体上运行多个联合患者关联模型后，可用投票融合规则生成结果。下面的五路组合和局部结构参数由两组互不重叠的严格留出文章验证选择：

```bash
uv run patientphex-solver judge-associations \
  --split a --candidates outputs/pred_a_cnn_judged_entities.jsonl \
  --joint --model modelH \
  --output outputs/pred_a_cnn_modelh_joint_strict.jsonl

uv run patientphex-solver judge-associations \
  --split a --candidates outputs/pred_a_cnn_judged_entities.jsonl \
  --joint --include-uncertain --model modelH \
  --output outputs/pred_a_cnn_modelh_joint.jsonl

uv run patientphex-solver judge-associations \
  --split a --candidates outputs/pred_a_cnn_judged_entities.jsonl \
  --joint --model modelS5_6S \
  --output outputs/pred_a_cnn_s56_joint_strict.jsonl

uv run patientphex-solver judge-associations \
  --split a --candidates outputs/pred_a_cnn_judged_entities.jsonl \
  --joint --include-uncertain --model modelS5_6S \
  --output outputs/pred_a_cnn_s56_joint.jsonl

uv run patientphex-solver judge-associations \
  --split a --candidates outputs/pred_a_cnn_judged_entities.jsonl \
  --joint --include-uncertain --model modelK1-instruct-2507 \
  --output outputs/pred_a_cnn_k1_joint.jsonl

uv run patientphex-solver vote-associations \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --base outputs/pred_a_cnn_judged_entities.jsonl \
  --sources \
    outputs/pred_a_cnn_modelh_joint_strict.jsonl \
    outputs/pred_a_cnn_modelh_joint.jsonl \
    outputs/pred_a_cnn_s56_joint_strict.jsonl \
    outputs/pred_a_cnn_s56_joint.jsonl \
    outputs/pred_a_cnn_k1_joint.jsonl \
  --min-votes 2 \
  --single-patient-source-indices 1 2 3 5 \
  --single-patient-min-votes 2 \
  --structure-previous-distance 800 \
  --structure-next-distance 200 \
  --structure-wide-sections CASE RESULTS FIG TABLE \
  --structure-wide-previous-distance 5000 \
  --structure-wide-next-distance 200 \
  --propagate-explicit-groups \
  --output outputs/pred_a_cnn_final_v2.jsonl

uv run patientphex-solver validate \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --predicted outputs/pred_a_cnn_final_v2.jsonl
```

最终五源策略在两组互不重叠的 10 篇留出文章上分别得到 `0.719385` 和 `0.742198`，合并 20 篇评分为 `0.735699`；加入 K1 严格实体补充、S5.6 扩展 CNN 实体补充及 H 联合关联增强后，留出分数为 `0.728466`、`0.764182`，合并为 `0.752624`。进一步用 H、S5.6、S5.5 三个模型对扩展 CNN 候选做 2/3 多数票，并对新增实体关联取 H/S5.6 的 2/2 交集，留出分数为 `0.731542`、`0.767005`，合并为 `0.755623`。留出结果仅用于本地模型选择，最终榜单成绩仍以提交平台返回值为准。

最终实体增强路线不需要本机 GPU 训练。它复用已经生成的 PhenoTagger 原始候选，先用 K1 严格模式补充保守候选，再从 H-uncertain + K1-strict 实体上提取更宽松的 CNN 候选，并用 S5.6 严格模式裁决：

```bash
uv run patientphex-solver judge-entities \
  --split a \
  --candidates outputs/pred_a_cnn_additions.jsonl \
  --model modelK1-instruct-2507 \
  --output outputs/pred_a_cnn_additions_judged_k1.jsonl

uv run patientphex-solver merge-entities \
  --base outputs/pred_a_cnn_judged_entities.jsonl \
  --additions outputs/pred_a_cnn_additions_judged_k1.jsonl \
  --output outputs/pred_a_cnn_judged_entities_hu_ks.jsonl

uv run patientphex-solver fuse-cnn-entities \
  --base outputs/pred_a_cnn_judged_entities_hu_ks.jsonl \
  --cnn outputs/phenotagger_cnn_a_raw.jsonl \
  --min-score 0.95 \
  --min-text-length 2 \
  --max-per-identifier 100 \
  --allow-overlap \
  --output outputs/pred_a_cnn_broad095.jsonl \
  --additions-output outputs/pred_a_cnn_broad095_additions.jsonl

uv run patientphex-solver judge-entities \
  --split a \
  --candidates outputs/pred_a_cnn_broad095_additions.jsonl \
  --model modelS5_6S \
  --output outputs/pred_a_cnn_broad095_judged_s56.jsonl
```

`judge-entities` 未指定 `--include-uncertain`，因此上面两次 API 裁决都只保留明确接受的实体。最终合并支持一次传入多个裁决文件，重复实体会按 offset、length、identifier 和 note 去重：

```bash
uv run patientphex-solver merge-entities \
  --base outputs/pred_a_cnn_final_v2.jsonl \
  --additions outputs/pred_a_cnn_additions_judged_k1.jsonl \
             outputs/pred_a_cnn_broad095_judged_s56.jsonl \
  --output outputs/pred_a_cnn_final_v3_base.jsonl
```

对于扩展候选，可以让三个模型分别裁决，再只保留至少两个模型接受的完全相同标注。S5.5 等推理模型建议提高输出上限：

```bash
uv run patientphex-solver judge-entities \
  --split a \
  --candidates outputs/pred_a_cnn_broad095_additions.jsonl \
  --model modelS5_5 \
  --batch-size 10 \
  --calibration-per-label 6 \
  --max-tokens 4000 \
  --output outputs/pred_a_cnn_broad095_judged_s55.jsonl

uv run patientphex-solver judge-entities \
  --split a \
  --candidates outputs/pred_a_cnn_broad095_additions.jsonl \
  --model modelH \
  --output outputs/pred_a_cnn_broad095_judged_h.jsonl

uv run patientphex-solver merge-entities \
  --base outputs/pred_a_cnn_final_v2.jsonl \
  --additions outputs/pred_a_cnn_additions_judged_k1.jsonl \
  --output outputs/pred_a_cnn_final_v4_k1_base.jsonl

uv run patientphex-solver vote-entities \
  --base outputs/pred_a_cnn_final_v4_k1_base.jsonl \
  --sources outputs/pred_a_cnn_broad095_judged_h.jsonl \
           outputs/pred_a_cnn_broad095_judged_s56.jsonl \
           outputs/pred_a_cnn_broad095_judged_s55.jsonl \
  --min-votes 2 \
  --output outputs/pred_a_cnn_final_v4_base.jsonl
```

对于只新增实体的关联，可以保留已有五源结果，再把新实体的 modelH 联合关联按文章结构过滤后并入：

```bash
uv run patientphex-solver judge-associations \
  --split a --candidates outputs/pred_a_cnn_broad095_judged_s56.jsonl \
  --joint --model modelH \
  --output outputs/pred_a_cnn_broad095_assoc_h.jsonl

uv run patientphex-solver augment-associations \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --base outputs/pred_a_cnn_final_v3_base.jsonl \
  --sources outputs/pred_a_cnn_broad095_assoc_h.jsonl \
  --min-votes 1 \
  --structure-previous-distance 800 \
  --structure-next-distance 200 \
  --structure-wide-sections CASE RESULTS FIG TABLE \
  --structure-wide-previous-distance 5000 \
  --structure-wide-next-distance 200 \
  --propagate-explicit-groups \
  --output outputs/pred_a_cnn_final_v3.jsonl

uv run patientphex-solver validate \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --predicted outputs/pred_a_cnn_final_v3.jsonl
```

正式 A 集推荐结果包含 20 篇文章、1650 个实体和 495 个患者-表型关联值；`outputs/pred_a_cnn_final_v4.jsonl` 的 SHA-256 为 `c3cda568f278d1694a9bd607541617aca91ea35e1e223147d942b8191d8d100a`。其关联生成命令为：

```bash
uv run patientphex-solver judge-associations \
  --split a \
  --candidates outputs/pred_a_cnn_broad095_judged_s56.jsonl \
  --joint \
  --model modelS5_6S \
  --output outputs/pred_a_cnn_broad095_assoc_s56.jsonl

uv run patientphex-solver augment-associations \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --base outputs/pred_a_cnn_final_v4_base.jsonl \
  --sources outputs/pred_a_cnn_broad095_assoc_h.jsonl \
           outputs/pred_a_cnn_broad095_assoc_s56.jsonl \
  --min-votes 2 \
  --structure-previous-distance 800 \
  --structure-next-distance 200 \
  --structure-wide-sections CASE RESULTS FIG TABLE \
  --structure-wide-previous-distance 5000 \
  --structure-wide-next-distance 200 \
  --output outputs/pred_a_cnn_final_v4.jsonl

uv run patientphex-solver validate \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --predicted outputs/pred_a_cnn_final_v4.jsonl
```

在 v4 之上还可以补充文章内明确定义的短缩写。候选由长名称与括号中的缩写确定，
不需要本地模型；H、S5.6、S5.5 三路 API 必须全部接受同一 occurrence，并只保留
长度不超过 3 的短缩写。该限制用于排除 `MODY`、`TRMA` 等在留出集上大量误报的
长疾病缩写：

```bash
uv run patientphex-solver abbreviation-candidates \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --base outputs/pred_a_cnn_final_v4_base.jsonl \
  --output outputs/pred_a_abbreviation_candidates.jsonl

uv run patientphex-solver judge-entities \
  --split a --candidates outputs/pred_a_abbreviation_candidates.jsonl \
  --model modelH \
  --output outputs/pred_a_abbreviation_judged_h.jsonl

uv run patientphex-solver judge-entities \
  --split a --candidates outputs/pred_a_abbreviation_candidates.jsonl \
  --model modelS5_6S \
  --output outputs/pred_a_abbreviation_judged_s56.jsonl

uv run patientphex-solver judge-entities \
  --split a --candidates outputs/pred_a_abbreviation_candidates.jsonl \
  --model modelS5_5 --batch-size 10 --calibration-per-label 6 \
  --max-tokens 4000 \
  --output outputs/pred_a_abbreviation_judged_s55.jsonl

uv run patientphex-solver vote-entities \
  --base outputs/pred_a_cnn_final_v4.jsonl \
  --sources outputs/pred_a_abbreviation_judged_h.jsonl \
            outputs/pred_a_abbreviation_judged_s56.jsonl \
            outputs/pred_a_abbreviation_judged_s55.jsonl \
  --min-votes 3 --max-text-length 3 \
  --output outputs/pred_a_cnn_final_v5.jsonl
```

短缩写全票策略在两组 10 篇留出上的总分为 `0.741764` 和 `0.767817`，
合并 20 篇为 `0.760195`；同一次重放中的 v4 分别为 `0.733771`、`0.768009`
和 `0.757050`。正式 A 集 v5 仍须以平台分数为准。

在 v5 之上，可以只补充三个独立强模型一致认可的患者-表型关联。三个模型均在
v5 的完整实体集合上联合判断所有患者，不启用 uncertain；最终仅添加三路全票的
关联，不改变实体。该策略在两组留出上的总分为 `0.741764` 和 `0.769873`，合并
20 篇由 v5 的 `0.760195` 提升到 `0.761581`。第一组没有新增项，第二组增加 3 个
TP、0 个 FP，因此保留 v5 作为无额外关联的回退版本：

```bash
uv run patientphex-solver judge-associations \
  --split a --candidates outputs/pred_a_cnn_final_v5.jsonl \
  --joint --model modelE6-397 \
  --output outputs/pred_a_cnn_e6397_joint_v5.jsonl

uv run patientphex-solver judge-associations \
  --split a --candidates outputs/pred_a_cnn_final_v5.jsonl \
  --joint --model modelB3-v4-p \
  --output outputs/pred_a_cnn_b3v4p_joint_v5.jsonl

uv run patientphex-solver judge-associations \
  --split a --candidates outputs/pred_a_cnn_final_v5.jsonl \
  --joint --model modelH-4.6O \
  --output outputs/pred_a_cnn_h46o_joint_v5.jsonl

uv run patientphex-solver augment-associations \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --base outputs/pred_a_cnn_final_v5.jsonl \
  --sources outputs/pred_a_cnn_e6397_joint_v5.jsonl \
            outputs/pred_a_cnn_b3v4p_joint_v5.jsonl \
            outputs/pred_a_cnn_h46o_joint_v5.jsonl \
  --min-votes 3 \
  --output outputs/pred_a_cnn_final_v7.jsonl

uv run patientphex-solver validate \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --predicted outputs/pred_a_cnn_final_v7.jsonl
```

正式 A 集 v7 包含 20 篇文章、1674 个实体和 501 个患者-表型关联值；
`outputs/pred_a_cnn_final_v7.jsonl` 的 SHA-256 为
`b62163fe3b1a0ae28a556a11186de80471fa38e450bc1fa13eee32c7f6617cc7`。
留出结果只用于选择融合规则，是否超过榜首仍以天池平台实际评分为准。

当前推荐的 API-only 路线在 v7 的基础上取两个独立 API 实体结果的完整 occurrence
交集，再用 `modelH` 对所有患者做联合关联。关联阶段针对患者数量使用条件融合：患者数为
`2–7` 时取 v7 关联和 `modelH` 关联的并集，患者数为 `1` 或大于 `7` 时只使用
`modelH` 关联；最后删除没有对应正实体的关联值。这一步同时支持复合 HPO ID 与 `-1`
未映射文本，且不需要 GPU。

相较于对所有文章直接并集的旧策略（两组严格留出分别为 `0.839225` 和 `0.861317`，
合并 20 篇为 `0.854794`），条件策略分别得到 `0.843558` 和 `0.863813`，合并 20 篇为
`0.857876`。合并关联评估为 micro F1 `0.833474`、macro F1 `0.808896`，TP/FP/FN
为 `493/107/90`。A 集结果包含 20 篇文章、1284 个实体和 401 个关联值：

```bash
uv run patientphex-solver select-entities \
  --base outputs/pred_a_cnn_final_v7.jsonl \
  --sources outputs/pred_a_s56_entities_fuzzy_cached.jsonl \
           outputs/pred_a_e6_entities_replay.jsonl \
  --min-votes 2 \
  --output outputs/pred_a_api_entity_intersection_v7.jsonl

uv run patientphex-solver judge-associations \
  --split a \
  --candidates outputs/pred_a_api_entity_intersection_v7.jsonl \
  --joint \
  --model modelH \
  --output outputs/pred_a_api_intersection_modelh_joint.jsonl

uv run patientphex-solver fuse-associations \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --base outputs/pred_a_api_entity_intersection_v7.jsonl \
  --primary outputs/pred_a_api_entity_intersection_v7.jsonl \
  --secondary outputs/pred_a_api_intersection_modelh_joint.jsonl \
  --union-patient-count-range 2 7 \
  --output outputs/pred_a_api_intersection_v7_h_conditional.jsonl

uv run patientphex-solver clip-associations \
  --input outputs/pred_a_api_intersection_v7_h_conditional.jsonl \
  --output outputs/pred_a_api_intersection_v7_h_conditional_clipped.jsonl

uv run patientphex-solver validate \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --predicted outputs/pred_a_api_intersection_v7_h_conditional_clipped.jsonl
```

推荐提交文件 `outputs/pred_a_api_intersection_v7_h_conditional_clipped.jsonl` 的
SHA-256 为 `dca6bcfa984701225542f338686f7e1a117e7fbc23e5d2924155e8b27862b328`。
留出分数用于本地选择策略，最终排名仍以天池实际提交结果为准。

也可以用 held-out 训练文章作为 few-shot 示例，让 API 发现词典遗漏的候选实体。模型只负责提出原文 span，HPO ID 仍由本地本体链接器校验：

```bash
uv run patientphex-solver discover-entities \
  --split a \
  --candidates outputs/pred_a_v7_base.jsonl \
  --model modelS5_6S \
  --output outputs/pred_a_discovered.jsonl
```

严格留出验证选出的最终关联融合策略：单患者文章取两个模型结果的并集，多患者文章采用联合模型结果。实体始终来自 `--base` 文件：

```bash
uv run patientphex-solver fuse-associations \
  --expected PatientPheX-V1-A/PatientPheX-A.jsonl \
  --base outputs/pred_a_v7_entity_consensus.jsonl \
  --primary outputs/pred_a_v8_calibrated_assoc_uncertain.jsonl \
  --secondary outputs/pred_a_v13_modelh_joint_entities1322.jsonl \
  --output outputs/pred_a_final_fused.jsonl
```

所有 API 原始响应都按请求内容 SHA-256 缓存在 `cache/llm/`。该目录已被 `.gitignore` 排除，避免将数据或响应提交到 Git。
