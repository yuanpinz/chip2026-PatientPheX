# PatientPheX solver

这是 CHIP2026 PatientPheX 的可复现混合系统。项目使用 `uv` 管理 Python 3.12 环境和锁文件，预测结果输出为赛题要求的 UTF-8 JSONL。

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

所有 API 原始响应都按请求内容 SHA-256 缓存在 `cache/llm/`。该目录已被 `.gitignore` 排除，避免将数据或响应提交到 Git。
