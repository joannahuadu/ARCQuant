# Full Eval 实验记录：joint_plus 与 bf16 teacher 全量评测

**日期**：2026-03-28  
**目标**：将当前两条最值得继续验证的方案
- `joint_plus`
- `joint_bf16_teacher`

分别在：
- 全部 `lm_eval`
- 全部 `LongBench`

上完整评测一遍，并将结果保存为独立 JSON，便于后续横向对比。

---

## 实验对象

### 1. joint_plus

**来源**：`P2-E`  
**定位**：在 `joint_v1` 基础上加入 `output_scale`，是目前 `P2` 线下最优方案。

**checkpoint**
```text
outputs/Llama-3.1-8B/NVFP4/joint_plus_20260327_040637/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt
```

**已有局部结果**
| Task | score |
|------|-------|
| arc_challenge | 0.5111 |
| rte | 0.6787 |
| ceval-valid | 0.4510 |

---

### 2. joint_bf16_teacher

**来源**：`P3-A1`  
**定位**：将 `teacher` 从 `NVFP4 dense` 切到 `bf16 dense` 的目标函数修正实验。

**checkpoint**
```text
outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_20260327_072940/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt
```

**已有局部结果**
| Task | score |
|------|-------|
| arc_challenge | 0.4881 |
| rte | 0.7040 |
| ceval-valid | 0.4324 |

---

## 评测范围

### lm_eval 全量任务

```text
arc_challenge
arc_easy
boolq
mmlu
ceval
openbookqa
piqa
rte
winogrande
```

说明：
- `model/main.py` 内部会将 `ceval` 映射为 `ceval-valid`
- 默认 few-shot override：
  - `mmlu = 5`
  - `ceval-valid = 5`
- 默认 batch size override：
  - `mmlu = 1`
  - `ceval-valid = 1`
- 其余任务使用 `--lm_eval_batch_size 4`

### LongBench 全量任务

本仓库 `model/eval_longbench.py` 的 `ALL_TASKS` 即为全量，评测脚本不显式传 `--tasks`，直接使用默认全集。

---

## 启动方式

为避免与正在运行的训练 / 诊断评测冲突，本轮统一安排在 **GPU 3** 串行执行。

### 调度脚本

```text
run_full_eval_joint_plus_bf16_teacher.sh
```

### 后台启动命令

```bash
setsid bash /data/shichao/ARCQuant/run_full_eval_joint_plus_bf16_teacher.sh \
  >> /data/shichao/ARCQuant/results/logs/run_full_eval_joint_plus_bf16_teacher.log 2>&1 \
  < /dev/null &
```

### 队列顺序

1. `joint_plus` full `lm_eval`
2. `joint_plus` full `LongBench`
3. `joint_bf16_teacher` full `lm_eval`
4. `joint_bf16_teacher` full `LongBench`

---

## 实际评测命令

### 1. joint_plus lm_eval

```bash
CUDA_VISIBLE_DEVICES=3 /home/shichao/miniconda3/envs/smoothquant/bin/python model/main.py \
    /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_plus_20260327_040637/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --softmax_alpha_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_plus_20260327_040637/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --tasks arc_challenge arc_easy boolq mmlu ceval openbookqa piqa rte winogrande \
    --lm_eval_batch_size 4 \
    --output_file /data/shichao/ARCQuant/results/joint_plus_lm_eval.json
```

### 2. joint_plus LongBench

```bash
CUDA_VISIBLE_DEVICES=3 /home/shichao/miniconda3/envs/smoothquant/bin/python model/eval_longbench.py \
    /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_plus_20260327_040637/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --softmax_alpha_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_plus_20260327_040637/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --output_file /data/shichao/ARCQuant/results/joint_plus_longbench.json
```

### 3. joint_bf16_teacher lm_eval

```bash
CUDA_VISIBLE_DEVICES=3 /home/shichao/miniconda3/envs/smoothquant/bin/python model/main.py \
    /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_20260327_072940/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --softmax_alpha_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_20260327_072940/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --tasks arc_challenge arc_easy boolq mmlu ceval openbookqa piqa rte winogrande \
    --lm_eval_batch_size 4 \
    --output_file /data/shichao/ARCQuant/results/joint_bf16_teacher_lm_eval.json
```

### 4. joint_bf16_teacher LongBench

```bash
CUDA_VISIBLE_DEVICES=3 /home/shichao/miniconda3/envs/smoothquant/bin/python model/eval_longbench.py \
    /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_20260327_072940/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --softmax_alpha_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_bf16_teacher_20260327_072940/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --output_file /data/shichao/ARCQuant/results/joint_bf16_teacher_longbench.json
```

---

## 输出文件

### 结果 JSON

```text
results/joint_plus_lm_eval.json
results/joint_plus_longbench.json
results/joint_bf16_teacher_lm_eval.json
results/joint_bf16_teacher_longbench.json
```

### 日志

```text
results/logs/run_full_eval_joint_plus_bf16_teacher.log
results/logs/joint_plus_lm_eval.log
results/logs/joint_plus_longbench.log
results/logs/joint_bf16_teacher_lm_eval.log
results/logs/joint_bf16_teacher_longbench.log
```

---

## 当前运行状态

<!-- AUTO_STATUS_START -->
**记录时间**：2026-03-28 10:14 UTC

| Job | 状态 |
|-----|------|
| joint_plus lm_eval | 🔄 running |
| joint_plus LongBench | ⏳ pending |
| joint_bf16_teacher lm_eval | ⏳ pending |
| joint_bf16_teacher LongBench | ⏳ pending |

### 当前可见进程

```text
shichao  1452711 1452702 99 10:01 ?        01:52:23 /home/shichao/miniconda3/envs/smoothquant/bin/python model/main.py /data/shichao/models/Llama-3.1-8B --quant_type NVFP4 --no_xw_reorder --use_x_mask --x_mask_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_plus_20260327_040637/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt --softmax_alpha_ckpt /data/shichao/ARCQuant/outputs/Llama-3.1-8B/NVFP4/joint_plus_20260327_040637/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt --tasks arc_challenge arc_easy boolq mmlu ceval openbookqa piqa rte winogrande --lm_eval_batch_size 4 --output_file /data/shichao/ARCQuant/results/joint_plus_lm_eval.json
```

### 调度日志摘录

```text
[2026-03-28 10:00:00] Start lm_eval for joint_plus on GPU 3.
[2026-03-28 10:00:41] Start lm_eval for joint_plus on GPU 3.
[2026-03-28 10:01:38] Start lm_eval for joint_plus on GPU 3.
```
<!-- AUTO_STATUS_END -->

## lm_eval 结果汇总

<!-- AUTO_LM_EVAL_START -->
| Task | Metric | joint_plus | joint_bf16_teacher |
|------|--------|------------|--------------------|
| arc_challenge | acc_norm | TBD | TBD |
| arc_easy | acc | TBD | TBD |
| boolq | acc | TBD | TBD |
| mmlu | acc | TBD | TBD |
| ceval-valid | acc | TBD | TBD |
| openbookqa | acc_norm | TBD | TBD |
| piqa | acc | TBD | TBD |
| rte | acc | TBD | TBD |
| winogrande | acc | TBD | TBD |
<!-- AUTO_LM_EVAL_END -->

## LongBench 结果汇总

<!-- AUTO_LONGBENCH_START -->
| Task | joint_plus | joint_bf16_teacher |
|------|------------|--------------------|
| narrativeqa | TBD | TBD |
| qasper | TBD | TBD |
| multifieldqa_en | TBD | TBD |
| multifieldqa_zh | TBD | TBD |
| hotpotqa | TBD | TBD |
| 2wikimqa | TBD | TBD |
| musique | TBD | TBD |
| dureader | TBD | TBD |
| gov_report | TBD | TBD |
| qmsum | TBD | TBD |
| multi_news | TBD | TBD |
| vcsum | TBD | TBD |
| trec | TBD | TBD |
| triviaqa | TBD | TBD |
| samsum | TBD | TBD |
| lsht | TBD | TBD |
| passage_count | TBD | TBD |
| passage_retrieval_en | TBD | TBD |
| passage_retrieval_zh | TBD | TBD |
| lcc | TBD | TBD |
| repobench-p | TBD | TBD |
| average | TBD | TBD |
<!-- AUTO_LONGBENCH_END -->

---

## 备注

- 之前用于排查后台启动问题的临时测试进程已手动清理，不影响当前正式队列。
- 本记录只负责保存“本轮全量评测安排与运行状态”；最终分数待对应 JSON 生成后再补。
