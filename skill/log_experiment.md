# log.py Softmax Stats 实验记录

**Model**: Llama-3.1-8B  
**Task**: arc_challenge (0-shot, bsz=4)  
**Date**: 2026-03-26  
**Goal**: 收集 4 种配置下 softmax attention 分布统计（per-layer, per-head）及激活量化 x_q 零率，分析 2:4 sparsity 对 attention 的影响。

---

## 实验配置

| ID | Config | GPU | PID | Exp Dir | Status |
|----|--------|-----|-----|---------|--------|
| `nvfp4` | NVFP4, no sparsity | 0 | 785044 | `outputs/log_nvfp4_arc_20260326_025853` | ✅ done |
| `joint` | NVFP4 + x_mask + softmax_alpha | 1 | 785045 | `outputs/log_joint_arc_20260326_025853` | ✅ done |
| `xmask` | NVFP4 + x_mask only | 2 | 788321 | `outputs/log_xmask_arc_20260326_030918` | ✅ done |
| `bf16` | bf16 raw (no quant) | 2 | 788320 | `outputs/log_bf16_arc_20260326_030918` | ✅ done |

xmask 和 bf16 在 GPU 2 上串行运行（`&&` 链接，chain PID=788320）。

### 启动命令

```bash
# nvfp4 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -u log.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --tasks arc_challenge --lm_eval_batch_size 4 \
    --softmax_stats --softmax_stats_per_layer --softmax_stats_per_head \
    --exp_dir outputs/log_nvfp4_arc_20260326_025853

# joint (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -u log.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --softmax_alpha_ckpt outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --tasks arc_challenge --lm_eval_batch_size 4 \
    --softmax_stats --softmax_stats_per_layer --softmax_stats_per_head \
    --exp_dir outputs/log_joint_arc_20260326_025853

# xmask → bf16 chain (GPU 2)
CUDA_VISIBLE_DEVICES=2 python -u log.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt outputs/Llama-3.1-8B/NVFP4/baseline_xmask_20260323_084054/llama-3.1-8b_xmask_wikitext2_max_NVFP4.pt \
    --tasks arc_challenge --lm_eval_batch_size 4 \
    --softmax_stats --softmax_stats_per_layer --softmax_stats_per_head \
    --exp_dir outputs/log_xmask_arc_... && \
python -u log.py /data/shichao/models/Llama-3.1-8B \
    --skip_quantize \
    --tasks arc_challenge --lm_eval_batch_size 4 \
    --softmax_stats --softmax_stats_per_layer --softmax_stats_per_head \
    --exp_dir outputs/log_bf16_arc_...
```

### Checkpoints
- xmask: `outputs/Llama-3.1-8B/NVFP4/baseline_xmask_20260323_084054/llama-3.1-8b_xmask_wikitext2_max_NVFP4.pt`
- joint: `outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt`

---

## 代码修复（log.py）

1. **flatquant import** (line 69):  
   `from flatquant.softmax_stats import ...` → `from softmax_stats import ...`

2. **lm_eval 0.4.8 API** (lines 604–615):  
   `initialize_tasks()` + `lm_eval.tasks.ALL_TASKS` → `TaskManager()` + `task_manager.all_tasks`

3. **新增 `--skip_quantize`**:  
   跳过 reorder/quantize，直接运行 bf16 模型采集 softmax stats（不含 x_q stats，bf16 无 QLinearLayer）

---

## 收集的统计量

### Softmax Stats (`SoftmaxStatsScope`, `--softmax_stats_per_layer --softmax_stats_per_head`)
- 全局分布：softmax 值的线性/log10 直方图
- entropy_mean/std：每行 softmax 的信息熵
- top1_lag_mean：top-1 注意力位置的平均偏移（局部性）
- row_max_mean/std：每行最大值分布
- per-layer + per-head：以上统计分 32 层、32 头存储

### Activation Stats (`_log_xq_stats`, 仅量化配置有效)
- 每个 `QLinearLayer` 的量化后激活 `x_q`
- `zero_ratio`：零元素比例
- `two_zeros_ratio`：每组 4 个元素中 ≥2 个为零的比例（2:4 sparsity 满足率）

---

## 监控程序

```bash
python watch_log_experiments.py          # 每 2 分钟刷新
python watch_log_experiments.py --once   # 查看一次
# 后台常驻：PID=789195, 日志 results/logs/watch_log.log
```

---

## nvfp4 结果（✅ 已完成）

### Softmax Stats 全局摘要
| 指标 | 值 |
|------|----|
| n_calls | 32960 |
| entropy_mean | 0.9167 |
| entropy_std | 0.7184 |
| top1_lag_mean | 126.06 |

### x_q 零率统计（224 个模块，layer 0–31）

| Module | zero_ratio | two_zeros_ratio |
|--------|-----------|-----------------|
| layers.0.self_attn.q_proj | 0.0830 | 0.0435 |
| layers.31.mlp.down_proj | 0.2703 | 0.3020 |
| (其余见 `outputs/log_nvfp4_arc_20260326_025853/run.log`) | | |

Act 分布图 (224 PNGs): `outputs/log_nvfp4_arc_20260326_025853/act_dist/`  
Softmax stats: `outputs/log_nvfp4_arc_20260326_025853/softmax_stats.{pt,json}`

---

## joint 结果（✅ 已完成，2026-03-26 03:46 UTC）

**arc_challenge acc_norm**: 0.5068 (vs nvfp4 0.5205, Δ = −1.37%)

### Softmax Stats 全局摘要

| 指标 | nvfp4 | joint | Δ |
|------|-------|-------|---|
| n_calls | 32960 | 32960 | — |
| entropy_mean | 0.9167 | 0.9336 | +0.0169 |
| entropy_norm_mean | 0.2595 | 0.2642 | +0.0047 |
| row_max_mean | 0.7516 | 0.7470 | −0.0046 |
| top1_lag_mean | 126.06 | 126.05 | −0.003 |

结论：softmax_alpha 使 attention 分布轻微变均匀（entropy↑, row_max↓），但效果微小。

### x_q 2:4 满足率（two_zeros_ratio）per-layer 对比

| 层 | nvfp4 | joint | Δ |
|----|-------|-------|---|
| 0 | 0.0926 | 0.6416 | **+0.549** |
| 1 | 0.0793 | 0.5356 | +0.456 |
| 2 | 0.0807 | 0.6166 | **+0.536** |
| 4 | 0.0678 | 0.4306 | +0.363 |
| 8 | 0.0545 | 0.2403 | +0.186 |
| 15 | 0.0603 | 0.1485 | +0.088 |
| 25 | 0.0745 | 0.1510 | +0.077 |
| 30 | 0.0692 | 0.0774 | +0.008 |
| 31 | 0.0844 | 0.0905 | +0.006 |
| **avg** | **0.0679** | **0.2382** | **+0.1703** |

关键发现：
- 早期层（0–4）x_mask gate 大幅提升 2:4 满足率（+45~55%），gates 已有效学习稀疏
- 深层（28–31）改善极小（+1~2%），gates 倾向保留密度（与研究记录的 gate mean 0.85–0.95 一致）
- 全局平均从 6.79% → 23.82%（**+17%**）

Act 分布图 (224 PNGs): `outputs/log_joint_arc_20260326_025853/act_dist/`
Softmax stats: `outputs/log_joint_arc_20260326_025853/softmax_stats.{pt,json}`

## xmask 结果（✅ 已完成，2026-03-26 ~04:00 UTC）

**arc_challenge acc_norm**: 0.4991 (vs nvfp4 0.5205 −2.14%, vs joint 0.5068 −0.77%)

### Softmax Stats 三向对比

| 指标 | nvfp4 | xmask | joint | xmask Δ | joint Δ |
|------|-------|-------|-------|---------|---------|
| entropy_mean | 0.9167 | 0.9373 | 0.9336 | +0.0206 | +0.0169 |
| entropy_norm_mean | 0.2595 | 0.2652 | 0.2642 | +0.0057 | +0.0047 |
| row_max_mean | 0.7516 | 0.7460 | 0.7470 | −0.0056 | −0.0046 |
| top1_lag_mean | 126.06 | 126.05 | 126.05 | −0.003 | −0.003 |

注意：xmask entropy 略高于 joint，softmax_alpha 并非单调增加 attention 均匀度。

### x_q two_zeros_ratio 三向对比（per-layer 平均）

| 层 | nvfp4 | xmask | joint |
|----|-------|-------|-------|
| 0 | 0.0926 | 0.6415 | 0.6416 |
| 1 | 0.0793 | 0.5153 | 0.5356 |
| 2 | 0.0807 | 0.6141 | 0.6166 |
| 4 | 0.0678 | 0.4466 | 0.4306 |
| 8 | 0.0545 | 0.2549 | 0.2403 |
| 15 | 0.0603 | 0.1582 | 0.1485 |
| 25 | 0.0745 | 0.1564 | 0.1510 |
| 30 | 0.0692 | 0.0778 | 0.0774 |
| 31 | 0.0844 | 0.0909 | 0.0905 |
| **avg** | **0.0679** | **0.2446** | **0.2382** |

关键发现：
- xmask 平均 2:4 满足率 24.46%，joint 23.82%（joint 略低）
- 联合优化时 softmax_alpha 分担部分稀疏压力，gates 不需要全力强制稀疏
- 深层（28-31）三种配置均差距极小

### arc_challenge 准确率汇总

| Config | acc_norm | Δ vs nvfp4 | Δ vs bf16 (0.5316) |
|--------|----------|------------|---------------------|
| nvfp4 | 0.5205 | — | −1.11% |
| xmask | 0.4991 | −2.14% | −3.25% |
| joint | 0.5068 | −1.37% | −2.48% |

joint 优于 xmask (+0.77%)，但未达到 bf16 1% 内目标。

Softmax stats: `outputs/log_xmask_arc_20260326_030918/softmax_stats.{pt,json}`

## bf16 结果（✅ 已完成，2026-03-26 ~04:10 UTC）

**arc_challenge acc_norm**: 0.5316 (reference baseline)

### Softmax Stats 全局摘要
| 指标 | bf16 | nvfp4 | xmask | joint |
|------|------|-------|-------|-------|
| entropy_mean | **0.9101** | 0.9167 | 0.9373 | 0.9336 |
| entropy_norm_mean | **0.2574** | 0.2595 | 0.2652 | 0.2642 |
| row_max_mean | **0.7540** | 0.7516 | 0.7460 | 0.7470 |
| top1_lag_mean | 126.098 | 126.058 | 126.053 | 126.055 |
| zero_ratio | 0.4867 | 0.4866 | 0.4866 | 0.4866 |

关键发现：
- **bf16 entropy 最低**（0.910），nvfp4 量化反而略微均匀化 attention（+0.007）
- xmask/joint entropy 更高（0.937/0.934），x_mask gate 强制稀疏导致注意力分布更均匀
- bf16 row_max 最高（0.754），量化和稀疏都导致 peak attention 降低

Softmax stats: `outputs/log_bf16_arc_20260326_030918/softmax_stats.{pt,json}`

---

## 完整四向对比（arc_challenge, bsz=4）

### 准确率

| Config | acc_norm | Δ vs bf16 |
|--------|----------|-----------|
| bf16 | 0.5316 | — |
| nvfp4 | 0.5205 | −1.11% |
| joint | 0.5068 | −2.48% |
| xmask | 0.4991 | −3.25% |

（joint > xmask：softmax_alpha 从 xmask 损失中回收 +0.77%）

### x_q two_zeros_ratio（2:4 N:M 满足率）

| 层 | nvfp4 | xmask | joint |
|----|-------|-------|-------|
| 0 | 0.0926 | 0.6415 | 0.6416 |
| 4 | 0.0678 | 0.4466 | 0.4306 |
| 8 | 0.0545 | 0.2549 | 0.2403 |
| 15 | 0.0603 | 0.1582 | 0.1485 |
| 25 | 0.0745 | 0.1564 | 0.1510 |
| 31 | 0.0844 | 0.0909 | 0.0905 |
| **avg** | **0.0679** | **0.2446** | **0.2382** |

bf16 无 QLinearLayer，无 x_q 统计。

---

## 分析目标

1. **x_q 2:4 满足率对比**：nvfp4 vs xmask vs joint 各层 `two_zeros_ratio`  
   预期：joint >> xmask > nvfp4（gate 学到稀疏）
2. **softmax entropy 对比**：joint 的 softmax_alpha 是否改变 attention 集中度？
3. **top1_lag 对比**：sparsity 是否影响局部注意力偏好？
4. **per-layer 差异**：早期层（0–3）vs 深层（25–31）的稀疏/entropy 变化
