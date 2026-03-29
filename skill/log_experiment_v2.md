# log.py Softmax Stats 实验记录 V2（修正残差通道统计）

**Model**: Llama-3.1-8B
**Task**: arc_challenge (0-shot, bsz=4)
**Date**: 2026-03-28
**Goal**: 修复 `_log_xq_stats` 排除残差通道后，重新收集 4 种配置下 x_q 零率和 2:4 满足率。

---

## 修复内容

`log.py` `_log_xq_stats`: 之前统计 `zero_ratio` / `two_zeros_ratio` 时包含了 `select_num` 个残差通道。
残差通道不经过 x_mask，始终非稀疏，导致统计偏低。
修复后仅对前 `in_features` 列（`x_q[:, :-select_num]`）计算稀疏统计。

---

## 实验配置

| ID | Config | GPU | PID | Exp Dir | Status |
|----|--------|-----|-----|---------|--------|
| `nvfp4` | NVFP4, no sparsity | 0 | 1373227 | `outputs/log_nvfp4_arc_20260328_030303` | done |
| `joint` | NVFP4 + x_mask + softmax_alpha | 1 | 1373248 | `outputs/log_joint_arc_20260328_030303` | done |
| `xmask` | NVFP4 + x_mask only | 3 | 1373270 | `outputs/log_xmask_arc_20260328_030303` | done |
| `bf16` | bf16 raw (no quant) | 3 | (after xmask) | `outputs/log_bf16_arc_20260328_030303` | done |

xmask 和 bf16 在 GPU 3 上串行运行。

### 启动命令

```bash
# nvfp4 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -u log.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --tasks arc_challenge --lm_eval_batch_size 4 \
    --softmax_stats --softmax_stats_per_layer --softmax_stats_per_head \
    --exp_dir outputs/log_nvfp4_arc_20260328_030303

# joint (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -u log.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --softmax_alpha_ckpt outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --tasks arc_challenge --lm_eval_batch_size 4 \
    --softmax_stats --softmax_stats_per_layer --softmax_stats_per_head \
    --exp_dir outputs/log_joint_arc_20260328_030303

# xmask -> bf16 chain (GPU 3)
CUDA_VISIBLE_DEVICES=3 python -u log.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt outputs/Llama-3.1-8B/NVFP4/baseline_xmask_20260323_084054/llama-3.1-8b_xmask_wikitext2_max_NVFP4.pt \
    --tasks arc_challenge --lm_eval_batch_size 4 \
    --softmax_stats --softmax_stats_per_layer --softmax_stats_per_head \
    --exp_dir outputs/log_xmask_arc_20260328_030303 && \
python -u log.py /data/shichao/models/Llama-3.1-8B \
    --skip_quantize \
    --tasks arc_challenge --lm_eval_batch_size 4 \
    --softmax_stats --softmax_stats_per_layer --softmax_stats_per_head \
    --exp_dir outputs/log_bf16_arc_20260328_030303
```

### Checkpoints
- xmask: `outputs/Llama-3.1-8B/NVFP4/baseline_xmask_20260323_084054/llama-3.1-8b_xmask_wikitext2_max_NVFP4.pt`
- joint: `outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt`

---

## 结果

### arc_challenge 准确率

| Config | acc_norm | Δ vs bf16 | V1 acc_norm | 变化 |
|--------|----------|-----------|-------------|------|
| bf16 | 0.5316 | — | 0.5316 | = |
| nvfp4 | 0.5205 | −1.11% | 0.5205 | = |
| joint | 0.5068 | −2.48% | 0.5068 | = |
| xmask | 0.4991 | −3.25% | 0.4991 | = |

准确率与 V1 完全一致（代码修改仅影响统计日志，不影响推理）。

---

### Softmax Stats 全局摘要（与 V1 一致）

| 指标 | bf16 | nvfp4 | xmask | joint |
|------|------|-------|-------|-------|
| n_calls | 32960 | 32960 | 32960 | 32960 |
| entropy_mean | **0.9101** | 0.9167 | 0.9373 | 0.9336 |
| entropy_norm_mean | **0.2574** | 0.2595 | 0.2652 | 0.2642 |
| row_max_mean | **0.7540** | 0.7516 | 0.7460 | 0.7470 |
| top1_lag_mean | 126.098 | 126.058 | 126.053 | 126.055 |

---

### x_q 统计（V2 修正：排除残差通道）

**关键改进**：V1 统计包含了 `select_num` 个残差通道（不经 x_mask、始终非零），导致 zero_ratio 和 two_zeros_ratio 被稀释。V2 仅统计原始通道。

#### V1 vs V2 全局平均 two_zeros_ratio 对比

| Config | V1 (含残差) | V2 (排除残差) | 差异 |
|--------|------------|--------------|------|
| nvfp4 | 0.0679 | **0.0689** | +0.0010 |
| xmask | 0.2446 | **0.2534** | +0.0088 |
| joint | 0.2382 | **0.2467** | +0.0085 |

nvfp4 变化小（残差通道本身也接近随机零率），xmask/joint 提升约 +0.85%（残差通道的非零值被正确排除）。

#### V2 per-layer two_zeros_ratio（每层 7 模块平均）

| 层 | nvfp4 | xmask | joint | xmask Δnvfp4 | joint Δnvfp4 |
|----|-------|-------|-------|-------------|-------------|
| 0 | 0.0935 | **0.6515** | **0.6514** | +0.558 | +0.558 |
| 1 | 0.0801 | **0.5392** | **0.5606** | +0.459 | +0.481 |
| 2 | 0.0817 | **0.6337** | **0.6361** | +0.552 | +0.554 |
| 3 | 0.0721 | **0.5745** | **0.5521** | +0.502 | +0.480 |
| 4 | 0.0686 | **0.4695** | **0.4527** | +0.401 | +0.384 |
| 8 | 0.0552 | **0.2645** | **0.2494** | +0.209 | +0.194 |
| 15 | 0.0618 | **0.1650** | **0.1548** | +0.103 | +0.093 |
| 20 | 0.0682 | **0.2259** | **0.2208** | +0.158 | +0.153 |
| 25 | 0.0758 | **0.1621** | **0.1566** | +0.086 | +0.081 |
| 28 | 0.0793 | **0.1047** | **0.1029** | +0.025 | +0.024 |
| 30 | 0.0699 | **0.0795** | **0.0789** | +0.010 | +0.009 |
| 31 | 0.0845 | **0.0916** | **0.0911** | +0.007 | +0.007 |
| **avg** | **0.0689** | **0.2534** | **0.2467** | **+0.185** | **+0.178** |

#### V2 per-layer zero_ratio（每层 7 模块平均）

| 层 | nvfp4 | xmask | joint | xmask Δnvfp4 | joint Δnvfp4 |
|----|-------|-------|-------|-------------|-------------|
| 0 | 0.1204 | **0.5023** | **0.5023** | +0.382 | +0.382 |
| 1 | 0.1106 | **0.4111** | **0.4242** | +0.301 | +0.314 |
| 2 | 0.1117 | **0.4705** | **0.4718** | +0.359 | +0.360 |
| 3 | 0.1053 | **0.4329** | **0.4186** | +0.328 | +0.313 |
| 4 | 0.1032 | **0.3689** | **0.3591** | +0.266 | +0.256 |
| 8 | 0.0939 | **0.2444** | **0.2353** | +0.150 | +0.141 |
| 15 | 0.0985 | **0.1811** | **0.1747** | +0.083 | +0.076 |
| 20 | 0.1026 | **0.2168** | **0.2140** | +0.114 | +0.111 |
| 25 | 0.1077 | **0.1757** | **0.1714** | +0.068 | +0.064 |
| 28 | 0.1101 | **0.1309** | **0.1299** | +0.021 | +0.020 |
| 30 | 0.1038 | **0.1128** | **0.1126** | +0.009 | +0.009 |
| 31 | 0.1132 | **0.1197** | **0.1197** | +0.007 | +0.007 |
| **avg** | **0.1030** | **0.2338** | **0.2297** | **+0.131** | **+0.127** |

---

### 关键发现（与 V1 一致，数值更精确）

1. **早期层（0–4）x_mask gate 大幅提升 2:4 满足率**：+38~56%（V1 报告 +36~55%，V2 更高因排除了非稀疏残差通道）
2. **深层（28–31）改善极小**：+0.7~2.5%，gates 倾向保留密度
3. **xmask vs joint**：xmask 全局 two_zeros_ratio 25.34% > joint 24.67%（+0.67%），联合优化时 softmax_alpha 分担部分稀疏压力
4. **nvfp4 基线**：排除残差通道后 two_zeros_ratio 从 6.79% → 6.89%，zero_ratio 从旧值提升到 10.30%
5. **残差通道影响量级**：xmask/joint 的 two_zeros_ratio 提升约 0.85%（从 V1 的 24.5/23.8% → V2 的 25.3/24.7%），说明残差通道对统计的稀释效应虽然存在但不大

---

### 输出文件

| Config | Run Log | Softmax Stats | Act Dist |
|--------|---------|---------------|----------|
| bf16 | `outputs/log_bf16_arc_20260328_030303/run.log` | `.../softmax_stats.{pt,json}` | (无 QLinearLayer) |
| nvfp4 | `outputs/log_nvfp4_arc_20260328_030303/run.log` | `.../softmax_stats.{pt,json}` | `.../act_dist/` |
| xmask | `outputs/log_xmask_arc_20260328_030303/run.log` | `.../softmax_stats.{pt,json}` | `.../act_dist/` |
| joint | `outputs/log_joint_arc_20260328_030303/run.log` | `.../softmax_stats.{pt,json}` | `.../act_dist/` |
