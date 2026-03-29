# Phase 2 实验记录：改进 joint 校准

**目标**：在维持 2:4 N:M 稀疏度的前提下，将 arc_challenge acc_norm 从 0.5068 提升到 ≥ 0.5263（与 bf16 差距 ≤ 1%）

**背景**：
- bf16 baseline: 0.5316
- joint (wikitext2, alpha_reg=0.01): 0.5068 (−2.47% vs bf16)
- 当前差距: −1.95% 需要补回

---

## Phase 1 消融结论（前置实验）

| Config | arc_norm | Δ vs joint | 结论 |
|--------|----------|------------|------|
| joint (full) | 0.5068 | baseline | wikitext2, alpha_reg=0.01 |
| skip28-31 x only | 0.5043 | −0.26% | alpha 未一起 skip → 不一致 |
| skip24-31 x only | 0.5017 | −0.51% | 同上 |
| skip28-31 x+alpha | 0.5043 | −0.26% | layers 28-31 alpha≈1.0，无效 |
| **skip24-31 x+alpha** | **0.5094** | **+0.26%** | 微小改善，非决定性因素 |

**结论**：skip_layers 方向提升有限（最多 +0.26%）。根本问题是 softmax_alpha 过拟合 wikitext2 分布，以及 x_mask gates 在 wikitext2 上校准但泛化到 arc_challenge 不足。

---

## P2-A：C4 校准数据

**假设**：用多样化网页文本 (C4) 替代 Wikipedia 风格 (wikitext2) 做校准，让 gates 和 alpha 学到更通用的激活模式，改善在 arc_challenge (MCQ) 和 rte (NLI) 上的泛化。

**变更**：`--dataset c4`（其余所有超参与原 joint_v1 相同）
**注意**：channel reorder index 仍使用 wikitext2 统计（symlink），仅校准前向 pass 的数据换成 C4。

### 启动命令
```bash
CUDA_VISIBLE_DEVICES=1 python -u model/cali_joint.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --dataset c4 \
    --nsamples 128 --cali_bsz 4 --epochs 30 --lr 5e-3 --gate_cost 0.0001 \
    --x_mask_token_gate_mode token_all \
    --x_mask_token_use_layer_scale \
    --x_mask_token_mlp_shared \
    --trainable_alpha --alpha_lr 1e-3 --alpha_reg 0.01 \
    --exp_name joint_c4
```

| 字段 | 值 |
|------|-----|
| GPU | 1 (PID=1003994) |
| 日志 | `results/logs/p2a_cali_c4_v2.log` |
| 输出 ckpt | `outputs/Llama-3.1-8B/NVFP4/joint_c4_{datetime}/llama-3.1-8b_joint_c4_max_NVFP4.pt` |
| 启动时间 | 2026-03-26 15:33 UTC |
| 状态 | ✅ done |

### 结果
| Task | bf16 | nvfp4 | joint (wiki) | **joint_c4** | Δ vs joint |
|------|------|-------|--------------|-------------|------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5068 | **0.4812** | −2.56% |
| rte | 0.7004 | 0.6859 | 0.6643 | **0.6931** | +2.88% |
| ceval-valid | 0.5245 | 0.4844 | 0.4190 | **0.4421** | +2.31% |

**结论**：C4 校准使 rte 和 ceval 明显恢复（分别 +2.88% 和 +2.31%），但 arc_challenge 进一步下滑（−2.56% vs joint_v1）。C4 数据对 MCQ 无效，反而有害。

---

## P2-B：更强 alpha 正则（alpha_reg=0.1）

**假设**：原 joint 的 alpha_reg=0.01 导致 softmax_alpha 在 wikitext2 上轻微过拟合（实测 alpha 均值仅偏离 1.0 约 0.3%~0.8%，但已造成 rte −4.33%、ceval −2.08%）。加强正则至 0.1（10×）将 alpha 进一步收敛到 1.0 附近，减少负迁移，同时保留对 arc_challenge 有利的稀疏结构。

**变更**：`--alpha_reg 0.1`（其余与原 joint_v1 相同，包括 wikitext2 数据）

### 启动命令
```bash
CUDA_VISIBLE_DEVICES=2 python -u model/cali_joint.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --dataset wikitext2 \
    --nsamples 128 --cali_bsz 4 --epochs 30 --lr 5e-3 --gate_cost 0.0001 \
    --x_mask_token_gate_mode token_all \
    --x_mask_token_use_layer_scale \
    --x_mask_token_mlp_shared \
    --trainable_alpha --alpha_lr 1e-3 --alpha_reg 0.1 \
    --exp_name joint_alphareg01
```

| 字段 | 值 |
|------|-----|
| GPU | 2 (PID=1003995) |
| 日志 | `results/logs/p2b_cali_alphareg01_v2.log` |
| 输出 ckpt | `outputs/Llama-3.1-8B/NVFP4/joint_alphareg01_{datetime}/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt` |
| 启动时间 | 2026-03-26 15:33 UTC |
| 状态 | ✅ done |

### 结果
| Task | bf16 | nvfp4 | joint (reg=0.01) | **joint (reg=0.1)** | Δ vs joint |
|------|------|-------|-----------------|---------------------|------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5068 | **0.4863** | −2.05% |
| rte | 0.7004 | 0.6859 | 0.6643 | **0.6787** | +1.44% |
| ceval-valid | 0.5245 | 0.4844 | 0.4190 | **0.4383** | +1.93% |

**结论**：alpha_reg=0.1 几乎禁止 alpha 偏离 1.0（|α-1|≈0.0004），rte 和 ceval 小幅恢复，但 arc_challenge 仍下滑。说明 alpha 修正对 arc 是有益的，过度压制 alpha 反而无效。

---

## P2-C：解耦训练（wikitext2）

**假设**：joint 同时训练 gates + softmax_alpha 导致 co-adaptation。
- Phase 1（gate-only, 20 epochs）：alpha 冻结在 1.0，gates 单独对 bf16 MSE 最小化
- Phase 2（alpha-only, 15 epochs）：gates 冻结，alpha 单独补偿 gate 引入的 distortion

**变更**：`--decouple_training --gate_epochs 20 --alpha_epochs 15`（其余与 joint_v1 相同）

```bash
CUDA_VISIBLE_DEVICES=0 python -u model/cali_joint.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --dataset wikitext2 \
    --nsamples 128 --cali_bsz 4 --lr 5e-3 --gate_cost 0.0001 \
    --x_mask_token_gate_mode token_all \
    --x_mask_token_use_layer_scale \
    --x_mask_token_mlp_shared \
    --trainable_alpha --alpha_lr 1e-3 --alpha_reg 0.01 \
    --decouple_training --gate_epochs 20 --alpha_epochs 15 \
    --exp_name joint_decouple
```

| 字段 | 值 |
|------|-----|
| GPU | 0 (PID=1063195) |
| 日志 | `results/logs/p2c_cali_decouple.log` |
| 输出 ckpt | `outputs/Llama-3.1-8B/NVFP4/joint_decouple_{datetime}/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt` |
| 启动时间 | 2026-03-27 03:30 UTC |
| 状态 | ✅ done |

### 结果
| Task | bf16 | nvfp4 | joint_v1 | **joint_decouple** | Δ vs joint |
|------|------|-------|----------|-------------------|------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5068 | **0.4906** | −1.62% |
| rte | 0.7004 | 0.6859 | 0.6643 | **0.6570** | −0.73% |
| ceval-valid | 0.5245 | 0.4844 | 0.4190 | **0.4361** | +1.71% |

**补充**：该 ckpt 仅包含 `softmax_alpha`，**不包含 `output_scale`**。

**结论**：解耦训练没有缓解 ARC 退化，反而比 `joint_v1` 更差；但 ceval-valid 有一定恢复，说明 `softmax_alpha` 的第二阶段补偿能修一部分任务分布偏移，无法解决 ARC 的核心问题。

---

## P2-D：任务混合校准数据（arc_mix）

**假设**：用 arc_challenge train + arc_easy train + rte train 混合文本作为 calibration 数据，使 gates 和 `softmax_alpha` 在 MCQ/NLI 任务格式的 activations 上训练，改善对这些 benchmark 的泛化。

**变更**：`--dataset arc_mix`（其余与 joint_v1 相同，30 epochs joint）

关键差异：reorder index 仍用 wikitext2（symlink），只有 calibration forward pass 的输入数据换成 arc_mix。

```bash
CUDA_VISIBLE_DEVICES=1 python -u model/cali_joint.py /data/shichao/models/Llama-3.1-8B \
    --quant_type NVFP4 --no_xw_reorder \
    --dataset arc_mix \
    --nsamples 128 --cali_bsz 4 --lr 5e-3 --gate_cost 0.0001 \
    --x_mask_token_gate_mode token_all \
    --x_mask_token_use_layer_scale \
    --x_mask_token_mlp_shared \
    --trainable_alpha --alpha_lr 1e-3 --alpha_reg 0.01 \
    --epochs 30 \
    --exp_name joint_arc_mix
```

| 字段 | 值 |
|------|-----|
| GPU | 1 (PID=1063373) |
| 日志 | `results/logs/p2d_cali_arc_mix.log` |
| 输出 ckpt | `outputs/Llama-3.1-8B/NVFP4/joint_arc_mix_{datetime}/llama-3.1-8b_joint_arc_mix_max_NVFP4.pt` |
| 启动时间 | 2026-03-27 03:30 UTC |
| 状态 | ✅ done |

### 结果
| Task | bf16 | nvfp4 | joint_v1 | **joint_arc_mix** | Δ vs joint |
|------|------|-------|----------|------------------|------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5068 | **0.4983** | −0.85% |
| rte | 0.7004 | 0.6859 | 0.6643 | **0.6787** | +1.44% |
| ceval-valid | 0.5245 | 0.4844 | 0.4190 | **0.4458** | +2.67% |

**补充**：该 ckpt 仅包含 `softmax_alpha`，**不包含 `output_scale`**。

**结论**：`arc_mix` 对任务格式更匹配，确实提升了 `rte` 和 `ceval-valid`，但 ARC 仍低于 `joint_v1`。这说明“换 calibration 数据”更像是在做任务特化，而不是提升通用的 ARC 鲁棒性。

---

## P2-E：joint_plus（补充实验，加入 output_scale）

**定位**：该实验不属于 `P2-C / P2-D` 主线结果，而是用于单独验证 `output_scale` 是否带来额外补偿能力。

**关键区别**：
- `P2-C joint_decouple` / `P2-D joint_arc_mix` 的 ckpt 只有 `softmax_alpha`
- `joint_plus` 的 ckpt 同时包含 `softmax_alpha + output_scale`

**ckpt 结构证明**：
- `joint_decouple`: `['layers', 'meta', 'softmax_alpha']`
- `joint_arc_mix`: `['layers', 'meta', 'softmax_alpha']`
- `joint_plus`: `['layers', 'meta', 'output_scale', 'softmax_alpha']`

**当前状态**：
- ckpt 已完成：`outputs/Llama-3.1-8B/NVFP4/joint_plus_20260327_040637/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt`
- 评测完成：`results/joint_plus_eval.json`

### 结果
| Task | bf16 | nvfp4 | joint_v1 | **joint_plus** | Δ vs joint_v1 |
|------|------|-------|----------|----------------|----------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5068 | **0.5111** | +0.43% |
| rte | 0.7004 | 0.6859 | 0.6643 | **0.6787** | +1.44% |
| ceval-valid | 0.5245 | 0.4844 | 0.4190 | **0.4510** | +3.19% |

**结论**：`joint_plus` 是目前第一个在 ARC 上超过 `joint_v1` 的补偿版本，说明 `output_scale` 确实提供了额外收益；同时 `rte` 和 `ceval-valid` 也同步提升。虽然 ARC 仍略低于 `nvfp4` (`0.5111 < 0.5145`)，但已经明显优于只用 `softmax_alpha` 的 `P2-C / P2-D`。

---

## 综合对比（P2 全部完成）

| Config | arc_norm | rte | ceval | Δarc vs bf16 | Δarc vs joint_v1 |
|--------|----------|-----|-------|--------------|------------------|
| bf16 | 0.5316 | 0.7004 | 0.5245 | — | — |
| nvfp4 | 0.5145 | 0.6859 | 0.4844 | −1.71% | — |
| xmask | 0.4855 | 0.7076 | 0.4398 | −8.68% | — |
| joint_wiki_reg001 | **0.5068** | 0.6643 | 0.4190 | −2.47% | baseline |
| joint_wiki_reg010 (P2-B) | 0.4863 | 0.6787 | 0.4383 | −4.21% | −2.05% ↓ |
| joint_c4_reg001 (P2-A) | 0.4812 | 0.6931 | 0.4421 | −4.89% | −2.56% ↓ |
| joint_decouple (P2-C) | 0.4906 | 0.6570 | 0.4361 | −4.10% | −1.62% ↓ |
| joint_arc_mix (P2-D) | 0.4983 | **0.6787** | **0.4458** | −3.27% | −0.85% ↓ |
| joint_plus (P2-E) | **0.5111** | **0.6787** | **0.4510** | −1.97% | **+0.43%** |

**目标线**：arc_norm ≥ 0.5263（bf16 −1%）

### P2 结论
- `joint_plus` (加入 `output_scale`) 是当前最好的 `joint` 系配置，首次在 ARC 上超过 `joint_v1`
- `P2-C` 解耦训练没有改善 ARC，说明 co-adaptation 不是主瓶颈
- `P2-D` `arc_mix` 能提升 `rte / ceval-valid`，但无法超过 `joint_v1` 的 ARC
- 问题根源更像是：`x_mask` 2:4 sparsity 天花板 + 当前 teacher/目标函数不匹配；仅靠 `softmax_alpha` 的补偿空间有限
- `joint_plus` 证明 `output_scale` 是有效补偿项，能同时改善 `ARC / RTE / CEval`
- **下一步**：优先转向 `bf16 teacher` / mixed-teacher 这类目标函数修正实验，而不是继续在 `P2` 范围内微调超参

### LongBench（joint vs nvfp4）
| Task | nvfp4 | joint | Δ |
|------|-------|-------|---|
| narrativeqa | 0.1987 | 0.1517 | −4.70% |
| hotpotqa | 0.1079 | 0.1019 | −0.60% |
| gov_report | 0.1643 | 0.1592 | −0.51% |
| triviaqa | 0.1792 | 0.1806 | +0.14% |
| samsum | 0.1665 | 0.1601 | −0.64% |
| passage_retrieval_en | 0.0050 | 0.0000 | −0.50% |
| **avg** | **0.1369** | **0.1256** | **−1.13%** |

joint LongBench avg (0.1256) 低于 nvfp4 (0.1369)，narrativeqa 下降最大 (−4.70%)。
