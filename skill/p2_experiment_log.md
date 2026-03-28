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

**假设**：joint 同时训练 gates + alpha 导致 co-adaptation。
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
| 状态 | 🔄 running |

### 结果（待填）
| Task | bf16 | nvfp4 | joint_v1 | **joint_decouple** | Δ vs joint |
|------|------|-------|----------|-------------------|------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5068 | TBD | — |
| rte | 0.7004 | 0.6859 | 0.6643 | TBD | — |
| ceval-valid | 0.5245 | 0.4844 | 0.4190 | TBD | — |

---

## P2-D：任务混合校准数据（arc_mix）

**假设**：用 arc_challenge train + arc_easy train + rte train 混合文本作为 calibration 数据，使 gates 和 alpha 在 MCQ/NLI 任务格式的 activations 上训练，改善对这些 benchmark 的泛化。

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
| 状态 | 🔄 running |

### 结果（待填）
| Task | bf16 | nvfp4 | joint_v1 | **joint_arc_mix** | Δ vs joint |
|------|------|-------|----------|------------------|------------|
| arc_challenge | 0.5316 | 0.5145 | 0.5068 | TBD | — |
| rte | 0.7004 | 0.6859 | 0.6643 | TBD | — |
| ceval-valid | 0.5245 | 0.4844 | 0.4190 | TBD | — |

**状态**：⏳ 等待完成

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

**目标线**：arc_norm ≥ 0.5263（bf16 −1%）

### P2 结论
- joint_v1 (wiki, reg=0.01) 仍是最好的 arc_challenge 配置
- P2 两个方向（换数据、加正则）均未能提升 arc，反而变差
- 问题根源：x_mask 2:4 sparsity 天花板约束，softmax_alpha 微调空间有限
- **下一步**：考虑从架构层面改变，如 P2-C 解耦训练，或放弃 joint 思路，探索其他补偿机制

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
