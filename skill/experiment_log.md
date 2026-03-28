# ARCQuant Joint Sparsity+Alpha Calibration Experiment Log

## Research Goal
NeurIPS paper: Apply 2:4 N:M sparsity on low-bit quantized models (ARCQuant FP4)
with <1% accuracy loss vs bf16 on arc_challenge benchmark.

## Problem Analysis

### Current Two-Step Approach (Sequential)
1. **Step 1 (cali_x_mask)**: Learn per-group gate `r` via layer-wise MSE distillation
   - Teacher: quantized model without x_mask (dense activations)
   - Student: quantized model with x_mask (2:4 sparse)
   - Loss: MSE(student_output, teacher_output)
   - Result: some groups have r~0 (naturally sparse), many have r~1 (resist sparsity)

2. **Step 2 (cali_softmax_alpha)**: Fit per-head alpha to match bf16 entropy
   - Records bf16 model's normalized attention entropy H_norm per layer/head
   - Grid-searches alpha in [0.5, 1.5] to minimize (H_norm_sparse - H_norm_bf16)^2
   - Applied as: query_states *= alpha (scales Q before QK^T)

### Root Cause of Failure
Sequential calibration creates **objective conflict**:
- Gates learned in Step 1 are optimal for alpha=1 (no correction)
- Alpha learned in Step 2 changes attention dynamics, making Step 1 gates suboptimal
- Combined effect can be worse than either alone

### Empirical Evidence (Llama-3.1-8B, arc_challenge)
| Config | vs baseline |
|--------|------------|
| Step 1 (all layers) | baseline |
| Step 1 skip 3,4,5,14,15 | worse |
| Step 1 + Step 2 (all layers) | worse |
| Step 1 skip 3,4,5,14,15 + Step 2 skip 3,4,5,14,15,19-31 | better |

Contradiction: layers 3,4,5,14,15 benefit from sparsity alone but combined
sparsity+alpha hurts them. Deep layers (19-31) alpha is actively harmful.

## Proposed Solution: Joint Calibration

**File**: `model/cali_joint.py`

Key idea: Train gate logits AND softmax_alpha simultaneously per layer.
- Teacher: layer output with x_mask disabled AND alpha=1
- Student: layer output with x_mask enabled AND trainable alpha
- Loss: MSE + alpha_reg * (alpha - 1)^2 + gate_cost + delta_l2
- Both corrections co-adapt during optimization

### Benefits
1. No manual skip_layers needed - optimizer finds optimal alpha per layer
2. Gates and alpha don't conflict since they're jointly optimized
3. If a layer doesn't need alpha, it stays at 1.0 via regularization
4. Single calibration pass instead of two

## Experiment Pipeline

### Step 0: Generate Reorder Indices
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n smoothquant python reorder_indices.py \
    --model meta-llama/Llama-3.1-8B \
    --dataset wikitext2 --act_sort_metric max
```

### Step 1: Baseline x_mask Only
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n smoothquant python model/cali_x_mask.py \
    meta-llama/Llama-3.1-8B --dataset wikitext2 --quant_type NVFP4 \
    --nsamples 128 --epochs 30 --lr 5e-3 --gate_cost 0.0001 \
    --x_mask_token_gate_mode token_all --no_xw_reorder \
    --exp_name baseline_xmask
```

### Step 2: Joint Calibration
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n smoothquant python model/cali_joint.py \
    meta-llama/Llama-3.1-8B --dataset wikitext2 --quant_type NVFP4 \
    --nsamples 128 --epochs 30 --lr 5e-3 --gate_cost 0.0001 \
    --x_mask_token_gate_mode token_all --no_xw_reorder \
    --trainable_alpha --alpha_lr 1e-3 --alpha_reg 0.01 \
    --exp_name joint_v1
```

### Step 3: Evaluate on arc_challenge
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n smoothquant python model/main.py \
    meta-llama/Llama-3.1-8B --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask --x_mask_ckpt <ckpt_path> \
    --softmax_alpha_ckpt <alpha_ckpt_path> \
    --tasks arc_challenge --lm_eval_batch_size 16
```

## Results (Llama-3.1-8B, arc_challenge, 0-shot)

| Config | acc | acc_norm | Δ vs bf16 | Δ vs NVFP4 | Status |
|--------|-----|----------|-----------|------------|--------|
| bf16 reference | 0.5137 | 0.5316 | — | — | done |
| NVFP4 (no sparsity, no_xw_reorder) | 0.4770 | 0.5205 | -1.11% | baseline | done |
| NVFP4 + x_mask (baseline) | 0.4659 | 0.4991 | -3.25% | -2.14% | done |
| NVFP4 + joint (x_mask + alpha) | 0.4787 | 0.5068 | -2.48% | -1.37% | done |

### Completed steps
- [x] Step 0: reorder indices (avg bits = 4.87, saved to `saved/llama-3.1-8b_*_wikitext2_max.pt`)
- [x] Baseline: NVFP4 without sparsity: acc_norm = 0.5205
- [x] Step 1b: x_mask calibration (32/32 layers, ckpt: `outputs/Llama-3.1-8B/NVFP4/baseline_xmask_20260323_084054/llama-3.1-8b_xmask_wikitext2_max_NVFP4.pt`)
- [x] Evaluate x_mask on arc_challenge: acc_norm = 0.4991 (delta = -2.14% vs NVFP4)
- [x] Step 2: Joint calibration (32/32 layers, ckpt: `outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt`)
- [x] Evaluate joint on arc_challenge: acc_norm = 0.5068 (delta = -1.37% vs NVFP4)
- [x] bf16 reference: acc_norm = 0.5316 (via lm_eval direct, no quantization)

### In-progress: Full multi-task evaluation (2026-03-25)
- [ ] Expand evaluation to 9 lm_eval tasks: arc_challenge/easy, boolq, mmlu(5-shot), ceval(5-shot), openbookqa, piqa, rte, winogrande
- [ ] LongBench subset (6 tasks): narrativeqa, hotpotqa, gov_report, triviaqa, samsum, passage_retrieval_en
- Running on GPU 1/2/3; results → `results/{bf16,nvfp4,xmask,joint}_{lm_eval,longbench}.json`
- See full results tracker: `skill/eval_all.md`

### Analysis
- x_mask sparsity alone drops acc_norm by 2.14% (0.5205 → 0.4991)
- **Joint calibration recovers 0.77% of the 2.14% drop** (0.4991 → 0.5068), reducing gap by 36%
- Deep layers (25-31) have high gate means (0.85-0.95), meaning they resist sparsity
- Early layers (0-3) have low gate means (<0.1), meaning they naturally accept sparsity
- Alpha values learned by joint calibration are modest (typically 1.0 ± 0.02), suggesting small but consistent corrections
- Deeper layers show larger alpha deviations (up to 1.032 at layer 26), confirming attention sensitivity to sparsity in deeper layers
- acc (not normalized) actually *improved* with joint calibration (0.4787 vs 0.4770 NVFP4 baseline)

### Notes
- lm_eval_batch_size=4 needed (16 causes OOM on single 4090 with fakequant fallback)
- `initialize_tasks` removed in lm_eval 0.4.8; use `TaskManager` instead
- `--no-banner` flag not supported by this conda version; removed from scripts
- `termcolor` package needed by cali_x_mask.py, installed manually
- SSL to HuggingFace intermittent; datasets may need retry to cache
- **OOM Fix**: `_last_x_mask_*` attrs on x_mask modules hold computation graph refs; must clear before `layer.cpu()`
- **OOM Fix**: `quantize_e2m1` used `torch.bucketize` to avoid 15x tensor expansion
