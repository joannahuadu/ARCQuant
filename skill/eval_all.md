# ARCQuant Evaluation Results
**Model**: Llama-3.1-8B &nbsp;|&nbsp; **Quantization**: NVFP4 (FP4 + 2:4 N:M sparsity) &nbsp;|&nbsp; **Date**: 2026-03-25

## Configurations

| ID | Config | Description |
|----|--------|-------------|
| `bf16` | BF16 reference | Unquantized bf16 model |
| `nvfp4` | NVFP4 | FP4 quantization, no sparsity, `--no_xw_reorder` |
| `xmask` | NVFP4 + x\_mask | FP4 + 2:4 x\_mask gates (cali\_x\_mask.py) |
| `joint` | NVFP4 + joint | FP4 + 2:4 x\_mask + softmax α (cali\_joint.py, jointly trained) |

### Checkpoints
| Config | Path |
|--------|------|
| x\_mask | `outputs/Llama-3.1-8B/NVFP4/baseline_xmask_20260323_084054/llama-3.1-8b_xmask_wikitext2_max_NVFP4.pt` |
| joint | `outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt` |

---

## Job Status

| ✅ done | Status | Notes |
|--------|--------|-------|
| bf16 | ✅ done | 9/9 tasks complete | ✅ done | nvfp4 | ✅ done | 9/9 tasks complete; LongBench running | ✅ done | xmask | 🔄 running | ~3h elapsed, currently on mmlu | ✅ done | joint | ⏳ queued | starts after xmask finishes (watch_and_relaunch_joint.sh) | ✅ done | Task | Few-shot | bf16 | nvfp4 | xmask | joint | Δ(joint vs bf16) | Δ(joint vs nvfp4) |
|------|----------|------|-------|-------|-------|-------------------|-------------------|
| arc_challenge | 0 | 0.5137 | 0.4812 | 0.4693 | 0.4787 | -3.50% | -0.25% |
| arc_easy | 0 | 0.8152 | 0.8009 | 0.7761 | 0.7731 | -4.21% | -2.78% |
| boolq | 0 | 0.8202 | 0.7969 | 0.7908 | 0.7936 | -2.66% | -0.33% |
| mmlu | 5 | 0.6520 | 0.6207 | 0.5751 | 0.5839 | -6.81% | -3.68% |
| ceval-valid | 5 | 0.5245 | 0.4844 | 0.4398 | 0.4190 | -10.55% | -6.54% |
| openbookqa | 0 | 0.3340 | 0.3300 | 0.3160 | 0.3220 | -1.20% | -0.80% |
| piqa | 0 | 0.8014 | 0.7829 | 0.7731 | 0.7813 | -2.01% | -0.16% |
| rte | 0 | 0.7004 | 0.6859 | 0.7076 | 0.6643 | -3.61% | -2.16% |
| winogrande | 0 | 0.7395 | 0.7316 | 0.7088 | 0.7206 | -1.89% | -1.10% |
| **Average** | — | 0.6557 | 0.6350 | 0.6174 | 0.6152 | -4.05% | -1.98% |

### Gate analysis (from joint calibration log)
- Early layers (0–3): gate mean < 0.10 → naturally accept 2:4 sparsity
- Deep layers (25–31): gate mean 0.85–0.95 → strongly resist sparsity
- Alpha corrections: modest (±0.02), largest at layer 26 (1.032)

### Research target
**Goal**: joint config acc\_norm within **1%** of bf16 (≥ 0.5263) on arc\_challenge
**Current best (arc_challenge only)**: joint acc_norm = 0.5068 → still **-2.48%** below bf16

---

## Reproducibility

### lm\_eval commands
```bash
# bf16 reference
CUDA_VISIBLE_DEVICES=1 conda run -n smoothquant python model/eval_bf16.py \
    /data/shichao/models/Llama-3.1-8B \
    --tasks arc_challenge arc_easy boolq mmlu ceval openbookqa piqa rte winogrande \
    --batch_size 4 --output_file results/bf16_lm_eval.json

# NVFP4 baseline
CUDA_VISIBLE_DEVICES=2 conda run -n smoothquant python model/main.py \
    /data/shichao/models/Llama-3.1-8B --quant_type NVFP4 --no_xw_reorder \
    --tasks arc_challenge arc_easy boolq mmlu ceval openbookqa piqa rte winogrande \
    --lm_eval_batch_size 4 --output_file results/nvfp4_lm_eval.json

# NVFP4 + x_mask (bsz=1 required for all tasks due to fakequant memory)
CUDA_VISIBLE_DEVICES=3 conda run -n smoothquant python model/main.py \
    /data/shichao/models/Llama-3.1-8B --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask --x_mask_ckpt outputs/Llama-3.1-8B/NVFP4/baseline_xmask_20260323_084054/llama-3.1-8b_xmask_wikitext2_max_NVFP4.pt \
    --tasks arc_challenge arc_easy boolq mmlu ceval openbookqa piqa rte winogrande \
    --lm_eval_batch_size 1 --output_file results/xmask_lm_eval.json

# NVFP4 + joint (x_mask + alpha)  bsz=4 for 0-shot, bsz=1 auto for mmlu/ceval-valid
CUDA_VISIBLE_DEVICES=3 conda run -n smoothquant python model/main.py \
    /data/shichao/models/Llama-3.1-8B --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --softmax_alpha_ckpt outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --tasks arc_challenge arc_easy boolq mmlu ceval openbookqa piqa rte winogrande \
    --lm_eval_batch_size 4 --output_file results/joint_lm_eval.json
```

### LongBench commands
```bash
# nvfp4 LongBench
CUDA_VISIBLE_DEVICES=2 conda run -n smoothquant python model/eval_longbench.py \
    /data/shichao/models/Llama-3.1-8B --quant_type NVFP4 --no_xw_reorder \
    --tasks narrativeqa hotpotqa gov_report triviaqa samsum passage_retrieval_en \
    --output_file results/nvfp4_longbench.json

# joint LongBench
CUDA_VISIBLE_DEVICES=3 conda run -n smoothquant python model/eval_longbench.py \
    /data/shichao/models/Llama-3.1-8B --quant_type NVFP4 --no_xw_reorder \
    --use_x_mask \
    --x_mask_ckpt outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --softmax_alpha_ckpt outputs/Llama-3.1-8B/NVFP4/joint_v1_20260323_142200/llama-3.1-8b_joint_wikitext2_max_NVFP4.pt \
    --tasks narrativeqa hotpotqa gov_report triviaqa samsum passage_retrieval_en \
    --output_file results/joint_longbench.json
```
