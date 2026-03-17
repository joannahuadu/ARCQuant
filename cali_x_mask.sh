#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LOG_DIR="/gemini/code/NMSparsity/ARCQuant/logs"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

nohup env CUDA_VISIBLE_DEVICES=2 python model/cali_x_mask.py \
  "$MODEL_PATH" \
  --dataset wikitext2 \
  --act_sort_metric max \
  --quant_type NVFP4 \
  --nsamples 128 \
  --seqlen 2048 \
  --cali_bsz 2 \
  --epochs 30 \
  --lr 5e-3 \
  --gate_cost 0.0001 \
  --x_mask_tau 1.0 \
  --x_mask_alpha 1.0 \
  --x_mask_r_thr -1 \
  --x_mask_token_gate_mode token_all \
  --no_xw_reorder \
  >> "$LOG_DIR/cali_NVFP4_g1e-4_token_all_lr5e-3_bs2_ep30.log" 2>&1 &

