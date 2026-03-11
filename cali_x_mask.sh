#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash cali_x_mask.sh /PATH/TO/YOUR/MODEL/ [extra cali_x_mask.py args...]
#
# Prereq:
#   - Run `python reorder_indices.py ...` first so ./saved/*_reorder_index_*.pt exists.
#
# Output:
#   - ./saved/${model}_xmask_${dataset}_${metric}_${quant_type}.pt

MODEL="${1:?missing model path}"
shift || true

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${dir}/model/cali_x_mask.py" "${MODEL}" \
  --dataset wikitext2 \
  --act_sort_metric max \
  --quant_type NVFP4 \
  --nsamples 128 \
  --seqlen 2048 \
  --cali_bsz 4 \
  --epochs 1 \
  --lr 1e-3 \
  --x_mask_tau 1.0 \
  --x_mask_alpha 1.0 \
  --x_mask_r_thr -1 \
  --x_mask_token_gate_mode token_all \
  "$@"

