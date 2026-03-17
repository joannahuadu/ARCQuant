#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash cali_softmax_alpha.sh /PATH/TO/YOUR/MODEL/ [extra cali_softmax_alpha.py args...]
#
# Prereq:
#   - Run `python reorder_indices.py ...` first so ./saved/*_reorder_index_*.pt exists.
#   - If you want x-mask behavior consistent with eval, pass:
#       --use_x_mask --x_mask_ckpt /path/to/xmask.pt [--x_mask_eval_hard]
#
# Output:
#   - ./outputs/${model}/${quant_type}/softmax_alpha_${dataset}_${metric}.pt

MODEL="${1:?missing model path}"
shift || true

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${dir}/model/cali_softmax_alpha.py" "${MODEL}" \
  --dataset wikitext2 \
  --act_sort_metric max \
  --quant_type NVFP4 \
  --nsamples 32 \
  --seqlen 2048 \
  --sample_rows 256 \
  --max_rows_per_head 64 \
  --alpha_min 0.5 \
  --alpha_max 1.5 \
  --alpha_step 0.0025 \
  "$@"
