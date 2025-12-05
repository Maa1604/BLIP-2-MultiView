#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Train BLIP-2 or Region-BLIP-2 on PadChest-GR
# ============================================================

PYTHON=${PYTHON:-python}

# Experiment name (change as needed)
EXP_NAME="blip2_mimic_cxr_vqa_6"    # e.g. "blip2_base_padchestgr"

# Training hyperparameters
EPOCHS=30
BATCH_SIZE=4
ACCUM=32
LR=3e-4
NUM_BEAMS=2
MAX_NEW_TOKENS=128
SAVE_EVERY=0

# Optional: select GPU
export CUDA_VISIBLE_DEVICES=0

${PYTHON} train_blip2_mimic_cxr_vqa.py \
  --exp_name "${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --accumulate_grad_batches "${ACCUM}" \
  --lr "${LR}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --save_every "${SAVE_EVERY}"
