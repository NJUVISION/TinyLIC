#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-../Kodak}
CKPT_ROOT=${CKPT_ROOT:-ckpts}
ID=${ID:-4}
EXTRA_ARGS=${EXTRA_ARGS:-}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=. python3 scripts/test.py \
    --architecture tinylic_10k \
    --dataset "$DATASET" \
    --path "$CKPT_ROOT/tinylic_10k/${ID}.pth.tar" \
    $EXTRA_ARGS

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=. python3 scripts/test.py \
    --architecture tinylic_20k \
    --dataset "$DATASET" \
    --path "$CKPT_ROOT/tinylic_20k/${ID}.pth.tar" \
    $EXTRA_ARGS

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH=. python3 scripts/test.py \
    --architecture tinylic_50k \
    --dataset "$DATASET" \
    --path "$CKPT_ROOT/tinylic_50k/${ID}.pth.tar" \
    $EXTRA_ARGS
