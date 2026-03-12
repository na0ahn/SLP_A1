#!/bin/bash
# Evaluation script

CHECKPOINT="${1:-outputs/checkpoints/best_model.pt}"

echo "Evaluating: $CHECKPOINT"

python train.py \
    --eval_only \
    --checkpoint "$CHECKPOINT"
