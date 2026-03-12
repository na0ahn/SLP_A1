#!/bin/bash
# Full training script for KWS model

set -e

echo "================================"
echo "KWS Training - GSCv2 12-class"
echo "================================"

# Default config
CONFIG="${1:-configs/final.yaml}"
EPOCHS="${2:-40}"

echo "Config: $CONFIG"
echo "Epochs: $EPOCHS"
echo ""

# Run training
python train.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --wandb_mode offline

echo ""
echo "Training complete!"
echo "Best model: outputs/checkpoints/best_model.pt"
echo "Report assets: report_assets/"
