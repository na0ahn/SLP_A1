# KWS (Keyword Spotting) - Google Speech Commands v2

**Assignment 1:** Lightweight Keyword Spotting on GSCv2 (12-class)

---

## Project Overview

12-class keyword spotting system using Google Speech Commands Dataset v2.

- **10 target keywords:** yes, no, up, down, left, right, on, off, stop, go
- **2 special classes:** silence, unknown
- **Model:** DS-CNN (Depthwise Separable CNN), 617,100 parameters
- **Test accuracy: 96.36%** (target: 95%)

---

## Environment Setup

### Requirements
- Python 3.9+
- CUDA GPU (recommended)

### Install Dependencies

```bash
pip install torch torchaudio numpy matplotlib seaborn pandas scikit-learn tqdm wandb PyYAML soundfile
```

---

## Data Preparation

```bash
python scripts/prepare_data.py
```

Or manually download:

```bash
mkdir -p data
wget "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz" \
     -O data/speech_commands_v0.02.tar.gz
mkdir -p data/speech_commands_v0.02
tar -xzf data/speech_commands_v0.02.tar.gz -C data/speech_commands_v0.02
```

### Data Split

| Split | Samples | Ratio |
|-------|---------|-------|
| Train | 36,923  | ~8    |
| Valid | 4,445   | ~1    |
| Test  | 4,890   | ~1    |

---

## Training

```bash
# Full training (40 epochs, DS-CNN)
python train.py

# With W&B online tracking
export WANDB_API_KEY="your_api_key"
python train.py --wandb_mode online

# Quick sanity check (2 epochs)
python train.py --sanity
```

### Training Options

```bash
python train.py --config configs/final.yaml   # specify config
python train.py --epochs 60                    # override epochs
python train.py --model bcresnet               # use BC-ResNet instead
python train.py --batch_size 128               # override batch size
python train.py --lr 5e-4                      # override learning rate
```

---

## Reproduce Evaluation Results

The best model checkpoint is saved at `outputs/checkpoints/best_model.pt`.

### Step 1: Ensure dataset is prepared

```bash
ls data/speech_commands_v0.02/yes/  # should list .wav files
```

### Step 2: Run evaluation on the test set

```bash
# Evaluate using the best checkpoint (default path)
python train.py --eval_only

# Or specify the checkpoint path explicitly
python train.py --eval_only --checkpoint outputs/checkpoints/best_model.pt
```

### Expected Output

```
[Eval] Test accuracy: 0.9636 (96.36%)
```

### Checkpoint Details

| Item | Value |
|------|-------|
| File | `outputs/checkpoints/best_model.pt` |
| Size | ~7.2 MB |
| Best epoch | 39 (of 40) |
| Val accuracy | 96.13% |
| Test accuracy | 96.36% |
| Parameters | 617,100 |

The checkpoint contains: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `cfg`, `class_names`, `n_params`.

---

## Final Results

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 96.13% (epoch 39) |
| **Test Accuracy** | **96.36%** |
| Parameters | 617,100 / 2,500,000 |

### Per-class Test Accuracy

| Class | Accuracy | Class | Accuracy |
|-------|----------|-------|----------|
| yes | 98.33% | on | 93.69% |
| no | 95.56% | off | 94.28% |
| up | 97.18% | stop | 99.03% |
| down | 94.33% | go | 96.27% |
| left | 99.03% | silence | 100.0% |
| right | 97.22% | unknown | 91.18% |

---

## Configuration Summary

### Feature Extraction

| Parameter | Value |
|-----------|-------|
| Input | Log-Mel Spectrogram |
| Window | 40ms (640 samples) |
| Hop | 20ms (320 samples) |
| Mel bins | 80 |
| Output shape | (80, 49) |
| Normalization | Utterance-wise MVN |

### Model: DS-CNN

| Parameter | Value |
|-----------|-------|
| Architecture | Depthwise Separable CNN + Residual |
| Channels | [64, 128, 256, 256, 512] |
| Dropout | 0.2 |
| Parameters | 617,100 |

### Training

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine + Warmup (5 epochs) |
| Epochs | 40 |
| Batch size | 256 |
| Gradient clipping | 5.0 |
| Label smoothing | 0.05 |

### Augmentation & Regularization

| Type | Method |
|------|--------|
| Augmentation | RandomTimeShift (±100ms) |
| Augmentation | SpecAugment (2 time masks w=8, 2 freq masks w=8) |
| Regularization | Dropout (p=0.2) |
| Regularization | Weight Decay (1e-4) |
| Regularization | Label Smoothing (0.05) |

---

## Project Structure

```
SLP_A1/
├── README.md
├── train.py                     # Main entry point
├── configs/
│   ├── default.yaml
│   └── final.yaml               # Training configuration
├── scripts/
│   └── prepare_data.py          # Data download script
├── src/
│   ├── data/
│   │   ├── dataset.py           # GSCv2 dataset & dataloaders
│   │   └── transforms.py        # RandomTimeShift, SpecAugment
│   ├── features/
│   │   ├── logmel.py            # Log-Mel feature extractor
│   │   └── visualize.py         # Visualization utilities
│   ├── models/
│   │   ├── dscnn.py             # DS-CNN model
│   │   ├── bcresnet.py          # BC-ResNet model
│   │   └── model.py             # Model factory
│   ├── engine/
│   │   ├── train.py             # Training loop
│   │   ├── evaluate.py          # Evaluation loop
│   │   ├── losses.py            # Loss functions
│   │   ├── scheduler.py         # LR scheduler
│   │   ├── metrics.py           # Metrics
│   │   └── utils.py             # Utilities
│   ├── tracking/
│   │   └── wandb_logger.py      # W&B logger
│   └── report/
│       └── export_assets.py     # Report asset generation
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pt        # Best model checkpoint
│   ├── logs/
│   │   └── training_history.json
│   └── summaries/
│       └── final_results.json
└── data/
    └── speech_commands_v0.02/   # Dataset (download separately)
```

---

## W&B Experiment Tracking

```bash
# Online mode
export WANDB_API_KEY="your_api_key"
python train.py --wandb_mode online

# Sync offline logs later
wandb sync outputs/wandb/
```

Tracked metrics: train loss, train/val accuracy, learning rate, gradient norm, epoch time.

---

*SKKU Spoken Language Processing - Assignment 1 (2026)*
