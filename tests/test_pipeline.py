#!/usr/bin/env python3
"""
Lightweight sanity tests for the KWS pipeline.

Tests:
  1. Log-mel feature shape
  2. Model parameter count
  3. Forward pass output shape
  4. Augmentation output shape
  5. Loss function
  6. Dataset loading (sample check)

Usage:
    python tests/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_logmel_shape():
    """Log-mel output must be (80, 49) for 1s@16kHz."""
    from src.features.logmel import LogMelExtractor
    extractor = LogMelExtractor(n_fft=640)
    waveform = torch.randn(1, 16000)
    log_mel = extractor(waveform)
    assert log_mel.shape == (80, 49), f"Expected (80,49), got {log_mel.shape}"
    print(f"PASS: log_mel shape = {log_mel.shape}")


def test_dscnn_params():
    """DSCNN must have <= 2.5M trainable parameters."""
    from src.models.dscnn import DSCNN
    model = DSCNN()
    n_params = model.count_parameters()
    assert n_params <= 2_500_000, f"{n_params} > 2.5M!"
    print(f"PASS: DSCNN params = {n_params:,}")


def test_bcresnet_params():
    """BCResNet must have <= 2.5M trainable parameters."""
    from src.models.bcresnet import BCResNet
    model = BCResNet()
    n_params = model.count_parameters()
    assert n_params <= 2_500_000, f"{n_params} > 2.5M!"
    print(f"PASS: BCResNet params = {n_params:,}")


def test_dscnn_forward():
    """DSCNN forward pass must output (B, 12)."""
    from src.models.dscnn import DSCNN
    model = DSCNN()
    x = torch.randn(4, 1, 80, 49)
    y = model(x)
    assert y.shape == (4, 12), f"Expected (4,12), got {y.shape}"
    print(f"PASS: DSCNN output shape = {y.shape}")


def test_bcresnet_forward():
    """BCResNet forward pass must output (B, 12)."""
    from src.models.bcresnet import BCResNet
    model = BCResNet()
    x = torch.randn(4, 1, 80, 49)
    y = model(x)
    assert y.shape == (4, 12), f"Expected (4,12), got {y.shape}"
    print(f"PASS: BCResNet output shape = {y.shape}")


def test_time_shift():
    """RandomTimeShift preserves waveform length."""
    from src.data.transforms import RandomTimeShift
    ts = RandomTimeShift(max_shift_ms=100)
    waveform = torch.randn(1, 16000)
    shifted = ts(waveform)
    assert shifted.shape == waveform.shape, f"Shape mismatch: {shifted.shape}"
    print(f"PASS: TimeShift shape = {shifted.shape}")


def test_specaugment():
    """SpecAugment preserves spectrogram shape."""
    from src.data.transforms import SpecAugment
    sa = SpecAugment()
    log_mel = torch.randn(80, 49)
    augmented = sa(log_mel)
    assert augmented.shape == log_mel.shape, f"Shape mismatch: {augmented.shape}"
    print(f"PASS: SpecAugment shape = {augmented.shape}")


def test_loss():
    """Label smoothing loss must be a finite scalar."""
    from src.engine.losses import LabelSmoothingCrossEntropy
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.05)
    logits = torch.randn(8, 12)
    targets = torch.randint(0, 12, (8,))
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0, "Loss must be scalar"
    assert torch.isfinite(loss), "Loss must be finite"
    print(f"PASS: Loss = {loss.item():.4f}")


def test_feature_normalization():
    """Utterance MVN must produce zero-mean, unit-variance features."""
    from src.features.logmel import LogMelExtractor
    extractor = LogMelExtractor(n_fft=640, feature_norm="utterance_mvn")
    waveform = torch.randn(1, 16000) * 0.1 + 0.5  # Non-trivial waveform
    log_mel = extractor(waveform)
    mean = log_mel.mean().item()
    std = log_mel.std().item()
    assert abs(mean) < 0.1, f"Mean should be ~0, got {mean}"
    assert abs(std - 1.0) < 0.1, f"Std should be ~1, got {std}"
    print(f"PASS: MVN mean={mean:.4f}, std={std:.4f}")


def run_all_tests():
    tests = [
        test_logmel_shape,
        test_dscnn_params,
        test_bcresnet_params,
        test_dscnn_forward,
        test_bcresnet_forward,
        test_time_shift,
        test_specaugment,
        test_loss,
        test_feature_normalization,
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 50)
    print("KWS Pipeline Tests")
    print("=" * 50)

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
