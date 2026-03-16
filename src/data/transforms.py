"""
Audio and spectrogram augmentation transforms for KWS training.

Implements:
  - RandomTimeShift: random circular-free shift on waveform with zero-padding
  - SpecAugment: time and frequency masking on log-mel spectrograms
"""

import random

import torch
import torchaudio.transforms as T


class RandomTimeShift:
    """Randomly shift a waveform in time, padding with zeros."""

    def __init__(self, max_shift_ms: int = 100, sample_rate: int = 16000):
        self.max_shift_samples = int(max_shift_ms / 1000.0 * sample_rate)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-self.max_shift_samples, self.max_shift_samples)
        if shift == 0:
            return waveform

        result = torch.zeros_like(waveform)
        if waveform.dim() == 1:
            if shift > 0:
                result[shift:] = waveform[:-shift]
            else:
                result[:shift] = waveform[-shift:]
        else:
            # (1, N) or (C, N)
            if shift > 0:
                result[..., shift:] = waveform[..., :-shift]
            else:
                result[..., :shift] = waveform[..., -shift:]
        return result


class SpecAugment:
    """Apply SpecAugment (time and frequency masking) to a log-mel spectrogram."""

    def __init__(
        self,
        time_masks: int = 2,
        time_width: int = 8,
        freq_masks: int = 2,
        freq_width: int = 8,
    ):
        self.time_masks = time_masks
        self.freq_masks = freq_masks
        self.time_masking = T.TimeMasking(time_mask_param=time_width)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_width)

    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        # torchaudio masking expects at least 2D (freq, time)
        needs_unsqueeze = log_mel.dim() == 2
        if needs_unsqueeze:
            log_mel = log_mel.unsqueeze(0)  # (1, freq, time)

        for _ in range(self.freq_masks):
            log_mel = self.freq_masking(log_mel)
        for _ in range(self.time_masks):
            log_mel = self.time_masking(log_mel)

        if needs_unsqueeze:
            log_mel = log_mel.squeeze(0)

        return log_mel
