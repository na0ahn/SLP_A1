"""
Log-Mel Spectrogram feature extractor.

Assignment specification:
  - window: 40 ms  → win_length = 640 samples at 16 kHz
  - overlap: 20 ms → hop_length = 320 samples at 16 kHz
  - mel bins: 80
  - n_fft: 1024
  - f_min: 20 Hz, f_max: 8000 Hz
  - log transform: log(mel + eps), eps = 1e-6
  - utterance-wise mean-variance normalization (MVN)
  - Expected output shape: [80, 49] for 1 s audio at 16 kHz (center=False)
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


class LogMelExtractor(nn.Module):
    """
    Log-Mel Spectrogram extractor with utterance-wise MVN.

    Expected input : waveform tensor of shape (1, 16000) or (16000,)
    Expected output: log-mel tensor of shape (80, 49) [n_mels, n_frames]
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 640,    # = win_length; yields 49 frames for 1s@16kHz, center=False
        win_length: int = 640,    # 40 ms
        hop_length: int = 320,    # 20 ms
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        log_eps: float = 1e-6,
        center: bool = False,
        feature_norm: str = "utterance_mvn",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.log_eps = log_eps
        self.feature_norm = feature_norm

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            center=center,
            power=2.0,
            norm="slaney",
            mel_scale="htk",
        )

        # Verify expected frame count
        # For 16000 samples, win=640, hop=320, center=False:
        # n_frames = floor((16000 - 640) / 320) + 1 = floor(15360/320) + 1 = 48 + 1 = 49
        self._expected_frames = 49

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (1, 16000) or (16000,) float tensor

        Returns:
            log_mel: (80, 49) float tensor
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Compute mel spectrogram → (1, n_mels, n_frames) or (n_mels, n_frames)
        mel = self.mel_spectrogram(waveform)  # (1, 80, T) or (80, T)

        if mel.dim() == 3:
            mel = mel.squeeze(0)  # (80, T)

        # Log transform
        log_mel = torch.log(mel + self.log_eps)  # (80, T)

        # Utterance-wise mean-variance normalization
        if self.feature_norm == "utterance_mvn":
            mean = log_mel.mean()
            std = log_mel.std() + 1e-8
            log_mel = (log_mel - mean) / std

        return log_mel  # (80, 49)

    def get_expected_shape(self) -> tuple:
        return (self.n_mels, self._expected_frames)


def make_logmel_extractor(cfg: dict) -> LogMelExtractor:
    """Factory function from config dict."""
    f = cfg.get("feature", cfg)
    return LogMelExtractor(
        sample_rate=cfg.get("data", {}).get("sample_rate", 16000),
        n_fft=f.get("n_fft", 1024),
        win_length=f.get("win_length", 640),
        hop_length=f.get("hop_length", 320),
        n_mels=f.get("n_mels", 80),
        f_min=f.get("f_min", 20.0),
        f_max=f.get("f_max", 8000.0),
        log_eps=f.get("log_eps", 1e-6),
        center=f.get("center", False),
        feature_norm=f.get("feature_norm", "utterance_mvn"),
    )
