"""
Google Speech Commands v2 dataset for 12-class keyword spotting.

Classes: yes, no, up, down, left, right, on, off, stop, go, silence, unknown
"""

import os
import json
import math
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchaudio
try:
    torchaudio.set_audio_backend("soundfile")
except RuntimeError:
    pass
from torch.utils.data import Dataset, DataLoader

from src.features.logmel import make_logmel_extractor
from src.data.transforms import RandomTimeShift, SpecAugment

TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
ALL_LABELS = TARGET_WORDS + ["silence", "unknown"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(ALL_LABELS)}


def load_and_preprocess(path: str, target_length: int = 16000) -> torch.Tensor:
    """Load a wav file and pad/trim to target_length samples."""
    waveform, sr = torchaudio.load(path)
    waveform = waveform[0]  # mono: (N,)

    if waveform.shape[0] < target_length:
        waveform = F.pad(waveform, (0, target_length - waveform.shape[0]))
    elif waveform.shape[0] > target_length:
        waveform = waveform[:target_length]

    return waveform  # (target_length,)


def _read_file_list(path: str) -> set:
    """Read a validation/testing list file into a set of relative paths."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def build_splits(root: str, target_words: list, seed: int = 42):
    """
    Build train/val/test item lists from GSCv2 directory structure.

    Unknown and silence classes are subsampled to match the average
    per-target-word count, yielding ~36923:4445:4890 splits.

    Returns:
        (train_items, val_items, test_items) — each a list of dicts with
        keys: path, label, class_idx, is_noise
    """
    root = str(root)
    rng = random.Random(seed)
    val_set = _read_file_list(os.path.join(root, "validation_list.txt"))
    test_set = _read_file_list(os.path.join(root, "testing_list.txt"))

    target_set = set(target_words)
    train_target, val_target, test_target = [], [], []
    train_unknown, val_unknown, test_unknown = [], [], []

    # Walk all word directories (excluding _background_noise_)
    for word_dir in sorted(os.listdir(root)):
        word_path = os.path.join(root, word_dir)
        if not os.path.isdir(word_path) or word_dir.startswith("_"):
            continue

        if word_dir in target_set:
            label = word_dir
        else:
            label = "unknown"

        class_idx = LABEL_TO_IDX[label]

        for fname in sorted(os.listdir(word_path)):
            if not fname.endswith(".wav"):
                continue
            rel_path = f"{word_dir}/{fname}"
            full_path = os.path.join(root, rel_path)
            item = {
                "path": full_path,
                "label": label,
                "class_idx": class_idx,
                "is_noise": False,
            }

            if label != "unknown":
                if rel_path in val_set:
                    val_target.append(item)
                elif rel_path in test_set:
                    test_target.append(item)
                else:
                    train_target.append(item)
            else:
                if rel_path in val_set:
                    val_unknown.append(item)
                elif rel_path in test_set:
                    test_unknown.append(item)
                else:
                    train_unknown.append(item)

    # Compute average per-target-word count for each split
    n_target_words = len(target_words)  # 10
    avg_train = math.ceil(len(train_target) / n_target_words)
    avg_val = math.ceil(len(val_target) / n_target_words)
    avg_test = math.ceil(len(test_target) / n_target_words)

    # Subsample unknown to match average per-word count
    rng.shuffle(train_unknown)
    rng.shuffle(val_unknown)
    rng.shuffle(test_unknown)
    train_unknown = train_unknown[:avg_train]
    val_unknown = val_unknown[:avg_val]
    test_unknown = test_unknown[:avg_test]

    # Combine target + subsampled unknown
    train_items = train_target + train_unknown
    val_items = val_target + val_unknown
    test_items = test_target + test_unknown

    # Add silence samples from _background_noise_
    # Each noise file is split into 3 time-axis blocks (train/val/test)
    # with a 1s buffer between blocks to prevent near-boundary leakage.
    # All splits see all 6 noise files (diversity), but never share time regions.
    noise_dir = os.path.join(root, "_background_noise_")
    noise_files = sorted([
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith(".wav")
    ])

    target_length = 16000  # 1s at 16kHz
    chunk_stride = 16000   # non-overlapping 1s chunks
    buffer = 16000         # 1s buffer between split regions

    # For each noise file, divide time axis into train(80%) / val(10%) / test(10%)
    # with buffer zones between regions.
    train_chunks, val_chunks, test_chunks = [], [], []

    for noise_path in noise_files:
        info = torchaudio.info(noise_path)
        total = info.num_frames

        # Region boundaries (with buffer gaps)
        train_end = int(total * 0.8)
        val_start = train_end + buffer
        val_end = val_start + int(total * 0.1)
        test_start = val_end + buffer
        test_end = total

        for region_start, region_end, chunk_list in [
            (0, train_end, train_chunks),
            (val_start, val_end, val_chunks),
            (test_start, test_end, test_chunks),
        ]:
            pos = region_start
            while pos + target_length <= region_end:
                chunk_list.append((noise_path, pos))
                pos += chunk_stride

    rng.shuffle(train_chunks)
    rng.shuffle(val_chunks)
    rng.shuffle(test_chunks)

    silence_idx = LABEL_TO_IDX["silence"]

    for n_samples, item_list, chunks in [
        (avg_train, train_items, train_chunks),
        (avg_val, val_items, val_chunks),
        (avg_test, test_items, test_chunks),
    ]:
        for i in range(n_samples):
            path, start = chunks[i % len(chunks)]
            item_list.append({
                "path": path,
                "label": "silence",
                "class_idx": silence_idx,
                "is_noise": True,
                "start": start,
            })

    return train_items, val_items, test_items


class SpeechCommandsDataset(Dataset):
    """GSCv2 dataset returning (log_mel, class_idx) pairs."""

    def __init__(self, items: list, cfg: dict, is_train: bool = False):
        self.items = items
        self.cfg = cfg
        self.is_train = is_train
        self.sample_rate = cfg["data"]["sample_rate"]
        self.target_length = int(cfg["data"]["duration_sec"] * self.sample_rate)

        self.extractor = make_logmel_extractor(cfg)

        # Pre-load background noise waveforms for silence class
        noise_dir = os.path.join(cfg["data"]["root"], "_background_noise_")
        self._noise_waveforms = {}
        if os.path.isdir(noise_dir):
            for f in os.listdir(noise_dir):
                if f.endswith(".wav"):
                    fpath = os.path.join(noise_dir, f)
                    wv, _ = torchaudio.load(fpath)
                    self._noise_waveforms[fpath] = wv[0]  # mono

        # Augmentations (training only)
        self.time_shift = None
        self.spec_augment = None
        if is_train:
            aug_cfg = cfg.get("augmentation", {})
            shift_ms = aug_cfg.get("random_time_shift_ms", 0)
            if shift_ms > 0:
                self.time_shift = RandomTimeShift(
                    max_shift_ms=shift_ms, sample_rate=self.sample_rate
                )
            if aug_cfg.get("specaugment_enabled", False):
                self.spec_augment = SpecAugment(
                    time_masks=aug_cfg.get("specaugment_time_masks", 2),
                    time_width=aug_cfg.get("specaugment_time_width", 8),
                    freq_masks=aug_cfg.get("specaugment_freq_masks", 2),
                    freq_width=aug_cfg.get("specaugment_freq_width", 8),
                )

    def __len__(self):
        return len(self.items)

    def _load_noise_chunk(self, item: dict) -> torch.Tensor:
        """Extract a fixed 1-second chunk from a background noise file.

        All items have a 'start' key set during build_splits for determinism.
        Train diversity comes from augmentation (TimeShift, SpecAugment).
        """
        waveform = self._noise_waveforms.get(item["path"])
        if waveform is None:
            waveform, _ = torchaudio.load(item["path"])
            waveform = waveform[0]

        start = item["start"]
        chunk = waveform[start : start + self.target_length]
        if chunk.shape[0] < self.target_length:
            chunk = F.pad(chunk, (0, self.target_length - chunk.shape[0]))
        return chunk

    def __getitem__(self, idx):
        item = self.items[idx]

        if item["is_noise"]:
            waveform = self._load_noise_chunk(item)
        else:
            waveform = load_and_preprocess(item["path"], self.target_length)

        # Waveform augmentation
        if self.is_train and self.time_shift is not None:
            waveform = self.time_shift(waveform)

        # Extract log-mel features
        log_mel = self.extractor(waveform)  # (80, 49)

        # Spectrogram augmentation
        if self.is_train and self.spec_augment is not None:
            log_mel = self.spec_augment(log_mel)

        # Add channel dim for CNN input: (1, 80, 49)
        log_mel = log_mel.unsqueeze(0)

        return log_mel, item["class_idx"]


def _worker_init_fn(worker_id):
    """Ensure each DataLoader worker uses a different random seed."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)


def make_dataloaders(cfg: dict, seed: int = 42):
    """
    Build train/val/test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    train_items, val_items, test_items = build_splits(
        cfg["data"]["root"], cfg["data"]["target_words"], seed=seed
    )

    print(f"[Data] Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")

    train_ds = SpeechCommandsDataset(train_items, cfg, is_train=True)
    val_ds = SpeechCommandsDataset(val_items, cfg, is_train=False)
    test_ds = SpeechCommandsDataset(test_items, cfg, is_train=False)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=g,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, ALL_LABELS


def save_split_summary(train_items, val_items, test_items, out_dir: str):
    """Save per-class distribution of each split to JSON."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def count_classes(items):
        counts = defaultdict(int)
        for item in items:
            counts[item["label"]] += 1
        return dict(counts)

    train_counts = count_classes(train_items)
    val_counts = count_classes(val_items)
    test_counts = count_classes(test_items)

    summary = {
        "classes": ALL_LABELS,
        "total": {
            "train": len(train_items),
            "valid": len(val_items),
            "test": len(test_items),
        },
        "per_class": {
            "train": {c: train_counts.get(c, 0) for c in ALL_LABELS},
            "valid": {c: val_counts.get(c, 0) for c in ALL_LABELS},
            "test": {c: test_counts.get(c, 0) for c in ALL_LABELS},
        },
    }

    out_path = os.path.join(out_dir, "data_split_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Data] Split summary saved to {out_path}")
    print(f"[Data]   Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")
