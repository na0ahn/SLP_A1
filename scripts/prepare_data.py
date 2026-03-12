#!/usr/bin/env python3
"""
Download and prepare Google Speech Commands v2 dataset.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data_dir data/speech_commands_v0.02

This script:
  1. Downloads GSCv2 (~2.3 GB compressed)
  2. Extracts to data/speech_commands_v0.02/
  3. Verifies split files exist
  4. Prints dataset statistics
"""

import sys
import os
import argparse
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DEFAULT_DATA_DIR = "data/speech_commands_v0.02"
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"


def download_with_progress(url: str, dest: str):
    """Download file with progress bar."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 / total_size)
            mb = count * block_size / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  Progress: {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="")
        else:
            mb = count * block_size / 1024 / 1024
            print(f"\r  Downloaded: {mb:.1f} MB", end="")

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def extract_dataset(archive_path: str, data_dir: str):
    """Extract tar.gz archive."""
    print(f"Extracting {archive_path} to {data_dir}...")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print(f"Extraction complete: {data_dir}")


def verify_dataset(data_dir: str) -> bool:
    """Verify dataset integrity."""
    data_dir = Path(data_dir)

    # Check key files/dirs
    required = [
        "validation_list.txt",
        "testing_list.txt",
        "_background_noise_",
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]

    all_ok = True
    for r in required:
        path = data_dir / r
        if path.exists():
            print(f"  ✓ {r}")
        else:
            print(f"  ✗ {r} MISSING")
            all_ok = False

    return all_ok


def print_stats(data_dir: str):
    """Print dataset statistics."""
    data_dir = Path(data_dir)

    print("\nDataset statistics:")
    all_commands = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    ])

    total = 0
    for cmd in all_commands:
        n = len(list((data_dir / cmd).glob("*.wav")))
        total += n
        print(f"  {cmd:15s}: {n:5d} files")

    print(f"\n  {'TOTAL':15s}: {total:5d} files")

    # Split counts
    val_count = sum(1 for _ in open(data_dir / "validation_list.txt"))
    test_count = sum(1 for _ in open(data_dir / "testing_list.txt"))
    print(f"\n  validation_list.txt: {val_count} files")
    print(f"  testing_list.txt:    {test_count} files")
    print(f"  train (approx):      {total - val_count - test_count} files")


def main():
    parser = argparse.ArgumentParser(description="Prepare GSCv2 dataset")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Directory to store dataset")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download (dataset already present)")
    parser.add_argument("--url", default=DATASET_URL,
                        help="Dataset URL")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Check if already exists
    if (data_dir / "validation_list.txt").exists():
        print(f"Dataset already exists at {data_dir}")
        print("Verifying...")
        if verify_dataset(str(data_dir)):
            print_stats(str(data_dir))
            print("\n✓ Dataset ready!")
        return

    if args.skip_download:
        print(f"ERROR: Dataset not found at {data_dir} and --skip_download is set.")
        sys.exit(1)

    # Download
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir.parent / ARCHIVE_NAME

    if not archive_path.exists():
        download_with_progress(args.url, str(archive_path))
    else:
        print(f"Archive already exists: {archive_path}")

    # Extract
    extract_dataset(str(archive_path), str(data_dir))

    # Verify
    print("\nVerifying dataset...")
    ok = verify_dataset(str(data_dir))
    if ok:
        print_stats(str(data_dir))
        print("\n✓ Dataset ready!")

        # Clean up archive
        try:
            archive_path.unlink()
            print(f"Removed archive: {archive_path}")
        except Exception:
            pass
    else:
        print("\n✗ Dataset verification failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
