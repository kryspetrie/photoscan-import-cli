#!/usr/bin/env python3
"""
Train Fiducial Corner Detection Model (Single 4-Class Model)
=================================================================

Trains a single YOLO detection model with 4 classes (UL, UR, LL, LR)
to find and classify corner orientations in 640×640 crops.

The model learns to both DETECT the corner and CLASSIFY its orientation
from the distinctive L-shaped photo/background boundary pattern:
  - Class 0: UL (┏) — photo extends right + down
  - Class 1: UR (┓) — photo extends left + down
  - Class 2: LL (┗) — photo extends right + up
  - Class 3: LR (┛) — photo extends left + up

Uses yolo26n (nano) for fast inference — each corner is a simple visual
pattern that doesn't need a large model.

CRITICAL: NO flip augmentation! Flipping would change the corner
orientation (UL↔UR, UL↔LL), producing mislabeled training data.

USAGE
-----
    # Train the fiducial model
    python3 train_fiducial.py

    # Train with custom settings
    python3 train_fiducial.py --epochs 50 --batch 16

    # Force retrain
    python3 train_fiducial.py --force

Author: Photo Pose Detector Project
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# MPS fallback
if sys.platform == "darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def get_dataset_path():
    """Get default dataset path."""
    return Path(__file__).parent / "dataset_fiducial.yaml"


def train_fiducial(
    epochs: int = 50,
    patience: int = 15,
    batch: int = 32,
    imgsz: int = 640,
    model: str = "yolo26n.pt",
    cache: str = "disk",
    device: str = "cpu",
    workers: int = 8,
    skip_if_exists: bool = True,
):
    """Train the single 4-class fiducial corner detection model."""

    data = str(get_dataset_path())
    if not Path(data).exists():
        print(f"Error: Dataset YAML not found at {data}")
        print(f"Run: cd data_generator && python3 generate_fiducial.py --mode batch")
        sys.exit(1)

    project = "runs/fiducial"
    name = "fiducial-corner"

    # Check if already trained
    weights_path = Path(project) / name / "weights" / "best.pt"
    if skip_if_exists and weights_path.exists():
        print(f"\n⏭️  Skipping fiducial training — weights exist: {weights_path}")
        return

    print("\n" + "=" * 60)
    print("FIDUCIAL CORNER TRAINING — Single 4-Class Model")
    print("=" * 60)
    print(f"Model:      {model}")
    print(f"Dataset:    {data}")
    print(f"Classes:    4 (UL=0, UR=1, LL=2, LR=3)")
    print(f"Epochs:     {epochs}")
    print(f"Batch:      {batch}")
    print(f"Device:     {device}")
    print(f"Output:     {project}/{name}/")
    print()
    print("CRITICAL: No flip augmentation (fliplr=0, flipud=0)")
    print("  Flipping would change corner orientation!")
    print("=" * 60)

    yolo = YOLO(model)

    results = yolo.train(
        data=data,
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=name,
        exist_ok=True,
        # Optimization
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Augmentation — MODERATE (corner fills a small region of the crop)
        mosaic=0.3,        # Light mosaic (corner is small in crop)
        mixup=0.0,         # No mixup (would corrupt corner shape)
        copy_paste=0.0,    # No copy-paste
        scale=0.3,         # Moderate scale
        degrees=5.0,       # Small rotation (corner orientation matters!)
        translate=0.1,     # Moderate translation
        fliplr=0.0,        # NO HORIZONTAL FLIP — changes corner orientation
        flipud=0.0,        # NO VERTICAL FLIP — changes corner orientation
        hsv_h=0.015,
        hsv_s=0.5,         # Moderate saturation (backgrounds vary)
        hsv_v=0.3,
        # Training settings
        pretrained=True,
        close_mosaic=10,
        workers=workers,
        device=device,
        cache=cache,
        verbose=True,
        val=True,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("FIDUCIAL CORNER TRAINING COMPLETE")
    print("=" * 60)
    if weights_path.exists():
        print(f"Best model: {weights_path.absolute()}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train fiducial corner detection model (4 classes)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=str, default="disk")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if weights exist")

    args = parser.parse_args()

    train_fiducial(
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        model=args.model,
        device=args.device,
        workers=args.workers,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()