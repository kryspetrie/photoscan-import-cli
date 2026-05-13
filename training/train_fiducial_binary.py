#!/usr/local/bin/python3
"""
Train Binary Fiducial Corner Models (4 × YOLO26n)
====================================================

Trains 4 separate binary detection models, one for each corner type:
  UL, UR, LL, LR

Each model learns a simple task: "Is this corner type present in the crop?"
This avoids the 4-class confusion problem that plagued the single model,
where similar L-shaped patterns made classification nearly impossible.

Each model uses:
  - YOLO26n (nano) — fast on CPU
  - Binary (1 class: "corner" vs background)
  - No mosaic (crops already contain diversity) and NO flip augmentation
  - epochs=50, patience=15
  - batch=16, device=cpu, imgsz=640

Models are trained sequentially. Each takes ~30 min on CPU.

USAGE
-----
    # Train all 4 models sequentially
    /usr/local/bin/python3 train_fiducial_binary.py

    # Train a specific corner type
    /usr/local/bin/python3 train_fiducial_binary.py --corner ul

    # Force retrain (overwrite existing weights)
    /usr/local/bin/python3 train_fiducial_binary.py --force

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
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_fiducial_binary"

CORNERS = ['ul', 'ur', 'll', 'lr']

# Default hyperparameters (same as 4-class model where applicable)
DEFAULTS = {
    'epochs': 50,
    'patience': 15,
    'batch': 16,
    'imgsz': 640,
    'model': 'yolo26n.pt',
    'device': 'cpu',
    'workers': 8,
    'cache': 'disk',
}


def train_corner(corner_name, epochs=50, patience=15, batch=16, imgsz=640,
                 model='yolo26n.pt', device='cpu', workers=8, cache='disk',
                 force=False):
    """Train a single binary corner model."""

    data_yaml = DATA_DIR / corner_name / f"dataset_{corner_name}.yaml"
    if not data_yaml.exists():
        print(f"ERROR: Dataset YAML not found at {data_yaml}")
        print(f"Run: /usr/local/bin/python3 split_binary_datasets.py")
        return None

    project = "runs/binary_fiducial"
    name = f"fiducial-{corner_name}"

    # Check if already trained
    weights_path = Path(project) / name / "weights" / "best.pt"
    if not force and weights_path.exists():
        print(f"\n⏭️  Skipping {corner_name.upper()} — weights exist: {weights_path}")
        return weights_path

    print("\n" + "=" * 60)
    print(f"BINARY FIDUCIAL TRAINING — {corner_name.upper()} Corner")
    print("=" * 60)
    print(f"Model:      {model}")
    print(f"Dataset:    {data_yaml}")
    print(f"Classes:    1 (binary: corner vs. background)")
    print(f"Epochs:     {epochs}")
    print(f"Patience:   {patience}")
    print(f"Batch:      {batch}")
    print(f"Device:     {device}")
    print(f"Output:     {project}/{name}/")
    print()
    print("CRITICAL: No flip augmentation (fliplr=0, flipud=0)")
    print("=" * 60)

    yolo = YOLO(model)

    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=name,
        exist_ok=force,
        # Optimization
        optimizer="auto",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Augmentation — same conservative settings as 4-class model
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        cutmix=0.0,
        scale=0.3,
        degrees=5.0,
        translate=0.1,
        fliplr=0.0,        # NO HORIZONTAL FLIP — changes corner orientation
        flipud=0.0,        # NO VERTICAL FLIP — changes corner orientation
        hsv_h=0.015,
        hsv_s=0.3,
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
    print(f"FIDUCIAL {corner_name.upper()} TRAINING COMPLETE")
    print("=" * 60)
    if weights_path.exists():
        print(f"Best model: {weights_path.absolute()}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train binary fiducial corner detection models")
    parser.add_argument("--corner", choices=CORNERS + ['all'], default='all',
                        help="Corner type to train (default: all)")
    parser.add_argument("--epochs", type=int, default=DEFAULTS['epochs'])
    parser.add_argument("--patience", type=int, default=DEFAULTS['patience'])
    parser.add_argument("--batch", type=int, default=DEFAULTS['batch'])
    parser.add_argument("--imgsz", type=int, default=DEFAULTS['imgsz'])
    parser.add_argument("--model", type=str, default=DEFAULTS['model'])
    parser.add_argument("--device", type=str, default=DEFAULTS['device'])
    parser.add_argument("--workers", type=int, default=DEFAULTS['workers'])
    parser.add_argument("--cache", type=str, default=DEFAULTS['cache'])
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if weights exist")
    args = parser.parse_args()

    corners = CORNERS if args.corner == 'all' else [args.corner]

    print("=" * 60)
    print("BINARY FIDUCIAL CORNER TRAINING")
    print("=" * 60)
    print(f"Training {len(corners)} model(s): {', '.join(c.upper() for c in corners)}")
    print(f"Each model: {args.model}, {args.epochs} epochs, batch={args.batch}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    for corner in corners:
        start = datetime.now()
        result = train_corner(
            corner_name=corner,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            model=args.model,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            force=args.force,
        )
        elapsed = (datetime.now() - start).total_seconds() / 60
        print(f"\n  ⏱️  {corner.upper()} took {elapsed:.1f} minutes")
        results[corner] = result

    print("\n" + "=" * 60)
    print("ALL BINARY MODELS TRAINED")
    print("=" * 60)
    for corner in corners:
        weights = Path(f"runs/binary_fiducial/fiducial-{corner}/weights/best.pt")
        status = "✅" if weights.exists() else "❌"
        print(f"  {status} {corner.upper()}: {weights}")


if __name__ == "__main__":
    main()