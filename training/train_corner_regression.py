#!/usr/bin/env python3
"""
Train YOLO Corner Regression Model
====================================

Trains a lightweight YOLO pose model (yolo26n-pose) to detect photo corners
in tight crops. This model is the refinement stage of the photo detection pipeline:

  detect → pose (approximate) → corner crop → regression head → precise corner

The model uses 1 class (corner) with 1 keypoint (the exact corner position).
Input: 224-480px corner crops (synthetically generated).
Output: bounding box of visible edges + corner keypoint position.

Architecture: yolo26n-pose (3.7M params) — small and fast for this simple task.

V1 Training Configuration:
  - lr0=0.001 with cosine decay (gentler than V3 segment model's 0.002)
  - patience=20, max epochs 80 (stop early if no improvement)
  - kobj=2.0, rle=1.0 (emphasize keypoint precision)
  - box=3.0 (reduced for thin edges with poor IoU)
  - Augmentation: scale=0.5, degrees=15, translate=0.2 (diverse corner appearances)

Usage:
    python3 train_corner_regression.py
    python3 train_corner_regression.py --epochs 100 --batch 32
    python3 train_corner_regression.py --data /path/to/dataset_corner_regression.yaml
"""

import os
import sys
import argparse
from pathlib import Path

# MPS fallback
if sys.platform == "darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def get_default_dataset_path():
    """Get default corner regression dataset path."""
    return Path(__file__).parent / "dataset_corner_regression.yaml"


def train(
    data: str = None,
    epochs: int = 80,
    patience: int = 20,
    batch: int = 32,
    imgsz: int = 320,
    cache: str = "ram",
    model: str = "yolo26n-pose.pt",
    project: str = "runs/pose",
    name: str = "corner-regression-v1",
    # Optimization — gentle lr for precise corner prediction
    optimizer: str = "AdamW",
    lr0: float = 0.001,
    lrf: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    # Loss weights — emphasize keypoint precision over box accuracy
    box: float = 3.0,          # Moderate — thin edge bboxes have poor IoU
    cls: float = 0.3,          # Single class, but still needed for detection
    dfl: float = 1.5,          # Default distribution focal loss
    pose: float = 12.0,         # Key point localization
    kobj: float = 2.0,          # Higher than default (1.0) — emphasize keypoint confidence
    rle: float = 1.0,           # Rotation-equivariant loss for keypoint precision
    # Augmentation — generous since corner crops should be robust to appearance variation
    mosaic: float = 0.0,        # OFF — corner crops are small, mosaic creates artifacts
    mixup: float = 0.0,         # OFF — would corrupt keypoint positions
    copy_paste: float = 0.0,    # OFF — would create invalid corner patterns
    scale: float = 0.5,         # Moderate-high — corners at various scales
    degrees: float = 15.0,      # Moderate rotation — corner angles vary
    translate: float = 0.2,     # Moderate — corner position within crop varies
    flipud: float = 0.0,        # No vertical flip (changes corner semantics)
    fliplr: float = 0.5,        # Horizontal flip OK (corner position mirrors)
    hsv_h: float = 0.015,       # Slight color variation
    hsv_s: float = 0.3,         # Moderate saturation variation
    hsv_v: float = 0.3,         # Moderate brightness variation
    # Training settings
    pretrained: bool = True,
    close_mosaic: int = 0,
    workers: int = 4,
    device: str = "cpu",
    exist_ok: bool = True,
    verbose: bool = True,
    resume: bool = False,
):
    """Train YOLO-pose corner regression model."""

    if data is None:
        data = str(get_default_dataset_path())

    if not Path(data).exists():
        print(f"Error: Corner regression dataset not found at {data}")
        print("Run: cd data_generator && python3 generate_corner_regression.py --mode batch")
        sys.exit(1)

    print("=" * 60)
    print("YOLO CORNER REGRESSION MODEL TRAINING")
    print("=" * 60)
    print(f"Model:      {model}")
    print(f"Dataset:    {data}")
    print(f"Epochs:     {epochs}")
    print(f"Batch:      {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device:     {device}")
    print(f"Cache:      {cache}")
    print()
    print("Keypoint Configuration:")
    print("  kp0 = Corner position (the precise intersection of photo edges)")
    print("  kpt_shape = [1, 3]  (1 keypoint, 3 values: x, y, visibility)")
    print("  flip_idx = [0]      (single keypoint mirrors on horizontal flip)")
    print()
    print("Class Configuration:")
    print("  nc = 1  (single class: corner)")
    print()
    print("Loss Weights (V1 — keypoint-precision emphasis):")
    print(f"  box={box}  cls={cls}  dfl={dfl}  pose={pose}  kobj={kobj}  rle={rle}")
    print()
    print("Augmentation (moderate — corner crops are already varied):")
    print(f"  mosaic={mosaic}  close_mosaic={close_mosaic}")
    print(f"  scale={scale}  translate={translate}")
    print(f"  mixup={mixup}  copy_paste={copy_paste}")
    print(f"  degrees={degrees}  fliplr={fliplr}  flipud={flipud}")
    print("=" * 60)

    yolo_model = YOLO(model)

    results = yolo_model.train(
        data=data,
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=name,
        exist_ok=exist_ok,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        box=box,
        cls=cls,
        dfl=dfl,
        pose=pose,
        kobj=kobj,
        rle=rle,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        scale=scale,
        degrees=degrees,
        translate=translate,
        flipud=flipud,
        fliplr=fliplr,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        pretrained=pretrained,
        close_mosaic=close_mosaic,
        workers=workers,
        device=device,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    best_path = Path(project) / name / "weights" / "best.pt"
    if best_path.exists():
        print(f"Best model: {best_path}")
        print(f"  Size: {best_path.stat().st_size / 1e6:.1f} MB")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO corner regression model")
    parser.add_argument("--data", default=None,
                        help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="corner-regression-v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        model=args.model,
        device=args.device,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()