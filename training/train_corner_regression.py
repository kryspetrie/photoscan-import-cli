#!/usr/bin/env python3
"""
Train YOLO Corner Regression Model V2
=======================================

Trains a lightweight YOLO pose model (yolo26n-pose) to detect photo corners
in 320×320 crops. This model is the refinement stage of the photo detection pipeline:

  detect → pose (approximate) → corner crop → regression head → precise corner

The model uses 1 class (corner) with 1 keypoint (the exact corner position).
Input: 320×320 corner crops (synthetically generated).
Output: bounding box of visible edges + corner keypoint position.

V2 Changes (from V1 analysis — best mAP50-95=0.75, 59% recall):
  - Data fixes: fixed 320×320 image size, min bbox ≥ 32px, kp offset enforced
  - 25% negative/background samples (was 0% — model couldn't reject non-corners)
  - 10K train / 2K val data (was 5K/1K)

V2 Training Changes:
  - lr0=0.002 (was 0.001 — V1 was too slow to converge, matching V3 pose model)
  - patience=30 (was 20 — give more time given larger dataset)
  - Removed degenerate kp==center cases, larger bboxes → box weight restored
  - box=4.0 (was 3.0 — bboxes are now more meaningful with min 32px)

Architecture: yolo26n-pose (~3.7M params) — small and fast for this simple task.

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
    epochs: int = 100,
    patience: int = 30,
    batch: int = 32,
    imgsz: int = 320,
    cache: str = "ram",
    model: str = "yolo26n-pose.pt",
    project: str = "runs/pose",
    name: str = "corner-regression-v2",
    # Optimization — match V3 pose model's proven lr
    optimizer: str = "AdamW",
    lr0: float = 0.002,
    lrf: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    # Loss weights — V2 rebalanced for larger bboxes and better detection
    box: float = 4.0,          # V2: was 3.0 — bboxes now 32px+ min, more meaningful
    cls: float = 0.3,          # Single class, but needed for foreground/background
    dfl: float = 1.5,          # Default distribution focal loss
    pose: float = 12.0,        # Keypoint localization
    kobj: float = 2.0,         # Emphasize keypoint confidence
    rle: float = 1.0,          # Rotation-equivariant loss for keypoint precision
    # Augmentation — moderate, matching V3 pose model approach
    mosaic: float = 0.0,       # OFF — corner crops are small, mosaic creates artifacts
    mixup: float = 0.0,        # OFF — would corrupt keypoint positions
    copy_paste: float = 0.0,   # OFF — would create invalid corner patterns
    scale: float = 0.3,        # V2: was 0.5 — reduced to keep bboxes meaningful
    degrees: float = 5.0,       # V2: was 15.0 — reduced, corner semantics change with rotation
    translate: float = 0.1,    # V2: was 0.2 — reduced, corner position is key signal
    flipud: float = 0.0,      # No vertical flip (changes corner semantics)
    fliplr: float = 0.5,      # Horizontal flip OK (corner position mirrors)
    hsv_h: float = 0.015,     # Slight color variation
    hsv_s: float = 0.3,       # Moderate saturation variation
    hsv_v: float = 0.3,       # Moderate brightness variation
    # Training settings
    pretrained: bool = True,
    close_mosaic: int = 0,
    workers: int = 4,
    device: str = "cpu",
    exist_ok: bool = True,
    verbose: bool = True,
    resume: bool = False,
):
    """Train YOLO-pose corner regression model (V2)."""

    if data is None:
        data = str(get_default_dataset_path())

    if not Path(data).exists():
        print(f"Error: Corner regression dataset not found at {data}")
        print("Run: cd data_generator && python3 generate_corner_regression.py --mode batch")
        sys.exit(1)

    print("=" * 60)
    print("YOLO CORNER REGRESSION MODEL TRAINING — V2")
    print("=" * 60)
    print(f"Model:      {model}")
    print(f"Dataset:    {data}")
    print(f"Epochs:     {epochs}")
    print(f"Patience:   {patience}")
    print(f"Batch:      {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device:     {device}")
    print(f"Cache:      {cache}")
    print()
    print("V2 Changes from V1:")
    print("  Data: fixed 320×320, min_bbox=32px, kp offset enforced, 25% negatives")
    print("  lr0: 0.002 (was 0.001 — too slow)")
    print("  box: 4.0 (was 3.0 — bboxes now more meaningful)")
    print("  scale: 0.3 (was 0.5 — too aggressive for small bboxes)")
    print("  degrees: 5.0 (was 15.0 — corner semantics change with rotation)")
    print("  translate: 0.1 (was 0.2 — position is key signal)")
    print("  patience: 30 (was 20 — larger dataset needs more time)")
    print()
    print("Keypoint Configuration:")
    print("  kp0 = Corner position (the precise intersection of photo edges)")
    print("  kpt_shape = [1, 3]  (1 keypoint, 3 values: x, y, visibility)")
    print("  flip_idx = [0]      (single keypoint mirrors on horizontal flip)")
    print()
    print("Class Configuration:")
    print("  nc = 1  (single class: corner)")
    print()
    print("Loss Weights (V2 — rebalanced for larger bboxes):")
    print(f"  box={box}  cls={cls}  dfl={dfl}  pose={pose}  kobj={kobj}  rle={rle}")
    print()
    print("Augmentation (conservative — preserve corner position):")
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
        cache=cache,
        verbose=verbose,
        val=True,
        plots=True,
        seed=0,
        deterministic=True,
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
        description="Train YOLO corner regression model (V2)")
    parser.add_argument("--data", default=None,
                        help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache", default="ram")
    parser.add_argument("--name", default="corner-regression-v2")
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
        cache=args.cache,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()