#!/usr/bin/env python3
"""
Train YOLO Fiducial Pose Model
================================

Trains a YOLO26s-pose model to detect visible photo boundary segments
(corner fiducials) on 640×640 crops from the detection→pose pipeline.

Each instance is a `photo_segment` with 2 keypoints (segment endpoints).
A single class is used — geometric grouping determines quadrilateral
assignment at inference time.

This model replaces the single-photo pose model for corner detection
when the crop may contain partial or edge-running boundaries:
  - Multiple corners in the image
  - Sides extending the length/height of the image edge
  - One corner or no corners in the image

Augmentation is reduced compared to detection training because:
  - Keypoints must track through augmentations (no mixup/copy_paste)
  - flip_idx=[1, 0] handles horizontal flip (swaps kp0↔kp1)
  - No vertical flip (would change orientation semantics)
  - Moderate mosaic (segments at various positions need context)

USAGE
-----
    python3 train_fiducial_pose.py --epochs 100 --batch 16
    python3 train_fiducial_pose.py --resume
    python3 train_fiducial_pose.py --data /path/to/dataset_fiducial_pose.yaml

Author: Photo Pose Detector Project
Version: 1 - Fiducial Pose Training
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
    """Get default fiducial pose dataset path."""
    return Path(__file__).parent / "dataset_fiducial_pose.yaml"


def train(
    data: str = None,
    epochs: int = 100,
    patience: int = 30,
    batch: int = 16,
    imgsz: int = 640,
    cache: str = "ram",
    model: str = "yolo26s-pose.pt",
    project: str = "runs/pose",
    name: str = "fiducial-pose-segments",
    # Optimization
    optimizer: str = "auto",
    lr0: float = 0.001,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    # Augmentation (reduced for keypoint accuracy)
    mosaic: float = 0.3,       # Moderate — segments need spatial context
    mixup: float = 0.0,        # Disabled — would corrupt keypoint positions
    copy_paste: float = 0.0,   # Disabled — would create invalid segment combos
    scale: float = 0.3,        # Moderate — photos can be at various scales
    degrees: float = 5.0,      # Conservative rotation (doesn't change segment semantics)
    translate: float = 0.1,    # Moderate — segments already at various positions
    flipud: float = 0.0,      # No vertical flip (would swap top/bottom)
    fliplr: float = 0.5,      # Horizontal flip OK with flip_idx=[1,0]
    hsv_h: float = 0.015,
    hsv_s: float = 0.3,
    hsv_v: float = 0.3,
    # Training settings
    pretrained: bool = True,
    close_mosaic: int = 10,
    workers: int = 4,
    device: str = "cpu",
    exist_ok: bool = True,
    verbose: bool = True,
    resume: bool = False,
):
    """Train YOLO-pose model for fiducial segment detection."""

    if data is None:
        data = str(get_default_dataset_path())

    if not Path(data).exists():
        print(f"Error: Fiducial pose dataset not found at {data}")
        print("Run: cd data_generator && python3 generate_fiducial_pose.py --mode batch")
        sys.exit(1)

    print("=" * 60)
    print("YOLO FIDUCIAL POSE MODEL TRAINING")
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
    print("  kp0 = Segment endpoint A")
    print("  kp1 = Segment endpoint B")
    print("  kpt_shape = [2, 3]  (2 keypoints, 3 values: x, y, visibility)")
    print("  flip_idx = [1, 0]   (horizontal flip swaps endpoints)")
    print()
    print("Class Configuration:")
    print("  nc = 1  (single class: photo_segment)")
    print("  Geometric grouping determines quadrilateral assignment")
    print()
    print("Augmentation (reduced for keypoints):")
    print(f"  scale={scale}  translate={translate}  mosaic={mosaic}")
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
        resume=resume,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    best_model = Path(project) / name / "weights" / "best.pt"
    if best_model.exists():
        print(f"Best model: {best_model.absolute()}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO fiducial pose model for segment detection")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolo26s-pose.pt")
    parser.add_argument("--project", type=str, default="runs/pose")
    parser.add_argument("--name", type=str, default="fiducial-pose-segments")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=str, default="ram")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--lr0", type=float, default=0.001)

    args = parser.parse_args()

    train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        model=args.model,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        mosaic=args.mosaic,
        degrees=args.degrees,
        scale=args.scale,
        lr0=args.lr0,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()