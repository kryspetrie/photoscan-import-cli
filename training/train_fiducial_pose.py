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
  - Mosaic is DISABLED — synthetic data already contains multi-photo
    grid scenes with varied backgrounds; adding mosaic creates false
    segment boundaries at composite seams.

V2 Changes (from V1 analysis — FAILED):
  - cls=0.0 (was 0.5) — REMOVED foreground/background discrimination;
    mAP50 stuck at ~0.24 for 21 epochs, no improvement from epoch 1.
  - rle=0.3 (was 1.0) — Over-reduced; box loss filled the void at 41%,
    worst for thin-segment detection.
  - lr0=0.001 (auto selected 0.002 for V1) — Too slow; combined with
    above issues, model never converged on detection.

V3 Changes (from V1+V2 analysis):
  - mosaic=0.0 (was 0.3) — synthetic data has built-in compositing (KEPT)
  - close_mosaic=0 (was 10) — no mosaic to phase out (KEPT)
  - optimizer=AdamW (was auto) — explicit control, no lr0 override (KEPT)
  - lr0=0.002 — match V1's auto-selected rate, proven convergence
  - cls=0.3 (was 0.0 in V2, 0.5 in V1) — reduced but NOT zero; still
    needed for foreground/background discrimination with nc=1
  - rle=0.5 (was 0.3 in V2, 1.0 in V1) — moderate reduction from V1's
    47% dominance, but not as aggressive as V2's 0.3
  - box=4.0 (was 7.5) — thin segment bounding boxes have inherently poor
    IoU; reducing box weight prevents box loss from dominating

USAGE
-----
    python3 train_fiducial_pose.py --epochs 100 --batch 16
    python3 train_fiducial_pose.py --resume
    python3 train_fiducial_pose.py --data /path/to/dataset_fiducial_pose.yaml

Author: PhotoScan Import CLI Project
Version: 3 - Loss rebalanced (box reduced, cls restored), mosaic off, lr0 restored
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
    name: str = "fiducial-pose-v3",
    # Optimization (V3: AdamW with V1-proven lr0=0.002)
    optimizer: str = "AdamW",
    lr0: float = 0.002,
    lrf: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    # Loss weights (V3: box reduced for thin segments, cls restored for fg/bg, rle moderated)
    box: float = 4.0,          # V3: was 7.5 — thin segment bboxes have poor IoU
    cls: float = 0.3,          # V3: was 0.0 in V2 (broke detection), 0.5 in V1
    dfl: float = 1.5,
    pose: float = 12.0,
    kobj: float = 1.0,
    rle: float = 0.5,          # V3: was 0.3 in V2 (over-reduced), 1.0 in V1 (47%)
    # Augmentation (V3: mosaic OFF — kept from V2, synthetic data has compositing)
    mosaic: float = 0.0,       # OFF — data already has grid/multi-photo scenes
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
    close_mosaic: int = 0,    # V3: no mosaic to phase out
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
    print("Loss Weights (V3 rebalanced — box reduced, cls restored, rle moderated):")
    print(f"  box={box}  cls={cls}  dfl={dfl}  pose={pose}  kobj={kobj}  rle={rle}")
    print()
    print("Augmentation (mosaic OFF — synthetic data has composites):")
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
    parser.add_argument("--name", type=str, default="fiducial-pose-v3")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=str, default="ram")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--cls", type=float, default=0.3)
    parser.add_argument("--rle", type=float, default=0.5)
    parser.add_argument("--box", type=float, default=4.0)

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
        optimizer=args.optimizer,
        lr0=args.lr0,
        cls=args.cls,
        rle=args.rle,
        box=args.box,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        degrees=args.degrees,
        scale=args.scale,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()