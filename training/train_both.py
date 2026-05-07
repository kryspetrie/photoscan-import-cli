#!/usr/bin/env python3
"""
Train Both Models
=================

Trains YOLO models from their separate datasets.

By default, trains ALL three models:
  1. Detection model (detection dataset)
  2. Pose model — single-photo crops (data_pose)
  3. Pose model — multi-photo scenes (data_pose_multi)

The two pose models use the same architecture (yolo26s-pose) but different
training data distributions, to determine whether the V1 pose model failure
was caused by configuration bugs or distribution mismatch.

USAGE
-----
    python3 train_both.py --epochs 100 --batch 16
    python3 train_both.py --detection-only
    python3 train_both.py --pose-only            # single-photo only
    python3 train_both.py --pose-multi-only       # multi-photo only
    python3 train_both.py --device cpu

Author: Photo Pose Detector Project
Version: 34 - Dual Pose Training
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# MPS fallback
if sys.platform == "darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from train_detection import train as train_detection
from train_pose import train as train_pose
from train_pose_multi import train as train_pose_multi


def train_both(
    epochs: int = 100,
    patience_det: int = 20,
    patience_pose: int = 30,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "cpu",
    workers: int = 4,
    detection_only: bool = False,
    pose_only: bool = False,
    pose_multi_only: bool = False,
    skip_if_exists: bool = True,
):
    """Train YOLO models."""

    print("\n" + "=" * 70)
    print("YOLO TWO-MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {epochs} | Batch: {batch} | Image Size: {imgsz}")
    print(f"Device: {device}")
    print("=" * 70)

    # Verify datasets exist for requested phases
    data_det = Path(__file__).parent / "dataset_detection.yaml"
    data_pose = Path(__file__).parent / "dataset_pose.yaml"
    data_pose_multi = Path(__file__).parent / "dataset_pose_multi.yaml"

    if not pose_only and not pose_multi_only and not data_det.exists():
        print(f"\n❌ Detection dataset not found at {data_det}")
        print("Run: cd data_generator && python3 generate_detection.py --mode batch")
        sys.exit(1)

    if detection_only and not data_det.exists():
        print(f"\n❌ Detection dataset not found at {data_det}")
        print("Run: cd data_generator && python3 generate_detection.py --mode batch")
        sys.exit(1)

    if pose_only and not data_pose.exists():
        print(f"\n❌ Pose dataset not found at {data_pose}")
        print("Run: cd data_generator && python3 generate_pose.py --mode batch")
        sys.exit(1)

    if pose_multi_only and not data_pose_multi.exists():
        print(f"\n❌ Multi-photo pose dataset not found at {data_pose_multi}")
        print("Run: cd data_generator && python3 generate_pose_multi.py --mode batch")
        sys.exit(1)

    if not pose_only and not pose_multi_only and not data_pose.exists():
        print(f"\n⚠️  Pose dataset not found at {data_pose}")
        print("Run: cd data_generator && python3 generate_pose.py --mode batch")
        print("Skipping single-photo pose training.\n")

    if not pose_only and not pose_multi_only and not data_pose_multi.exists():
        print(f"\n⚠️  Multi-photo pose dataset not found at {data_pose_multi}")
        print("Run: cd data_generator && python3 generate_pose_multi.py --mode batch")
        print("Skipping multi-photo pose training.\n")

    # Phase 1: Detection Model
    if not pose_only and not pose_multi_only:
        print("\n" + "-" * 70)
        print("PHASE 1: DETECTION MODEL TRAINING")
        print("-" * 70)

        detection_weights = Path("runs/detection/photo-detector/weights/best.pt")
        if skip_if_exists and detection_weights.exists():
            print(f"\n⏭️  Skipping detection training (weights exist)")
            print(f"   {detection_weights}")
        else:
            train_detection(
                epochs=epochs,
                patience=patience_det,
                batch=batch,
                imgsz=imgsz,
                device=device,
                workers=workers,
            )

    # Phase 2: Pose Model — Single-Photo Crops
    if not detection_only and not pose_multi_only and data_pose.exists():
        print("\n" + "-" * 70)
        print("PHASE 2: POSE MODEL TRAINING — SINGLE-PHOTO CROPS")
        print("-" * 70)

        pose_weights = Path("runs/pose/photo-corner-detector/weights/best.pt")
        if skip_if_exists and pose_weights.exists():
            print(f"\n⏭️  Skipping single-photo pose training (weights exist)")
            print(f"   {pose_weights}")
        else:
            train_pose(
                epochs=epochs,
                patience=patience_pose,
                batch=batch,
                imgsz=imgsz,
                device=device,
                workers=workers,
            )

    # Phase 3: Pose Model — Multi-Photo Scenes
    if not detection_only and not pose_only and data_pose_multi.exists():
        print("\n" + "-" * 70)
        print("PHASE 3: POSE MODEL TRAINING — MULTI-PHOTO SCENES")
        print("-" * 70)

        pose_multi_weights = Path("runs/pose-multi/photo-corner-detector-multi/weights/best.pt")
        if skip_if_exists and pose_multi_weights.exists():
            print(f"\n⏭️  Skipping multi-photo pose training (weights exist)")
            print(f"   {pose_multi_weights}")
        else:
            train_pose_multi(
                epochs=epochs,
                patience=patience_pose,
                batch=batch,
                imgsz=imgsz,
                device=device,
                workers=workers,
            )

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    det_path = Path("runs/detection/photo-detector/weights/best.pt")
    pose_path = Path("runs/pose/photo-corner-detector/weights/best.pt")
    pose_multi_path = Path("runs/pose-multi/photo-corner-detector-multi/weights/best.pt")

    if det_path.exists():
        print(f"   Detection:       {det_path.absolute()}")
    if pose_path.exists():
        print(f"   Pose (single):   {pose_path.absolute()}")
    if pose_multi_path.exists():
        print(f"   Pose (multi):    {pose_multi_path.absolute()}")

    print(f"\n⏱️  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO models (Detection + both Pose variants)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20,
                        help="Detection patience (pose uses 30)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--detection-only", action="store_true")
    parser.add_argument("--pose-only", action="store_true",
                        help="Train only single-photo pose model")
    parser.add_argument("--pose-multi-only", action="store_true",
                        help="Train only multi-photo pose model")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if weights exist")

    args = parser.parse_args()

    train_both(
        epochs=args.epochs,
        patience_det=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        detection_only=args.detection_only,
        pose_only=args.pose_only,
        pose_multi_only=args.pose_multi_only,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()