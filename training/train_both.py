#!/usr/bin/env python3
"""
Train All Models
================

Trains ALL models in the pipeline:
  1. Detection model (yolo26n)
  2. Pose model — single-photo crops (yolo26s-pose)
  3. Pose model — multi-photo scenes (yolo26s-pose)
  4. Fiducial corner model (yolo26n, 4 classes: UL, UR, LL, LR)

USAGE
-----
    python3 train_both.py --epochs 100 --batch 16
    python3 train_both.py --detection-only
    python3 train_both.py --pose-only
    python3 train_both.py --pose-multi-only
    python3 train_both.py --fiducial-only
    python3 train_both.py --device cpu

Author: Photo Pose Detector Project
Version: 36 - Single 4-Class Fiducial Model
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
from train_fiducial import train_fiducial as train_fiducial_single


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
    fiducial_only: bool = False,
    skip_if_exists: bool = True,
):
    """Train all YOLO models."""

    print("\n" + "=" * 70)
    print("YOLO FULL PIPELINE TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {epochs} | Batch: {batch} | Image Size: {imgsz}")
    print(f"Device: {device}")
    print("=" * 70)

    # Phase 1: Detection Model
    if not pose_only and not pose_multi_only and not fiducial_only:
        data_det = Path(__file__).parent / "dataset_detection.yaml"
        if not data_det.exists():
            print(f"\n❌ Detection dataset not found at {data_det}")
            print("Run: cd data_generator && python3 generate_detection.py --mode batch")
            sys.exit(1)

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
    if not detection_only and not pose_multi_only and not fiducial_only:
        data_pose = Path(__file__).parent / "dataset_pose.yaml"
        if not data_pose.exists():
            print(f"\n⚠️  Pose dataset not found at {data_pose}")
            print("Run: cd data_generator && python3 generate_pose.py --mode batch")
        else:
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
    if not detection_only and not pose_only and not fiducial_only:
        data_pose_multi = Path(__file__).parent / "dataset_pose_multi.yaml"
        if not data_pose_multi.exists():
            print(f"\n⚠️  Multi-photo pose dataset not found at {data_pose_multi}")
            print("Run: cd data_generator && python3 generate_pose_multi.py --mode batch")
        else:
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

    # Phase 4: Fiducial Corner Model (single 4-class model)
    if not detection_only and not pose_only and not pose_multi_only:
        data_fiducial = Path(__file__).parent / "dataset_fiducial.yaml"
        if not data_fiducial.exists():
            print(f"\n⚠️  Fiducial dataset not found at {data_fiducial}")
            print("Run: cd data_generator && python3 generate_fiducial.py --mode batch")
        else:
            print("\n" + "-" * 70)
            print("PHASE 4: FIDUCIAL CORNER MODEL TRAINING (4-class)")
            print("-" * 70)

            train_fiducial_single(
                epochs=epochs,
                patience=patience_det,
                batch=batch,
                device=device,
                workers=workers,
                skip_if_exists=skip_if_exists,
            )

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    model_paths = {
        'Detection': Path("runs/detection/photo-detector/weights/best.pt"),
        'Pose (single)': Path("runs/pose/photo-corner-detector/weights/best.pt"),
        'Pose (multi)': Path("runs/pose-multi/photo-corner-detector-multi/weights/best.pt"),
        'Fiducial': Path("runs/fiducial/fiducial-corner/weights/best.pt"),
    }
    for name, path in model_paths.items():
        if path.exists():
            print(f"   ✅ {name:18s}: {path.absolute()}")
        else:
            print(f"   ❌ {name:18s}: not found")

    print(f"\n⏱️  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description="Train all YOLO models in the pipeline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--detection-only", action="store_true")
    parser.add_argument("--pose-only", action="store_true",
                        help="Train only single-photo pose model")
    parser.add_argument("--pose-multi-only", action="store_true",
                        help="Train only multi-photo pose model")
    parser.add_argument("--fiducial-only", action="store_true",
                        help="Train only fiducial corner model")
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
        fiducial_only=args.fiducial_only,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()