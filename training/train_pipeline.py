#!/usr/bin/env python3
"""
Train Full Pipeline
===================

Trains all models in the pipeline:
  1. Detection model (yolo26n)
  2. Pose model — single-photo crops (yolo26s-pose)
  3. Binary fiducial corner models ×4 (yolo26n, one per corner type)

USAGE
-----
    python3 train_pipeline.py --epochs 100 --batch 16
    python3 train_pipeline.py --detection-only
    python3 train_pipeline.py --pose-only
    python3 train_pipeline.py --fiducial-only
    python3 train_pipeline.py --device cpu

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

from train_detection import train as train_detection
from train_pose import train as train_pose
from train_fiducial_binary import CORNERS


def train_pipeline(
    epochs: int = 100,
    patience_det: int = 20,
    patience_pose: int = 30,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "cpu",
    workers: int = 4,
    detection_only: bool = False,
    pose_only: bool = False,
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
    if not pose_only and not fiducial_only:
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
    if not detection_only and not fiducial_only:
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

    # Phase 3: Binary Fiducial Corner Models (4 models)
    if not detection_only and not pose_only:
        print("\n" + "-" * 70)
        print("PHASE 3: BINARY FIDUCIAL CORNER MODEL TRAINING (4 models)")
        print("-" * 70)

        from train_fiducial_binary import train_corner
        for corner in CORNERS:
            weights = Path(f"runs/binary_fiducial/fiducial-{corner}/weights/best.pt")
            if skip_if_exists and weights.exists():
                print(f"\n⏭️  Skipping {corner.upper()} training (weights exist)")
                print(f"   {weights}")
            else:
                train_corner(
                    corner_name=corner,
                    epochs=epochs,
                    patience=patience_det,
                    batch=batch,
                    imgsz=imgsz,
                    device=device,
                    workers=workers,
                    force=not skip_if_exists,
                )

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    model_paths = {
        'Detection': Path("runs/detection/photo-detector/weights/best.pt"),
        'Pose (single)': Path("runs/pose/photo-corner-detector/weights/best.pt"),
    }
    for corner in CORNERS:
        model_paths[f'Fiducial-{corner.upper()}'] = Path(f"runs/binary_fiducial/fiducial-{corner}/weights/best.pt")

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
    parser.add_argument("--fiducial-only", action="store_true",
                        help="Train only binary fiducial corner models")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if weights exist")

    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        patience_det=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        detection_only=args.detection_only,
        pose_only=args.pose_only,
        fiducial_only=args.fiducial_only,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()