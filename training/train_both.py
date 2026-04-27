#!/usr/bin/env python3
"""
Train Both Models
=================

Trains BOTH YOLO models (Detection and Pose) from the same dataset.

This script provides a convenient way to train both models sequentially,
ensuring they use the same training data and hyperparameters.

USAGE
-----
    # Train both models
    python train_both.py --epochs 100 --batch 16
    
    # Train only detection
    python train_both.py --detection-only
    
    # Train only pose
    python train_both.py --pose-only
    
    # Use CPU for both
    python train_both.py --device cpu

Author: Photo Pose Detector Project
Version: 32 - Two-Model Architecture
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# MPS fallback: torchvision::nms is not implemented for MPS device.
# Setting this env var causes MPS to fall back to CPU for unsupported ops.
# Must be set before importing torch/ultralytics.
if sys.platform == "darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import the individual training functions
from train_detection import train as train_detection
from train_pose import train as train_pose


def train_both(
    epochs: int = 100,
    patience: int = 20,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    workers: int = 8,
    detection_only: bool = False,
    pose_only: bool = False,
    skip_if_exists: bool = True,
):
    """
    Train both YOLO models.
    
    Args:
        epochs: Number of training epochs
        patience: Early stopping patience
        batch: Batch size
        imgsz: Input image size
        device: Device to use
        workers: Number of dataloader workers
        detection_only: Only train detection model
        pose_only: Only train pose model
        skip_if_exists: Skip training if weights already exist
    """
    
    print("\n" + "=" * 70)
    print("YOLO TWO-MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {epochs} | Batch: {batch} | Image Size: {imgsz}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Verify dataset exists
    base_dir = Path("../data")
    if not base_dir.exists():
        print(f"\n❌ Error: Dataset not found at {base_dir}")
        print("Run generate_batch.py first to create training data.")
        sys.exit(1)
    
    # Train Detection Model
    if not pose_only:
        print("\n" + "-" * 70)
        print("PHASE 1: DETECTION MODEL TRAINING")
        print("-" * 70)
        
        # Note: train_detection() will automatically set data/labels -> detection/labels
        # This is required because Ultralytics resolves labels via path substitution
        
        detection_weights = Path("runs/detection/photo-detector/weights/best.pt")
        if skip_if_exists and detection_weights.exists():
            print(f"\n⏭️  Skipping detection training (weights exist)")
            print(f"   {detection_weights}")
        else:
            train_detection(
                epochs=epochs,
                patience=patience,
                batch=batch,
                imgsz=imgsz,
                device=device,
                workers=workers,
            )
    
    # Train Pose Model
    if not detection_only:
        print("\n" + "-" * 70)
        print("PHASE 2: POSE MODEL TRAINING")
        print("-" * 70)
        
        # Note: train_pose() will automatically switch data/labels -> pose/labels
        # The symlink MUST be switched between detection and pose training!
        
        pose_weights = Path("runs/pose/photo-corner-detector/weights/best.pt")
        if skip_if_exists and pose_weights.exists():
            print(f"\n⏭️  Skipping pose training (weights exist)")
            print(f"   {pose_weights}")
        else:
            train_pose(
                epochs=epochs,
                patience=patience,
                batch=batch,
                imgsz=imgsz,
                device=device,
                workers=workers,
            )
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    print("\n📁 Model weights:")
    det_path = Path("runs/detection/photo-detector/weights/best.pt")
    pose_path = Path("runs/pose/photo-corner-detector/weights/best.pt")
    
    if det_path.exists():
        print(f"   Detection: {det_path.absolute()}")
    if pose_path.exists():
        print(f"   Pose:      {pose_path.absolute()}")
    
    print("\n📋 Summary:")
    print("   Detection Model:")
    print("      - Output: Axis-aligned bounding boxes")
    print("      - Use for: Initial detection, filtering")
    print()
    print("   Pose Model:")
    print("      - Output: 4 corner keypoints (LL, UL, UR, LR)")
    print("      - Use for: Precise corner detection, extraction")
    print()
    print("   Pipeline:")
    print("      1. Run detection to find photo regions")
    print("      2. Run pose on detected regions for corners")
    print("      3. Extract photos using corner quadrilateral")
    
    print(f"\n⏱️  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description="Train both YOLO models (Detection and Pose)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device (0, 1, ... or 'cpu')"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--detection-only", action="store_true",
        help="Only train detection model"
    )
    parser.add_argument(
        "--pose-only", action="store_true",
        help="Only train pose model"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force retraining even if weights exist"
    )
    
    args = parser.parse_args()
    
    train_both(
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        detection_only=args.detection_only,
        pose_only=args.pose_only,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()
