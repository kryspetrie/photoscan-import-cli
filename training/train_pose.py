#!/usr/bin/env python3
"""
Train YOLO Pose Model
=====================

Trains a YOLO pose model to detect 4 corner keypoints (LL, UL, UR, LR)
of photographs in images.

ARCHITECTURE
------------
This is a keypoint detection model that outputs 4 corner coordinates
per detected photo. The corners are detected in a specific order:
- kp0: Lower-Left (LL)  - minimum Y, minimum X
- kp1: Upper-Left (UL)  - maximum Y, minimum X
- kp2: Upper-Right (UR) - maximum Y, maximum X
- kp3: Lower-Right (LR) - minimum Y, maximum X

This order enables:
1. Proper horizontal flip augmentation (flip_idx = [2, 3, 0, 1])
2. Quadrilateral extraction from corner keypoints
3. Perspective correction of detected photos

OUTPUT FORMAT
------------
YOLO-pose format (13 columns per object):
    class_id x_center y_center width height kp0x kp0y kpc0 kp1x kp1y kpc1 kp2x kp2y kpc2 kp3x kp3y kpc3

Where visibility=2 means "visible and within image bounds"

USAGE
-----
    # Train pose model
    python train_pose.py --epochs 100 --batch 16

    # Resume training
    python train_pose.py --resume

    # Train with custom dataset path
    python train_pose.py --data /path/to/dataset_pose.yaml

Author: Photo Pose Detector Project
Version: 32 - Two-Model Architecture
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Check ultralytics installation
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def get_default_dataset_path():
    """Get default pose dataset path."""
    return Path(__file__).parent / "dataset_pose.yaml"


def train(
    data: str = None,
    epochs: int = 100,
    patience: int = 20,
    batch: int = 16,
    imgsz: int = 640,
    model: str = "yolo26n-pose.pt",  # Pose model
    project: str = "runs/pose",
    name: str = "photo-corner-detector",
    # Optimization
    optimizer: str = "auto",
    lr0: float = 0.001,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    # Augmentation (reduced for keypoint detection)
    mosaic: float = 0.5,
    mixup: float = 0.0,  # Disabled for pose
    copy_paste: float = 0.0,  # Disabled for pose
    scale: float = 0.3,  # Reduced scale
    degrees: float = 10.0,
    translate: float = 0.1,
    flipud: float = 0.0,  # No vertical flip (would swap top/bottom)
    fliplr: float = 0.5,  # Horizontal flip OK with flip_idx
    hsv_h: float = 0.015,
    hsv_s: float = 0.3,
    hsv_v: float = 0.3,
    # Training settings
    pretrained: bool = True,
    close_mosaic: int = 10,
    workers: int = 8,
    device: str = "0",
    # Output
    exist_ok: bool = True,
    verbose: bool = True,
    # Resume
    resume: bool = False,
):
    """
    Train YOLO-pose model for photo corner detection.
    
    Args:
        data: Path to dataset YAML file
        epochs: Number of training epochs
        patience: Early stopping patience
        batch: Batch size
        imgsz: Input image size
        model: Pretrained pose model to use (yolo26n-pose.pt)
        project: Project directory
        name: Experiment name
        resume: Resume from last checkpoint
        ... (other training parameters)
    """
    
    # Resolve dataset path
    if data is None:
        data = str(get_default_dataset_path())
    
    if not Path(data).exists():
        print(f"Error: Pose dataset not found at {data}")
        print("Run generate_batch.py first to create training data.")
        print("Pose labels should be in: data/pose/labels/")
        sys.exit(1)
    
    print("=" * 60)
    print("YOLO POSE MODEL TRAINING")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Dataset: {data}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print()
    print("Keypoint Configuration:")
    print("  kp0 = Lower-Left (LL)")
    print("  kp1 = Upper-Left (UL)")
    print("  kp2 = Upper-Right (UR)")
    print("  kp3 = Lower-Right (LR)")
    print("  flip_idx = [2, 3, 0, 1]")
    print("=" * 60)
    
    # Create model
    print(f"\nLoading model: {model}")
    try:
        yolo_model = YOLO(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ultralytics will attempt to download it automatically.")
        yolo_model = YOLO("yolo26n-pose.pt")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Model type: YOLO-Pose (4 keypoints)")
    print(f"  Output: Corner keypoints (LL, UL, UR, LR)")
    print()
    
    results = yolo_model.train(
        # Dataset
        data=data,
        
        # Training parameters
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        
        # Model save
        project=project,
        name=name,
        exist_ok=exist_ok,
        
        # Optimization
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        
        # Augmentation
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
        
        # Training settings
        pretrained=pretrained,
        close_mosaic=close_mosaic,
        workers=workers,
        device=device,
        
        # Output
        verbose=verbose,
        val=True,
        plots=True,
        
        # Pose-specific settings
        pose=True,
        
        # Resume
        resume=resume,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    best_model = Path(project) / name / "weights" / "best.pt"
    last_model = Path(project) / name / "weights" / "last.pt"
    
    if best_model.exists():
        print(f"Best model: {best_model.absolute()}")
    if last_model.exists():
        print(f"Last model: {last_model.absolute()}")
    
    print(f"\nTo export to ONNX:")
    print(f"  python export_onnx.py --model {best_model}")
    
    print(f"\nTo use the model for inference:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{best_model}')")
    print(f"  results = model.predict('image.jpg', save=True)")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train YOLO pose model for photo corner detection"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to dataset YAML file"
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
        "--model", type=str, default="yolo26n-pose.pt",
        help="Pretrained pose model (yolo26n-pose.pt)"
    )
    parser.add_argument(
        "--project", type=str, default="runs/pose",
        help="Project directory"
    )
    parser.add_argument(
        "--name", type=str, default="photo-corner-detector",
        help="Experiment name"
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
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    
    # Augmentation arguments
    parser.add_argument(
        "--mosaic", type=float, default=0.5,
        help="Mosaic augmentation probability (0-1)"
    )
    parser.add_argument(
        "--degrees", type=float, default=10.0,
        help="Rotation augmentation range (degrees)"
    )
    parser.add_argument(
        "--scale", type=float, default=0.3,
        help="Scale augmentation range"
    )
    parser.add_argument(
        "--lr0", type=float, default=0.001,
        help="Initial learning rate"
    )
    
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
        mosaic=args.mosaic,
        degrees=args.degrees,
        scale=args.scale,
        lr0=args.lr0,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
