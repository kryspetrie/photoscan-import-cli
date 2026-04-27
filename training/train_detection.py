#!/usr/bin/env python3
"""
Train YOLO Detection Model
==========================

Trains a standard YOLO detection model to find axis-aligned bounding boxes
around photographs in images.

ARCHITECTURE
------------
This is a standard object detection model that outputs rectangular bounding
boxes. It is used as:
1. Initial detection filter in a pipeline
2. Fast approximate localization before pose detection
3. Standalone detector when only rough position is needed

OUTPUT FORMAT
------------
Standard YOLO detection format:
    class_id x_center y_center width height

USAGE
-----
    # Train detection model
    python train_detection.py --epochs 100 --batch 16

    # Resume training
    python train_detection.py --resume

    # Train with custom dataset path
    python train_detection.py --data /path/to/dataset_detection.yaml

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

# Check ultralytics installation
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)

# Label symlink management
from label_links import ensure_labels_symlink, get_data_root_from_yaml


def get_default_dataset_path():
    """Get default detection dataset path."""
    return Path(__file__).parent / "dataset_detection.yaml"


def train(
    data: str = None,
    epochs: int = 100,
    patience: int = 20,
    batch: int = 16,
    imgsz: int = 640,
    model: str = "yolo11n.pt",  # Standard detection model
    project: str = "runs/detection",
    name: str = "photo-detector",
    # Optimization
    optimizer: str = "auto",
    lr0: float = 0.001,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    # Augmentation
    mosaic: float = 0.5,
    mixup: float = 0.1,
    copy_paste: float = 0.1,
    scale: float = 0.5,
    degrees: float = 10.0,
    translate: float = 0.1,
    flipud: float = 0.0,
    fliplr: float = 0.5,
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
    Train YOLO detection model for photo bounding box detection.
    
    Args:
        data: Path to dataset YAML file
        epochs: Number of training epochs
        patience: Early stopping patience
        batch: Batch size
        imgsz: Input image size
        model: Pretrained model to use
        project: Project directory
        name: Experiment name
        resume: Resume from last checkpoint
        ... (other training parameters)
    """
    
    # Resolve dataset path
    if data is None:
        data = str(get_default_dataset_path())
    
    if not Path(data).exists():
        print(f"Error: Detection dataset not found at {data}")
        print("Run generate_batch.py first to create training data.")
        print("Detection labels should be in: data/detection/labels/")
        sys.exit(1)
    
    # CRITICAL: Ensure data/labels symlink points to detection labels
    # Ultralytics resolves labels by replacing /images/ with /labels/ in the
    # image path. Without this symlink, all images are treated as backgrounds
    # and training produces zero losses (the "zero-loss bug").
    try:
        data_root = get_data_root_from_yaml(data)
        ensure_labels_symlink(data_root, "detection")
    except Exception as e:
        print(f"Error: Failed to set up label symlink: {e}")
        print("Detection labels must be accessible at data/labels/train/")
        sys.exit(1)
    
    print("=" * 60)
    print("YOLO DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Dataset: {data}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print("=" * 60)
    
    # Create model
    print(f"\nLoading model: {model}")
    try:
        yolo_model = YOLO(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ultralytics will attempt to download it automatically.")
        yolo_model = YOLO("yolo11n.pt")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Model type: Standard Detection")
    print(f"  Output: Axis-aligned bounding boxes")
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
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train YOLO detection model for photo bounding boxes"
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
        "--model", type=str, default="yolo11n.pt",
        help="Pretrained model (yolo11n.pt, yolo11s.pt, etc.)"
    )
    parser.add_argument(
        "--project", type=str, default="runs/detection",
        help="Project directory"
    )
    parser.add_argument(
        "--name", type=str, default="photo-detector",
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
        "--scale", type=float, default=0.5,
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
