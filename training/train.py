#!/usr/bin/env python3
"""
Photo Pose Detector - YOLO26 Pose Training

Fine-tunes YOLO26n-pose for photo corner detection.

Usage:
    python train.py

Requirements:
    - ultralytics: pip install ultralytics
    - Dataset must be generated first using generate_dataset.py
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
    """Get default dataset path."""
    return Path(__file__).parent / "dataset.yaml"


def train(
    data: str = None,
    epochs: int = 100,
    patience: int = 20,
    batch: int = 16,
    imgsz: int = 640,
    model: str = "yolo26n-pose.pt",
    project: str = "runs/pose",
    name: str = "photo-corner-detector",
    # Optimization
    optimizer: str = "auto",
    lr0: float = 0.001,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    # Augmentation (reduced for corner detection)
    mosaic: float = 0.5,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
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
    device: str = "0",  # GPU device, or 'cpu'
    # Output
    exist_ok: bool = True,
    verbose: bool = True,
):
    """
    Train YOLO26-pose model for photo corner detection.
    
    Args:
        data: Path to dataset YAML file
        epochs: Number of training epochs
        patience: Early stopping patience
        batch: Batch size
        imgsz: Input image size
        model: Pretrained model to use
        project: Project directory
        name: Experiment name
        optimizer: Optimizer (auto, SGD, Adam, AdamW, etc.)
        lr0: Initial learning rate
        lrf: Final learning rate factor
        momentum: SGD momentum
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs
        mosaic: Mosaic augmentation probability
        mixup: Mixup augmentation probability
        copy_paste: Copy-paste augmentation probability
        scale: Scale augmentation range
        degrees: Rotation augmentation range
        translate: Translation augmentation range
        flipud: Vertical flip probability
        fliplr: Horizontal flip probability
        hsv_h: HSV-Hue augmentation
        hsv_s: HSV-Saturation augmentation
        hsv_v: HSV-Value augmentation
        pretrained: Use pretrained weights
        close_mosaic: Disable mosaic in last N epochs
        workers: Number of dataloader workers
        device: Device to use (0, 1, 2, ... or 'cpu')
        exist_ok: Allow overwriting existing experiment
        verbose: Verbose output
    """
    
    # Resolve dataset path
    if data is None:
        data = str(get_default_dataset_path())
    
    if not Path(data).exists():
        print(f"Error: Dataset not found at {data}")
        print("Run generate_dataset.py first to create training data.")
        sys.exit(1)
    
    # Create model
    print(f"Loading model: {model}")
    try:
        yolo_model = YOLO(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"\nYou may need to download the pretrained model.")
        print("Ultralytics will attempt to download it automatically.")
        yolo_model = YOLO("yolo26n-pose.pt")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Dataset: {data}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Device: {device}")
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
        
        # Pose-specific
        pose=True,
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
        description="Train YOLO26-pose model for photo corner detection"
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
        help="Pretrained model"
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
    )


if __name__ == "__main__":
    main()
