#!/usr/bin/env python3
"""
Train YOLO Detection Model
==========================

Trains a standard YOLO detection model to find axis-aligned bounding boxes
around photographs in images.

Uses the self-contained data_detection/ dataset — no symlinks needed.

USAGE
-----
    python3 train_detection.py --epochs 100 --batch 16
    python3 train_detection.py --resume
    python3 train_detection.py --data /path/to/dataset_detection.yaml

Author: PhotoScan Import CLI Project
Version: 33 - Separate Data Pipelines
"""

import os
import sys
import argparse
from pathlib import Path

# MPS fallback: torchvision::nms is not implemented for MPS device.
if sys.platform == "darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def get_default_dataset_path():
    """Get default detection dataset path."""
    return Path(__file__).parent / "dataset_detection.yaml"


def train(
    data: str = None,
    epochs: int = 100,
    patience: int = 20,
    batch: int = 16,
    imgsz: int = 640,
    model: str = "yolo26n.pt",
    project: str = "runs/detection",
    name: str = "photo-detector",
    optimizer: str = "auto",
    lr0: float = 0.001,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
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
    pretrained: bool = True,
    close_mosaic: int = 10,
    workers: int = 4,
    device: str = "cpu",
    cache: str = "ram",
    exist_ok: bool = True,
    verbose: bool = True,
    resume: bool = False,
):
    """Train YOLO detection model for photo bounding box detection."""

    if data is None:
        data = str(get_default_dataset_path())

    if not Path(data).exists():
        print(f"Error: Detection dataset not found at {data}")
        print("Run: cd data_generator && python3 generate_detection.py --mode batch")
        sys.exit(1)

    print("=" * 60)
    print("YOLO DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Model:     {model}")
    print(f"Dataset:   {data}")
    print(f"Epochs:    {epochs}")
    print(f"Batch:     {batch}")
    print(f"Image size:{imgsz}")
    print(f"Device:    {device}")
    print(f"Cache:     {cache}")
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
        description="Train YOLO detection model for photo bounding boxes")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolo26n.pt")
    parser.add_argument("--project", type=str, default="runs/detection")
    parser.add_argument("--name", type=str, default="photo-detector")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=str, default="ram")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--degrees", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.5)
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