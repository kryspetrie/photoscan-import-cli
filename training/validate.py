#!/usr/bin/env python3
"""
Photo Pose Detector - Validation Script

Validates a trained model on its validation dataset.

Usage:
    python validate.py --model runs/detection/photo-detector/weights/best.pt
    python validate.py --model runs/pose/photo-corner-detector/weights/best.pt
"""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def validate(
    model: str = "runs/pose/photo-corner-detector/weights/best.pt",
    data: str = None,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "cpu",
    save_json: bool = True,
    conf: float = 0.001,
    iou: float = 0.6,
):
    """Validate trained model."""

    model_path = Path(model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Auto-detect dataset from model path
    if data is None:
        if "detection" in str(model_path):
            data = str(Path(__file__).parent / "dataset_detection.yaml")
        else:
            data = str(Path(__file__).parent / "dataset_pose.yaml")

    print(f"Loading model: {model_path}")
    yolo_model = YOLO(str(model_path))

    print(f"\nValidating...")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {data}")
    print(f"  Device: {device}")

    results = yolo_model.val(
        data=data,
        batch=batch,
        imgsz=imgsz,
        device=device,
        save_json=save_json,
        conf=conf,
        iou=iou,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    if hasattr(results, 'pose') and results.pose is not None:
        print(f"\nPose mAP50:    {results.pose.map50:.4f}")
        print(f"Pose mAP50-95: {results.pose.map:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate YOLO model")
    parser.add_argument("--model", type=str,
                        default="runs/pose/photo-corner-detector/weights/best.pt")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    validate(
        model=args.model,
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )