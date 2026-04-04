#!/usr/bin/env python3
"""
Photo Pose Detector - Validation Script

Validates the trained model on validation dataset.

Usage:
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
    device: str = "0",
    save_json: bool = True,
    save_hybrid: bool = True,
    conf: float = 0.001,
    iou: float = 0.6,
):
    """
    Validate trained model.
    
    Args:
        model: Path to model
        data: Dataset YAML
        batch: Batch size
        imgsz: Image size
        device: Device
        save_json: Save COCO JSON results
        save_hybrid: Save hybrid labels
        conf: Confidence threshold
        iou: IoU threshold for NMS
    """
    
    model_path = Path(model)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    
    if data is None:
        data = Path(__file__).parent / "dataset.yaml"
    
    print(f"Loading model: {model_path}")
    yolo_model = YOLO(str(model_path))
    
    print(f"\nValidating...")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {data}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    
    # Validate
    results = yolo_model.val(
        data=str(data),
        batch=batch,
        imgsz=imgsz,
        device=device,
        save_json=save_json,
        save_hybrid=save_hybrid,
        conf=conf,
        iou=iou,
        plots=True,
    )
    
    # Print metrics
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    if hasattr(results, 'pose'):
        print(f"Pose mAP50: {results.pose.map50:.4f}")
        print(f"Pose mAP50-95: {results.pose.map:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate YOLO26-pose model")
    parser.add_argument("--model", type=str, 
                       default="runs/pose/photo-corner-detector/weights/best.pt")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    
    args = parser.parse_args()
    
    validate(
        model=args.model,
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
