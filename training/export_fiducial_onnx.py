#!/usr/local/bin/python3
"""
Export the fiducial best.pt model to ONNX format.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).parent / "runs/detect/runs/fiducial/fiducial-corner/weights"
best_pt = WEIGHTS_DIR / "best.pt"

if not best_pt.exists():
    print(f"ERROR: {best_pt} not found. Cannot export.")
    exit(1)

print(f"Loading {best_pt} for ONNX export...")
model = YOLO(str(best_pt))

print("Exporting to ONNX...")
result = model.export(format="onnx", simplify=True)
print(f"ONNX export complete: {result}")