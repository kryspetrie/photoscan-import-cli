#!/usr/local/bin/python3
"""
Continue YOLO Fiducial Corner Training (Epochs 50→100)
========================================================
After YOLO completes training, last.pt has epoch=-1 which makes it
non-resumable with resume=True. Instead, we start a NEW training run
from the best weights, keeping all original hyperparameters but
extending epochs to 100 and patience to 30.

This is functionally equivalent to continuing training — the model
weights are initialized from the 50-epoch best, so we don't lose
any learned features. The learning rate schedule restarts from lr0
with a fresh warmup, which is standard practice for extended training.

CRITICAL: This uses all the SAME hyperparameters from the original
training (same dataset, same augmentation, same optimizer settings)
except for:
  - epochs: 50 → 100 (more training time)
  - patience: 15 → 30 (more room for improvement)

USAGE
-----
    nohup /usr/local/bin/python3 continue_fiducial.py > continue_fiducial.log 2>&1 &
    echo "PID: $!"

Author: Photo Pose Detector Project
"""

import os
import sys

# MPS fallback
if sys.platform == "darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
from ultralytics import YOLO

WEIGHTS = Path(__file__).parent / "runs/detect/runs/fiducial/fiducial-corner/weights/best.pt"
DATA = Path(__file__).parent / "dataset_fiducial.yaml"

if not WEIGHTS.exists():
    print(f"ERROR: Weights not found at {WEIGHTS}")
    sys.exit(1)

if not DATA.exists():
    print(f"ERROR: Dataset not found at {DATA}")
    sys.exit(1)

print("=" * 60)
print("CONTINUING YOLO FIDUCIAL CORNER TRAINING (50→100 epochs)")
print("=" * 60)
print(f"Weights:     {WEIGHTS}")
print(f"Dataset:     {DATA}")
print(f"Epochs:      100 (extending from 50)")
print(f"Patience:    30 (extended from 15)")
print(f"Note:        New training run from best 50-epoch weights")
print("=" * 60)

# Load the BEST weights from the completed 50-epoch run
model = YOLO(str(WEIGHTS))

# Start new training with ALL original hyperparameters
# Only change: epochs 50→100, patience 15→30
results = model.train(
    data=str(DATA),
    # Extended training
    epochs=100,
    patience=30,
    # Same model & dataset settings
    imgsz=640,
    batch=32,
    device="cpu",
    cache="disk",
    workers=8,
    # Same project/name so it goes to the same directory
    project="runs/fiducial",
    name="fiducial-corner-extended",
    exist_ok=False,
    # Same optimizer settings
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_bias_lr=0.1,
    warmup_momentum=0.8,
    # Same augmentation (CRITICAL: no flips!)
    mosaic=0.3,
    mixup=0.0,
    copy_paste=0.0,
    cutmix=0.0,
    scale=0.3,
    degrees=5.0,
    translate=0.1,
    fliplr=0.0,        # NO HORIZONTAL FLIP — changes corner orientation
    flipud=0.0,        # NO VERTICAL FLIP — changes corner orientation
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    # Same training settings
    pretrained=True,
    close_mosaic=10,
    deterministic=True,
    seed=0,
    val=True,
    plots=True,
    verbose=True,
    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,
)

print("\n" + "=" * 60)
print("FIDUCIAL CORNER EXTENDED TRAINING COMPLETE")
print("=" * 60)

# Print location of best weights
extended_weights = Path("runs/fiducial/fiducial-corner-extended/weights/best.pt")
if extended_weights.exists():
    print(f"Best model: {extended_weights.absolute()}")