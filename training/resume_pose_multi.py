#!/usr/local/bin/python3
"""
Resume YOLO Pose Multi-Photo Training
=======================================
Properly resumes from the last checkpoint using YOLO's built-in resume mechanism.

CRITICAL: This script uses the correct YOLO resume pattern:
  1. Load model from the checkpoint path (NOT the base model)
  2. Call train(resume=True) — YOLO reads optimizer, epoch, EMA state from checkpoint

This ensures training continues from epoch 3/300, NOT from scratch.

Author: Photo Pose Detector Project
"""

import os
import sys

# MPS fallback
if sys.platform == "darwin" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
from ultralytics import YOLO

CHECKPOINT = Path(__file__).parent / "runs/pose/runs/pose-multi/photo-corner-detector-multi/weights/last.pt"

if not CHECKPOINT.exists():
    print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
    print("Cannot resume training without a checkpoint.")
    sys.exit(1)

print("=" * 60)
print("RESUMING YOLO POSE MULTI-PHOTO TRAINING")
print("=" * 60)
print(f"Checkpoint: {CHECKPOINT}")
print(f"Expected: Resuming from epoch 3 of 300")
print("=" * 60)

# CRITICAL: Load from checkpoint, NOT the base model
model = YOLO(str(CHECKPOINT))

# CRITICAL: resume=True tells YOLO to continue from the checkpoint's epoch
# YOLO will load optimizer state, EMA, scaler, and best_fitness from the checkpoint
# It will also restore all training args from the checkpoint's train_args
results = model.train(resume=True)

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)