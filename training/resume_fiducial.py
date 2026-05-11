#!/usr/local/bin/python3
"""
Resume YOLO Fiducial Corner Training
======================================
Resumes fiducial training from the last checkpoint, extending total epochs
from 50 to 100 with increased patience (15 → 30).

CRITICAL: This script uses the correct YOLO resume pattern:
  1. Load model from the checkpoint path (NOT the base model)
  2. Call train(resume=True) — YOLO reads optimizer, epoch, EMA state from checkpoint
  3. Override epochs and patience to continue training beyond the original limit

This ensures training continues from where it left off, NOT from scratch.

USAGE
-----
    # After the initial 50-epoch training completes:
    nohup /usr/local/bin/python3 resume_fiducial.py > resume_fiducial.log 2>&1 &
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

CHECKPOINT = Path(__file__).parent / "runs/detect/runs/fiducial/fiducial-corner/weights/last.pt"

if not CHECKPOINT.exists():
    print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
    print("Cannot resume training without a checkpoint.")
    print("Wait for the current 50-epoch training to complete first.")
    sys.exit(1)

print("=" * 60)
print("RESUMING YOLO FIDUCIAL CORNER TRAINING")
print("=" * 60)
print(f"Checkpoint:  {CHECKPOINT}")
print(f"Epochs:      50 → 100 (extending by 50)")
print(f"Patience:    15 → 30 (more room for improvement)")
print("=" * 60)

# CRITICAL: Load from checkpoint, NOT the base model
model = YOLO(str(CHECKPOINT))

# CRITICAL: resume=True tells YOLO to continue from the checkpoint's epoch
# YOLO will load optimizer state, EMA, scaler, and best_fitness from the checkpoint
# Override epochs to 100 and patience to 30 to allow more training time
results = model.train(
    resume=True,
    epochs=100,
    patience=30,
)

print("\n" + "=" * 60)
print("Fiducial training complete!")
print("=" * 60)