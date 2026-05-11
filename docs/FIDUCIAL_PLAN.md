# Fiducial Corner Refinement Plan
==============================

## Overview

The pose model produces good initial corner estimates, but for high-quality
photo extraction we need sub-pixel corner accuracy. This plan introduces a
**single fiducial corner detection model** with 4 classes (UL, UR, LL, LR)
that both detects and classifies corner orientation from the distinctive
L-shaped photo/background boundary pattern.

### Why a Single 4-Class Model Instead of 4 Separate Models?

Each corner of a photo has a unique visual L-shape (┏ ┓ ┗ ┛). A single
4-class model can learn to both **detect** the corner AND **classify** its
orientation in one forward pass:

1. **Simpler deployment**: One model instead of four to train, export, and run
2. **Self-orienting**: The model tells you which corner type it found — no need
   to decide which model to run on which crop
3. **Shares features**: All 4 corner types share the same L-shaped boundary
   features — a single model learns these once instead of 4 times
4. **Efficient inference**: One forward pass gives you the corner type AND
   location, instead of running 4 separate models

## Architecture

### Single 4-Class Fiducial Model

| Class | Corner | L-Shape | Photo Extends |
|-------|--------|---------|---------------|
| 0 | UL | ┏ | → right, ↓ down |
| 1 | UR | ┓ | ← left, ↓ down |
| 2 | LL | ┗ | → right, ↑ up |
| 3 | LR | ┛ | ← left, ↑ up |

- **Type**: YOLO detection (4 classes: ul, ur, ll, lr)
- **Architecture**: yolo26n (nano — fast, sufficient for simple detection)
- **Input**: 640×640 crop centered near the approximate corner position
- **Output**: Bounding box with class indicating corner orientation
- **Corner position**: bbox center ≈ precise corner location
- **NO flip augmentation**: flipping changes corner orientation!

### Why Detection Instead of Pose/Keypoint?

Bounding box regression is generally more precise than keypoint regression for
small, well-defined objects. The corner is a visually compact feature — a
tight bbox around the L-shaped boundary gives a center point that is
effectively the corner location, and the class tells you which corner it is.

## Pipeline

```
Input Image
    │
    ▼
┌──────────────┐
│  Detection   │  1 forward pass
│   Model      │  → axis-aligned bounding boxes
└──────┬───────┘
       │ for each bbox, extract 4 corner crops (640×640):
       │
       ├─ UL crop ──┐
       ├─ UR crop ──┤
       ├─ LL crop ──┼──→ Single fiducial model ──→ 4 corners with types
       └─ LR crop ──┘
       │
       ▼
  4 precise corners per photo (with orientation labels)
       │
       ▼ (optional: iterate)
  Zoom in, re-crop, re-run fiducial model for sub-pixel accuracy
       │
       ▼
  Perspective warp → clean extracted photo
```

### Crop Extraction

For each detection bbox `(x1, y1, x2, y2)` with approximate corners:

| Corner | Approx. Position | Crop Region |
|--------|-------------------|-------------|
| UL | (x1, y1) | 640×640 centered near (x1, y1) |
| UR | (x2, y1) | 640×640 centered near (x2, y1) |
| LL | (x1, y2) | 640×640 centered near (x1, y2) |
| LR | (x2, y2) | 640×640 centered near (x2, y2) |

Each crop is centered NEAR the approximate corner (with offset), not exactly
on it, so the model learns to find corners at various positions in the frame.

### Iterative Refinement (Optional)

1. **Pass 1**: Crop around approximate corner → fiducial model → rough position
2. **Pass 2**: Re-crop around detected position → fiducial model → refined position

## Data Generation

### Key Design Principles

1. **Same background pipeline** as detection/pose generators (`random_base_background`
   + `apply_texture_overlay`) for consistent background variety
2. **Pre-multiplied alpha compositing** — avoids dark halos at photo edges
3. **Randomized corner position** within the 640×640 crop — not always centered
4. **Harris corner refinement** on binary mask for pixel-precise labels
5. **Large 2000×2000 render canvas** — no border replication, any corner that
   would extend beyond the canvas is simply skipped
6. **4 output classes** (UL=0, UR=1, LL=2, LR=3) in a single dataset

### Data Volume

- **Training**: ~4000 scenes → ~16000 crops total (~4000 per class)
- **Validation**: ~1000 scenes → ~4000 crops total (~1000 per class)

## NO Flip Augmentation

**CRITICAL**: No horizontal or vertical flip augmentation! Flipping changes
the corner orientation:
- Horizontal flip: UL ↔ UR, LL ↔ LR
- Vertical flip: UL ↔ LL, UR ↔ LR

Since the model must learn to classify orientation, flipping would produce
mislabeled training data.

## File Structure

```
data_generator/
├── generate_fiducial.py        # Single-model 4-class corner generator
├── generate_common.py          # Shared utilities
├── generate_detection.py       # Detection data
├── generate_pose.py            # Single-photo pose data
└── generate_pose_multi.py      # Multi-photo pose data

training/
├── dataset_fiducial.yaml       # Single 4-class dataset config
├── train_fiducial.py           # Single 4-class model training
├── train_detection.py
├── train_pose.py
├── train_pose_multi.py
└── train_both.py               # Updated to include fiducial

data_fiducial/                  # Single dataset (4 classes)
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

models/
├── detection_model.onnx
└── fiducial_corner.onnx        # Single model, 4 classes
```

## Inference Pipeline

```python
def extract_fiducial(image, detection_model, fiducial_model, iterations=2):
    # 1. Detect photo bounding boxes
    bboxes = detect_photos(image, detection_model)

    results = []
    for bbox in bboxes:
        # 2. Extract approximate corner positions from bbox
        approx_corners = {
            'ul': (bbox.x1, bbox.y1),
            'ur': (bbox.x2, bbox.y1),
            'll': (bbox.x1, bbox.y2),
            'lr': (bbox.x2, bbox.y2),
        }

        # 3. For each corner, run fiducial model with iterative refinement
        precise_corners = {}
        for name, (ax, ay) in approx_corners.items():
            for i in range(iterations):
                crop = extract_corner_crop(image, ax, ay)
                det = run_fiducial(crop, fiducial_model)
                # det.class tells us what orientation it found
                # det.center gives the precise corner position
                ax, ay = map_back_to_full_image(det.center, crop_offset)
            precise_corners[name] = (ax, ay)

        # 4. Perspective warp from 4 precise corners
        result = perspective_warp(image, precise_corners)
        results.append(result)

    return results
```

## Extract.py Pipeline Modes

The `extract.py` tool supports 5 pipeline modes:

| Mode | Pipeline | Best For |
|------|----------|----------|
| `detect` | Detection → bbox crop | Quick & dirty extraction |
| `pose` | Detection → Pose → crop/warp | Standard extraction |
| `fiducial` | Detection → Fiducial | Precise without pose model |
| `pose-fiducial` | Detection → Pose → Fiducial | Maximum accuracy |
| `multi-pose-fiducial` | Multi-pose → Fiducial | Multi-photo scenes |

**Recommended**: `pose-fiducial` with `--use-detection` for best accuracy.
The pose model provides better initial corner estimates than bbox corners,
so the fiducial model starts with a more accurate crop and converges faster.

```bash
# Maximum accuracy pipeline:
python extract.py pose-fiducial --use-detection --input scan.jpg --output ./extracted/

# Multi-photo context:
python extract.py multi-pose-fiducial --input scan.jpg --output ./extracted/

# Quick (no pose model):
python extract.py fiducial --input scan.jpg --output ./extracted/
```