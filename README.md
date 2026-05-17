# Photo Pose Detector

Detect the corners of physical photographs in camera-scanned images, extract each photo, and correct perspective distortion — powered by detection + pose models with geometric corner refinement.

## The Problem

When you photograph multiple physical photos laid out on a table (camera scanning), the resulting image contains:

- Multiple photos at different positions and rotations
- Perspective distortion from the camera angle
- Drop shadows and glare from overhead lighting
- Varied table surface backgrounds

The goal: **automatically detect each photo, locate its four corners precisely, and extract a clean perspective-corrected image.**

## Current Architecture

Two models work together in a pipeline, with optional corner refinement and CV post-processing:

```
Input Image
    │
    ▼
┌──────────────┐
│  Detection   │  Find photo bounding boxes (1 forward pass)
│   Model      │  → N axis-aligned bounding boxes
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Pose Model  │  Find approximate 4 corners per photo (1 pass per photo)
│  (single)    │  → 4 keypoints (LL, UL, UR, LR) + visibility
└──────┬───────┘
       │
       ▼ (optional, recommended)
┌──────────────────────────┐
│  Corner Refinement       │  Recover invisible/low-vis corners
│  (--corner-refine)        │  Crops around each corner → runs model again
└──────┬───────────────────┘
       │
       ▼ (optional)
┌──────────────────┐
│  CV Refinement   │  Sobel edge detection + line intersection → sub-pixel
│  (--cv-refine)    │
└──────┬───────────┘
       │
       ▼
  Perspective warp → clean extracted photo
```

### Why Corner Refinement

The pose model sometimes reports invisible corners (visibility ≈ 0) when a photo's edge is occluded by shadow, glare, or the scanner edge. Corner refinement crops around each approximate corner and runs the model again, recovering these invisible corners to visibility = 1.0.

- **Pose model** (default): Uses named keypoints directly — fast, single pass per corner, no classification needed.
- **Detection model** (fallback): Uses bounding box corners with geometric classification — no extra model session needed, but less precise.

### Keypoint Order

| Index | Name | Position |
|-------|------|----------|
| kp0 | LL | Lower-Left |
| kp1 | UL | Upper-Left |
| kp2 | UR | Upper-Right |
| kp3 | LR | Lower-Right |

## Quick Start

```bash
# Install
./setup.sh    # or: ./setup.sh --gpu

# Perspective warp with auto-fix (recommended)
photocrop --image scan.jpg --preset warp

# Best quality — recovers invisible corners
photocrop --image scan.jpg --preset best

# Coordinates only (for scripting)
photocrop --image scan.jpg --coords json --no-image
```

> If `photocrop` isn't on your PATH yet, you can also run it as a module:
> ```bash
> python -m onnx_inference --image scan.jpg --preset warp
> ```

## Presets

| Preset | What it does | Time | Use when |
|--------|-------------|------|----------|
| **quick** | Detect + pose only, no cropping | ~1s | You only need coordinates, no crops |
| **crop** | + auto-refine + adaptive margin + simple-corners crop | ~1s | You want rectangular crops with margin |
| **warp** | + auto-refine + adaptive margin + perspective warp | ~1s | You want perspective-corrected crops |
| **best** | + corner-refine + cv-refine + auto-refine + adaptive margin + warp | ~3s | Invisible corners or maximum quality |

## Performance

Approximate processing times for a 1512×2016 scan with 4 photos:

| Mode | Time | Corners Found |
|------|------|---------------|
| Baseline (detect + pose) | ~700ms | May miss invisible corners |
| + Corner refine (pose) | ~2,500ms | All 4/4 corners with vis=1.0 |
| + Corner refine (detection) | ~1,900ms | Most corners recovered |
| ~~Sweep~~ (superseded) | ~~6,900ms~~ | Same as corner refine but slower |

## Training Status

| Model | Status | Best mAP50 | Notes |
|-------|--------|-----------|-------|
| Detection | ✅ Complete | 0.995 | Epoch 47 |
| Pose (single) | ✅ Complete | 0.995 | Keypoint accuracy limited (mAP50-95=0.107) |

### Failed Approaches (Archived in `REMOVED/`)

| Approach | Why It Failed |
|----------|--------------|
| **Multi-pose model** | 0% background images → precision stuck at 0.50; overfitting after epoch 19 |
| **4-class fiducial** | Corner orientations visually indistinguishable → cls_loss ≈ random |
| **Binary fiducial** | Same problem — crop-level classification inherently ambiguous |

### Key Insight: Geometric Corner Refinement

Instead of training a new model to classify corners (which failed repeatedly), we reuse the **existing pose/detection models** on small crops around each corner. The pose model returns named keypoints that directly identify the corner. No new training needed, and it achieves vis=1.0 for all corners including previously invisible ones.

## Project Structure

```
photo-pose-detector/
├── download_oxford.py              # Download Oxford Buildings Dataset
├── download_textures.py           # Download DTD textures (backgrounds)
├── data_generator/                # Generate training data
├── training/                       # Model training scripts
├── models/                         # Exported ONNX models
│   ├── detection_ep47.onnx        # Detection model (recommended)
│   └── pose_single_ep42.onnx      # Pose model (recommended)
├── onnx_inference/                 # Main inference pipeline & CLI
│   └── photocrop.py                # Single-file pipeline (all stages)
├── docs/
│   ├── PHOTOCROP_USAGE.md          # Full CLI reference
│   ├── KOTLIN_USAGE.md             # Kotlin/Android integration
│   └── FUTURE_IMPROVEMENTS.md      # Architecture evolution & future work
└── REMOVED/                         # Archived failed approaches
```

## Full CLI Reference

See [`docs/PHOTOCROP_USAGE.md`](docs/PHOTOCROP_USAGE.md) for complete documentation including:

- All flags and options
- Corner refinement details
- Crop modes explained
- Adaptive margin
- Python API usage
- Troubleshooting

## Requirements

- Python 3.9+ (3.12 recommended)
- See [`requirements.txt`](requirements.txt) for full dependency list
- Quick install: `./setup.sh` (CPU) or `./setup.sh --gpu` (CUDA)

## License

Apache 2.0