# Photo Pose Detector

Detect the corners of physical photographs in camera-scanned images, extract each photo, and correct perspective distortion — powered by detection + pose models with geometric corner refinement and automatic rescue.

## The Problem

When you photograph multiple physical photos laid out on a table (camera scanning), the resulting image contains:

- Multiple photos at different positions and rotations
- Perspective distortion from the camera angle
- Drop shadows and glare from overhead lighting
- Varied table surface backgrounds

The goal: **automatically detect each photo, locate its four corners precisely, and extract a clean perspective-corrected image.**

## Current Architecture

Two production models work together in a pipeline, with an always-on rescue stage and optional corner refinement / CV post-processing. A third model (fiducial-pose segments) is in development:

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
┌──────────────┐
│ Pose Refine  │  Crop around each corner → re-run model for better localization
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Dedup      │  Remove duplicate detections
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Rescue     │  Always on — Sobel edge detection + line intersection
│ (automatic)  │  Recovers invisible/low-visibility corners when < 3 visible
└──────┬───────┘
       │
       ▼ (optional)
┌──────────────────────────┐
│  Corner Refinement       │  Crop around each corner → run model again
│  (--corner-refine)        │  Recover remaining invisible/low-vis corners
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

### Automatic Rescue

The pipeline includes an **always-on rescue stage** that requires no flags or configuration. When a photo has fewer than 3 visible corners, the rescue stage uses Sobel edge detection and line intersection analysis to recover invisible or low-visibility corners. This replaces the previous `--auto-refine` flag — rescue is now always active.

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
| **standard** | + pose-refine + adaptive margin + crop | ~1s | You want good-quality crops with margin |
| **thorough** | + pose-refine + corner-refine + cv-refine + adaptive margin + warp | ~3s | Invisible corners or maximum quality |

> The rescue stage always runs regardless of preset — no flag needed.

## Performance

Approximate processing times for a 1512×2016 scan with 4 photos:

| Mode | Time | Corners Found |
|------|------|---------------|
| Baseline (detect + pose) | ~700ms | May miss invisible corners |
| + Corner refine (pose) | ~2,500ms | All 4/4 corners with vis=1.0 |
| + Corner refine (detection) | ~1,900ms | Most corners recovered |

## Models

| Model | File | Status | Best mAP50 | Notes |
|-------|------|--------|-----------|-------|
| Detection | `models/detection_ep47.onnx` | ✅ Production | 0.995 | Epoch 47 |
| Pose (single) | `models/pose_single_ep42.onnx` | ✅ Production | 0.995 | Foundation of the pipeline |
| Fiducial-pose segments | (training in progress) | 🔄 In development | TBD | Detects visible edge segments for corner assembly |

### Failed Approaches

Previous attempts to classify corners directly all failed due to the inherent visual ambiguity of L-shaped photo corners:

| Approach | Why It Failed |
|----------|--------------|
| **Multi-pose model** | 0% background images → precision stuck at 0.50 |
| **4-class fiducial** | Corner orientations visually indistinguishable |
| **Binary fiducial** | Same problem — crop-level classification inherently ambiguous |

### Key Insight: Geometric Corner Refinement

Instead of training a new model to classify corners (which failed repeatedly), we reuse the **existing pose/detection models** on small crops around each corner. The pose model returns named keypoints that directly identify the corner. No new training needed, and it achieves vis=1.0 for all corners including previously invisible ones.

## Project Structure

```
photo-pose-detector/
├── data_generator/                # Synthetic training data generators
│   ├── generate_common.py        #   Shared utilities (backgrounds, textures, shadows)
│   ├── generate_detection.py     #   Detection model data generator
│   ├── generate_pose.py          #   ★ Pose model data generator
│   └── generate_fiducial_pose.py #   Fiducial-pose segment generator
├── data_detection/               # Detection training data (gitignored, regenerateable)
├── data_pose/                    # ★ Pose training data (gitignored, regenerateable)
├── data_fiducial_pose/           # Fiducial-pose training data (gitignored, in dev)
├── training/                     # Model training scripts & configs
│   ├── train_detection.py        #   Detection model trainer
│   ├── train_pose.py             #   ★ Pose model trainer
│   ├── train_fiducial_pose.py    #   Fiducial-pose trainer
│   ├── dataset_detection.yaml    #   Detection dataset config
│   ├── dataset_pose.yaml         #   ★ Pose dataset config
│   └── dataset_fiducial_pose.yaml#   Fiducial-pose dataset config
├── models/                       # Exported ONNX + PyTorch models
│   ├── detection_ep47.onnx       #   Active detection model
│   ├── detection_ep47.pt         #   Active detection model (PyTorch)
│   ├── pose_single_ep42.onnx     #   ★ Active pose model
│   └── pose_single_ep42.pt       #   ★ Active pose model (PyTorch)
├── onnx_inference/               # Main inference pipeline & CLI
│   └── photocrop.py              #   Single-file pipeline (all stages)
├── export/
│   └── export_onnx.py            # ONNX model export
├── tests/                        # Unit tests
├── docs/
│   ├── ARCHITECTURE.md           # Pipeline architecture & data flow
│   ├── PHOTOCROP_USAGE.md        # Full CLI reference
│   ├── KOTLIN_USAGE.md           # Kotlin/Android integration
│   └── FUTURE_IMPROVEMENTS.md    # Architecture evolution & future work
├── setup.sh                      # Installation script
├── pyproject.toml                # Package config
└── requirements.txt              # Python dependencies
```

★ = critical production pipeline component

## Full CLI Reference

See [`docs/PHOTOCROP_USAGE.md`](docs/PHOTOCROP_USAGE.md) for complete documentation including:

- All flags and options
- Corner refinement details
- Crop modes explained
- Adaptive margin
- Python API usage
- Troubleshooting

## Training

Each model has its own generator, dataset, and trainer:

```bash
# Generate data
cd data_generator
python generate_detection.py --mode batch
python generate_pose.py --mode batch
python generate_fiducial_pose.py --mode batch

# Train
cd training
python train_detection.py --epochs 100
python train_pose.py --epochs 100
python train_fiducial_pose.py --epochs 150

# Export to ONNX
python export/export_onnx.py --model runs/pose/photo-corner-detector/weights/best.pt
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full data flow diagram.

## Requirements

- Python 3.9+ (3.12 recommended)
- See [`requirements.txt`](requirements.txt) for full dependency list
- Quick install: `./setup.sh` (CPU) or `./setup.sh --gpu` (CUDA)

## License

Apache 2.0