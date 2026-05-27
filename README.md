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

Three production models work together in a multi-stage pipeline, with an always-on rescue stage and optional refinement stages. A fourth model (fiducial-pose segments) is in development:

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
       ▼ (optional, --pose-refine)
┌──────────────┐
│ Pose Refine  │  Re-derive bbox from keypoints → re-run pose on tighter crop
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Dedup      │  Remove duplicate detections
└──────┬───────┘
       │
       ▼ (enabled by default; --no-warp-recover to disable)
┌──────────────────────────┐
│   Warp Recovery            │  Compute warp score per photo; iteratively
│                            │  re-pose with larger crops for high-warp
│                            │  detections
└──────┬───────────────────┘
       │
       ▼
┌──────────────┐
│   Rescue     │  Always on — Sobel edge detection + line intersection
│ (automatic)  │  Recovers invisible/low-visibility corners when < 3 visible
└──────┬───────┘
       │
       ▼ (optional, --corner-refine)
┌──────────────────────────┐
│  Corner Refinement        │  320×320 crop around each corner → regression
│  (--corner-refine)         │  model for precise corner position (vis=1.0)
└──────┬───────────────────┘
       │
       ▼ (optional, --cv-refine)
┌──────────────┐
│  CV Refinement │  Sobel edge detection + line intersection → sub-pixel
│  (--cv-refine) │
└──────┬───────┘
       │
       ▼
  Perspective warp → clean extracted photo
```

### Warp Recovery

After dedup, the pipeline computes a **warp score** for each detected photo. A photo whose opposing edge lengths differ by more than 15% (warp > 1.15) is flagged for recovery. The pipeline iteratively re-runs the pose model with progressively larger crops until the warp drops below the threshold or max iterations is reached. This prevents misdetections from locking onto edges of adjacent photos.

Enabled by default in all presets. Disable with `--no-warp-recover`.

### Automatic Rescue

The pipeline includes an **always-on rescue stage** that requires no flags or configuration. When a photo has fewer than 3 visible corners, the rescue stage uses Sobel edge detection and line intersection analysis to recover invisible or low-visibility corners. This replaces the previous `--auto-refine` flag — rescue is now always active.

### Why Corner Refinement

The pose model sometimes reports invisible corners (visibility ≈ 0) when a photo's edge is occluded by shadow, glare, or the scanner edge. Corner refinement crops around each approximate corner and runs the model again, recovering these invisible corners to visibility = 1.0.

- **Regression model** (default, `--corner-refine-model regression`): Dedicated 320×320 corner regression model — most precise, recovers invisible corners reliably.
- **Pose model** (`--corner-refine-model pose`): Uses named keypoints directly — fast, single pass per corner, no classification needed.
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

# Default: corner_refine preset with warp-stretch crop
photocrop --image scan.jpg

# Fast scan — detect + pose only (~1s)
photocrop --image scan.jpg --preset fast

# Best quality
photocrop --image scan.jpg --preset corner_refine

# Coordinates only (for scripting)
photocrop --image scan.jpg --coords json --no-image
```

> If `photocrop` isn't on your PATH yet, you can also run it as a module:
> ```bash
> python -m onnx_inference --image scan.jpg
> ```

## Presets

| Preset | What it does | Time | Use when |
|--------|-------------|------|----------|
| **fast** | Detect + pose + dedup + warp recovery | ~1s | You only need coordinates, no crops |
| **pose_refine** | + pose-refine + adaptive margin + dedup + warp recovery | ~1s | You want good-quality crops with margin |
| **corner_refine** | + pose-refine + corner-refine (regression) + adaptive margin + dedup + warp recovery | ~3s | Invisible corners or maximum quality |

> The rescue stage always runs regardless of preset — no flag needed.
> Warp recovery is enabled by default in all presets. Disable with `--no-warp-recover`.

## Performance

Approximate processing times for a 1512×2016 scan with 4 photos:

| Mode | Time | Corners Found |
|------|------|---------------|
| Baseline (detect + pose) | ~700ms | May miss invisible corners |
| + Corner refine (pose) | ~2,500ms | All 4/4 corners with vis=1.0 |
| + Corner refine (detection) | ~1,900ms | Most corners recovered |
| + Corner refine (regression) | ~3,200ms | All 4/4 corners, best precision |

## Models

| Model | File | Status | Best mAP50 | Notes |
|-------|------|--------|-----------|-------|
| Detection | `models/detection_ep47.onnx` | ✅ Production | 0.995 | Epoch 47, yolo26s |
| Pose (single) | `models/pose_single_ep42.onnx` | ✅ Production | 0.995 | Foundation of the pipeline, yolo26s-pose |
| Corner regression | `models/corner-regression-v2.onnx` | ✅ Production | Pose mAP50-95=0.994 | 320×320 input, yolo26n-pose, epoch 24 |
| Fiducial-pose segments | (training in progress) | 🔄 In development | TBD | Detects visible edge segments for corner assembly |

### Failed Approaches

Previous attempts to classify corners directly all failed due to the inherent visual ambiguity of L-shaped photo corners:

| Approach | Why It Failed |
|----------|--------------|
| **Multi-pose model** | 0% background images → precision stuck at 0.50 |
| **4-class fiducial** | Corner orientations visually indistinguishable |
| **Binary fiducial** | Same problem — crop-level classification inherently ambiguous |

### Key Insight: Geometric Corner Refinement

Instead of training a new model to classify corners (which failed repeatedly), we reuse the **existing pose/detection models** on small crops around each corner. The pose model returns named keypoints that directly identify the corner. For even better precision, the dedicated corner regression model operates on 320×320 crops and achieves near-perfect localization.

## Project Structure

```
photo-pose-detector/
├── data_generator/                # Synthetic training data generators
│   ├── generate_common.py          #   Shared utilities (backgrounds, textures, shadows)
│   ├── generate_detection.py       #   Detection model data generator
│   ├── generate_pose.py            #   ★ Pose model data generator
│   ├── generate_corner_regression.py # Corner regression data generator
│   └── generate_fiducial_pose.py   #   Fiducial-pose segment generator
├── data_detection/                 # Detection training data (gitignored, regenerateable)
├── data_pose/                      # ★ Pose training data (gitignored, regenerateable)
├── data_corner_regression/         # Corner regression data (gitignored, regenerateable)
├── data_fiducial_pose/             # Fiducial-pose training data (gitignored, in dev)
├── training/                       # Model training scripts & configs
│   ├── train_detection.py          #   Detection model trainer
│   ├── train_pose.py               #   ★ Pose model trainer
│   ├── train_corner_regression.py  #   Corner regression trainer
│   ├── train_fiducial_pose.py      #   Fiducial-pose trainer
│   ├── dataset_detection.yaml      #   Detection dataset config
│   ├── dataset_pose.yaml           #   ★ Pose dataset config
│   ├── dataset_corner_regression.yaml # Corner regression dataset config
│   └── dataset_fiducial_pose.yaml  #   Fiducial-pose dataset config
├── models/                         # Exported ONNX + PyTorch models
│   ├── detection_ep47.onnx         #   Active detection model
│   ├── detection_ep47.pt           #   Active detection model (PyTorch)
│   ├── pose_single_ep42.onnx      #   ★ Active pose model
│   ├── pose_single_ep42.pt         #   ★ Active pose model (PyTorch)
│   └── corner-regression-v2.onnx   #   Corner regression model
├── onnx_inference/                 # Main inference pipeline & CLI
│   ├── photocrop.py                #   Single-file pipeline (all stages)
│   ├── __init__.py                 #   Package init
│   └── __main__.py                 #   python -m onnx_inference entry point
├── export/
│   └── export_onnx.py              # ONNX model export
├── tests/                          # Unit tests
├── docs/
│   ├── ARCHITECTURE.md             # Pipeline architecture & data flow
│   ├── PHOTOCROP_USAGE.md          # Full CLI reference
│   ├── KOTLIN_USAGE.md             # Kotlin/Android integration
│   └── FUTURE_IMPROVEMENTS.md      # Architecture evolution & future work
├── real_world_examples/            # Test images with expected outputs
├── setup.sh                        # Installation script
├── pyproject.toml                  # Package config
└── requirements.txt                # Python dependencies
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
python generate_corner_regression.py --mode batch
python generate_fiducial_pose.py --mode batch

# Train
cd training
python train_detection.py --epochs 100
python train_pose.py --epochs 100
python train_corner_regression.py --epochs 50
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