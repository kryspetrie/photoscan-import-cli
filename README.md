# Photo Pose Detector

Detect the corners of physical photographs in camera-scanned images, extract each photo, and correct perspective distortion — powered by detection + binary fiducial models with optional pose and CV refinement.

## The Problem

When you photograph multiple physical photos laid out on a table (camera scanning), the resulting image contains:

- Multiple photos at different positions and rotations
- Perspective distortion from the camera angle
- Drop shadows and glare from overhead lighting
- Varied table surface backgrounds

The goal: **automatically detect each photo, locate its four corners precisely, and extract a clean perspective-corrected image.**

## Current Architecture

Three model types work together in a pipeline:

```
Input Image
    │
    ▼
┌──────────────┐
│  Detection   │  Find where the photos are (bounding boxes)
│   Model      │  1 forward pass → axis-aligned bboxes
└──────┬───────┘
       │
       │ (optional, recommended for multi-photo scenes)
       ▼
┌──────────────┐
│  Pose Model  │  Find approximate 4 corner positions per photo
│  (single)    │  1 forward pass → 4 keypoints (LL, UL, UR, LR)
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Binary Fiducial ×4  │  Refine corners to pixel/sub-pixel accuracy
│  (iterative)          │  4 forward passes per photo → precise corners
└──────┬───────────────┘
       │
       │ (optional)
       ▼
┌──────────────────┐
│  CV Refinement   │  Edge detection + line intersection → sub-pixel
└──────┬───────────┘
       │
       ▼
  Perspective warp → clean extracted photo
```

### Why This Architecture

| Stage | What It Does | Why It's Needed |
|-------|-------------|-----------------|
| **Detection** | Find photo regions | Essential — tells you where to look |
| **Pose** (optional) | Approximate corners | Helpful for multi-photo scenes; bbox corners are less accurate initial positions |
| **Binary Fiducial** | Precise corners | Each of 4 models answers one yes/no question: "Is this corner type present?" |
| **CV Refinement** (optional) | Sub-pixel corners | Pure CV post-processing; no ML inference |

**Why binary fiducial instead of 4-class?** The 4 corner types have nearly
identical visual appearance (they're all L-shaped boundaries differing only by
90° rotation). A 4-class model couldn't classify them (cls_loss ≈ random).
Binary models eliminate the classification problem: each model only decides
"corner present or not."

### Model Details

| Model | Architecture | Purpose | Output |
|-------|-------------|---------|--------|
| **Detection** | YOLO26n | Find photo bboxes | Axis-aligned bounding boxes |
| **Pose** (single) | YOLO26s-pose | Approximate corners | 4 keypoints (LL, UL, UR, LR) |
| **Fiducial-UL** | YOLO26n | Find UL corner (┏) | Corner bbox |
| **Fiducial-UR** | YOLO26n | Find UR corner (┓) | Corner bbox |
| **Fiducial-LL** | YOLO26n | Find LL corner (┗) | Corner bbox |
| **Fiducial-LR** | YOLO26n | Find LR corner (┛) | Corner bbox |

### Keypoint Order (Pose Model)

| Index | Name | Position |
|-------|------|----------|
| kp0 | LL | Lower-Left |
| kp1 | UL | Upper-Left |
| kp2 | UR | Upper-Right |
| kp3 | LR | Lower-Right |

### Iterative Fiducial Refinement

The fiducial model is most accurate when the corner is near the crop center:

1. **Pass 1**: Crop around approximate corner (from pose or bbox) → detect corner → map back to image coords
2. **Pass 2**: Re-crop centered on detected position → detect again → more precise
3. **Pass 3** (optional): Repeat for sub-pixel accuracy

## Training Status

| Model | Status | Best mAP50 | Notes |
|-------|--------|-----------|-------|
| Detection | ✅ Complete | 0.995 | Epoch 47 |
| Pose (single) | ✅ Complete | 0.995 | Keypoint accuracy limited (mAP50-95=0.107) |
| Fiducial-UL | 🔄 Training | 0.917 (ep 10) | Currently training (lr=0.001, mosaic=0) |
| Fiducial-UR | ⏳ Queued | — | Awaits UL completion |
| Fiducial-LL | ⏳ Queued | — | Awaits UR completion |
| Fiducial-LR | ⏳ Queued | — | Awaits LL completion |

### Past Approaches (Failed)

| Approach | Why It Failed |
|----------|--------------|
| **Multi-pose model** | 0% background images → precision stuck at 0.50; overfitting after epoch 19 |
| **4-class fiducial** | Corner orientations indistinguishable → cls_loss ≈ random (ln(4)≈1.39) |

Failed files are archived in `REMOVED/`. See [`REMOVED/README.md`](REMOVED/README.md).

## Quick Start

```bash
# Install
./setup.sh    # or: ./setup.sh --gpu

# Download source data
python download_oxford.py
python download_textures.py

# Generate training data
cd data_generator
python3 generate_detection.py --mode batch --train-count 4000 --val-count 1000 --source ./images --output ../data_detection
python3 generate_pose.py --mode batch --train-count 4000 --val-count 1000 --source ./images --output ../data_pose
python3 generate_fiducial.py --mode batch --train-count 4000 --val-count 1000 --source ./images --output ../data_fiducial
cd ..

# Split fiducial data into 4 binary datasets
cd training
python3 split_binary_datasets.py

# Train full pipeline (detection + pose + fiducial)
python3 train_pipeline.py

# Train pose model (single-photo crops)
python3 train_pose.py --epochs 100 --batch 16 --device cpu --cache ram

# Train 4 binary fiducial models (sequential)
python3 train_fiducial_binary.py

# Validate
python3 validate.py --model runs/detection/photo-detector/weights/best.pt

# Export to ONNX
cd ../export
python3 export_onnx.py --all
```

## Inference

```bash
# Detect + crop with keypoints
photocrop --image scan.jpg --crop warp --crop-margin 0.02 --border-fill white

# Maximum quality (pose + fiducial + CV refine)
photocrop --image scan.jpg --preset best --fiducial-model fiducial_corner.onnx

# Batch processing
photocrop --image ./scans/ --output ./crops/ --crop warp

# Coordinates only (no image output)
photocrop --image scan.jpg --coords json --no-image
```

> 📄 Full CLI reference: [`docs/PHOTOCROP_USAGE.md`](docs/PHOTOCROP_USAGE.md)

## Project Structure

```
photo-pose-detector/
├── download_oxford.py              # Download Oxford Buildings Dataset
├── download_textures.py            # Download & process DTD textures
│
├── data_generator/
│   ├── generate_common.py          # Shared generation utilities
│   ├── generate_detection.py       # Detection data (bbox labels)
│   ├── generate_pose.py            # Pose data (keypoint labels, tight crops)
│   ├── generate_fiducial.py        # Fiducial data (corner crops, 4-class labels)
│   ├── SYSTEM_DOCUMENTATION.md     # Generator technical docs
│   └── images/                     # Source photos (Oxford Buildings)
│
├── training/
│   ├── train_detection.py          # Detection model training
│   ├── train_pose.py              # Single-photo pose model training
│   ├── train_fiducial_binary.py   # Binary fiducial model training (×4)
│   ├── split_binary_datasets.py   # Split 4-class data into 4 binary datasets
│   ├── dataset_detection.yaml      # Detection dataset config
│   ├── dataset_pose.yaml           # Pose dataset config
│   └── validate.py                 # Model validation
│
├── data_detection/                 # Detection training data
├── data_pose/                      # Pose training data (single-photo crops)
├── data_fiducial/                  # Fiducial source data (4-class, used for binary split)
├── data_fiducial_binary/           # Binary fiducial training data (ul/ur/ll/lr)
├── textures/                       # Background textures (DTD)
│
├── export/
│   └── export_onnx.py             # ONNX export
│
├── models/                          # Exported ONNX models
│   ├── detection_ep47.onnx         # Detection model (recommended)
│   ├── pose_single_ep42.onnx       # Single-photo pose model (recommended)
│   └── ...
│
├── onnx_inference/
│   ├── __init__.py
│   ├── __main__.py                 # `python -m onnx_inference` entry point
│   └── photocrop.py                # Main inference pipeline & CLI
│
├── docs/
│   ├── PHOTOCROP_USAGE.md          # Full CLI reference
│   ├── FIDUCIAL_BINARY.md          # Binary fiducial model documentation
│   ├── FIDUCIAL_PLAN.md            # Original 4-class plan (archived)
│   ├── FUTURE_IMPROVEMENTS.md       # Architecture evolution & future work
│   └── KOTLIN_USAGE.md             # Kotlin/Android integration
│
├── REMOVED/                          # Archived failed approaches
│   ├── README.md                     # What failed and why
│   ├── multi_pose/                   # Multi-pose model (overfit, 0% bg)
│   └── 4class_fiducial/             # 4-class fiducial (cls_loss ≈ random)
│
├── pyproject.toml
├── requirements.txt
└── setup.sh
```

## Training Hyperparameters

### Detection Model

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | yolo26n | Nano: fast, sufficient for bbox detection |
| lr0 | 0.001 | Standard for AdamW/auto |
| mosaic | 0.5 | Helps detect small photos in multi-photo scenes |
| fliplr | 0.5 | Photos are rotation-agnostic for detection |
| batch | 16 | Balanced for CPU |

### Pose Model (Single-Photo)

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | yolo26s-pose | Small: more capacity for keypoint precision |
| lr0 | 0.001 | Standard |
| mosaic | 0.3 | Photo fills frame; less mosaic needed |
| fliplr | 0.5 | flip_idx handles LL↔LR, UL↔UR correctly |
| scale | 0.2 | Less augmentation (photo already fills crop) |
| translate | 0.05 | Tight crop has less translation room |

### Binary Fiducial Models (Critical Settings)

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | yolo26n | Nano: corner detection/cropping is simpler than classification |
| **lr0** | **0.001** | **Default 0.01 causes violent oscillation** (mAP50 bouncing 0.26→0.86) |
| **optimizer** | **auto (AdamW)** | **SGD with lr=0.01 oscillates** |
| **mosaic** | **0.0** | **Crops are already tightly framed**; mosaic creates unrealistic composites |
| fliplr | 0.0 | Flipping changes corner orientation (mislabels data) |
| flipud | 0.0 | Same reason |
| hsv_s | 0.3 | Consistent with other models |
| batch | 16 | Standard |

## Requirements

- Python 3.9+ (3.12 recommended)
- See [`requirements.txt`](requirements.txt) for full dependency list
- Quick install: `./setup.sh` (CPU) or `./setup.sh --gpu` (CUDA)

## License

Apache 2.0