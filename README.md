# Photo Pose Detector

Detect the corners of physical photographs in camera-scanned images, extract each photo, and correct perspective distortion — powered by two custom YOLO models.

## The Problem

When you photograph multiple physical photos laid out on a table (camera scanning), the resulting image contains:

- Multiple photos at different positions and rotations
- Perspective distortion from the camera angle
- Drop shadows and glare from overhead lighting
- Varied table surface backgrounds

The goal: **automatically detect each photo, locate its four corners precisely, and extract a clean perspective-corrected image.**

## Current Status

**Both models are fully trained and exported to ONNX.** The project is ready for deployment.

| Model | Status | Best Epoch | mAP50 | mAP50-95 | ONNX Size |
|-------|--------|-----------|-------|----------|------------|
| **Detection** | ✅ Complete | 4 | 0.995 | 0.918 | 10.2 MB |
| **Pose** | ✅ Complete | 15 | 0.995 | 0.107 | 10.0 MB |

**What these numbers mean:**
- **mAP50 = 0.995** — both models find photos with near-perfect accuracy at 50% IoU
- **Detection mAP50-95 = 0.918** — bounding boxes are extremely precise
- **Pose mAP50-95 = 0.107** — strict keypoint localization is still improving; corners are detected correctly at coarse threshold but fine-grained precision (IoU > 75%) needs more training or data

### Trained Artifacts

```
models/
├── detection_model.onnx        # ONNX export, opset 17, dynamic shapes
└── pose_model.onnx              # ONNX export, opset 17, dynamic shapes

training/
└── runs/
    ├── detection/photo-detector/weights/
    │   ├── best.pt              # Best checkpoint (epoch 4)
    │   └── last.pt              # Last checkpoint (epoch 6)
    └── pose/photo-corner-detector/weights/
        ├── best.pt              # Best checkpoint (epoch 15)
        └── last.pt              # Last checkpoint (epoch 44)
```

> **Hardware note:** Models were trained on an Intel i7-9750H CPU (16 GB RAM) using `--device cpu --cache ram`. Training took ~2 hours for detection and ~27 hours for pose. On a CUDA GPU, training would complete in minutes.

---

## Two-Model Architecture

Two YOLO models work together in a pipeline — one finds where the photos are, the other pinpoints their exact corners:

```
Input Image
    │
    ▼
┌───────────────────────────┐
│  1. Detection Model (YOLO)│  → Where are the photos?
│     Axis-aligned bbox     │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│  2. Pose Model (YOLO-Pose)│  → Where are the 4 corners?
│     4 keypoints per photo │
└───────────────────────────┘
    │
    ▼
┌───────────────────────────┐
│  3. Extract & Correct      │  → Perspective transform
│     Crop each photo using  │
│     detected quadrilateral │
└───────────────────────────┘
```

**Why two models?** Detection is fast and robust at finding regions of interest. Pose is precise at localizing corners. Together they're more accurate than either alone.

### Model Details

| | Detection Model | Pose Model |
|---|---|---|
| **Purpose** | Find photo regions | Find exact corners |
| **YOLO variant** | Standard detection | YOLO-Pose (keypoints) |
| **Base weights** | `yolo11n.pt` | `yolo26n-pose.pt` |
| **Output per photo** | Bounding box (x, y, w, h) | 4 corner keypoints |
| **Label format** | 5 columns | 17 columns (bbox + 4×3 keypoints) |
| **ONNX output shape** | `(batch, 5, 8400)` | `(batch, 300, 18)` |
| **Speed** | Very fast | Fast |

### Keypoint Order

The Pose model outputs 4 corners in a fixed order. This order is critical for the perspective transform:

| Index | Name | Position |
|-------|------|----------|
| kp0 | LL | Lower-Left |
| kp1 | UL | Upper-Left |
| kp2 | UR | Upper-Right |
| kp3 | LR | Lower-Right |

Horizontal flip augmentation swaps: LL↔LR, UL↔UR (configured via `flip_idx: [2, 3, 0, 1]`).

---

## Training Data

### Download Source Data

Two datasets are needed for synthetic data generation. Download scripts are included:

```bash
# Download Oxford Buildings Dataset (5,062 source photos, ~1.5 GB)
python download_oxford.py

# Download and process DTD textures (85 selected textures, ~600 MB download)
python download_textures.py
```

| Dataset | Purpose | Count | License |
|---------|---------|-------|---------|
| [Oxford Buildings (Oxford5k)](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) | Source photo content | 5,062 images | CC BY-NC-SA 4.0 |
| [Describable Textures (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | Background textures | 85 selected images | CC BY-NC-SA 4.0 |

The DTD textures are automatically processed (resized to 1200×1200, converted to greyscale, brightness-normalized to medium grey) before being placed in `textures/`.

### Synthetic Data Generation

Training data is generated synthetically — no manual annotation required. The generator places real photo content onto random backgrounds with realistic distortions and effects.

**Source material:**
- **5,062 source photos** from the Oxford Buildings Dataset in `data_generator/images/`
- **85 background textures** (processed from DTD) in `textures/`

**Per-image variation:**
- 1–4 photos randomly placed per 640×640 image
- Random rotation (up to ±30° for single, ±5° for multi-photo)
- Drop shadows (2–5px offset, Gaussian blur, 15–35% opacity)
- Specular glare (2–4 elliptical flares per photo)
- Perspective warp (5% strength, corner-safe)
- Varied backgrounds (dark, light, colored + gradients + textures)

**Both label formats from a single pipeline:**

| Label Type | Columns | Example |
|------------|---------|---------|
| Detection | `class x y w h` | `0 0.501715 0.505848 0.378018 0.590964` |
| Pose | `class x y w h kp0x kp0y kp0v kp1x kp1y kp1v kp2x kp2y kp2v kp3x kp3y kp3v` | `0 0.501715 0.505848 0.378018 0.590964 0.312 0.801 2 0.690 0.801 2 0.689 0.210 2 0.314 0.210 2` |

> 📄 For full technical details on the generator (placement algorithm, rotation math, shadow rendering, etc.), see [`data_generator/SYSTEM_DOCUMENTATION.md`](data_generator/SYSTEM_DOCUMENTATION.md).

### Generate Training Data

```bash
cd data_generator

# Quick test — 10 images with debug overlays
python generate.py --count 10

# Full dataset — train/val split
python generate.py --mode batch --train-count 4000 --val-count 1000 --output ../data
```

This produces:

```
data/
├── images/
│   ├── train/              # Training images (shared by both models)
│   └── val/                # Validation images
├── detection/
│   └── labels/
│       ├── train/          # 5-column bounding box labels
│       └── val/
└── pose/
    └── labels/
        ├── train/          # 17-column keypoint labels (bbox + 4 corners × 3)
        └── val/
```

**Same images, different labels.** Each model's dataset YAML points to the shared images with its own label directory.

> ⚠️ **Symlink requirement:** Ultralytics resolves labels by replacing `/images/` with `/labels/` in the image path. Both YAMLs point to `data/images/`, so both expect `data/labels/`. The training scripts (`train_detection.py`, `train_pose.py`) automatically create/switch a symlink at `data/labels → data/detection/labels` or `data/pose/labels` as needed. See [`label_links.py`](training/label_links.py) for details.
>
> **Future improvement:** Generate data directly in `data/detection/images/` + `data/detection/labels/` (self-contained trees), eliminating the symlink entirely.

---

## Training

### Setup

```bash
# Quick setup — creates venv with CPU-only PyTorch
./setup.sh

# With CUDA 11.8 GPU support
./setup.sh --gpu

# With CUDA 12.1 GPU support
./setup.sh --cuda12
```

### Train Models

```bash
cd training

# Train detection model (finds bounding boxes)
python3 train_detection.py --epochs 100 --batch 16 --device cpu --cache ram

# Train pose model (finds corner keypoints)
python3 train_pose.py --epochs 100 --batch 16 --device cpu --cache ram

# Or train both sequentially
python3 train_both.py --epochs 100 --batch 16 --device cpu
```

**Important training notes:**
- `--cache ram` loads images into RAM on first epoch, making each subsequent epoch ~35 min instead of 8+ hours
- `--device cpu` is recommended on Intel Macs (MPS is slower than CPU on non-Apple Silicon)
- Mac users need `PYTORCH_ENABLE_MPS_FALLBACK=1` (set automatically by training scripts)
- The symlink at `data/labels` is automatically managed — switching between detection and pose training re-points it

### Hyperparameters

| Parameter | Detection | Pose | Why |
|-----------|-----------|------|-----|
| Base model | yolo11n | yolo26n-pose | Standard vs. pose architecture |
| Mosaic | 0.5 | 0.5 | Reduced from default |
| Mixup | 0.1 | 0.0 | Disabled for pose (moves keypoints) |
| Scale | 0.5 | 0.3 | Reduced for pose to preserve keypoint positions |
| Degrees | 10 | 10 | Moderate rotation augmentation |
| Flip LR | 0.5 | 0.5 | Pose uses `flip_idx` for correct keypoint swap |
| Flip UD | 0.0 | 0.0 | Would swap top/bottom corners incorrectly |
| Patience | 20 | 20 | Early stopping |

### Validate

```bash
python validate.py --model runs/detection/photo-detector/weights/best.pt
python validate.py --model runs/pose/photo-corner-detector/weights/best.pt
```

---

## Export & Deployment

### ONNX Export

```bash
cd export

# Export both models
python3 export_onnx.py --all

# Export a specific model
python3 export_onnx.py --model ../training/runs/detection/photo-detector/weights/best.pt

# Export with test inference
python3 export_onnx.py --all --test
```

Produces:
- `models/detection_model.onnx` — 10.2 MB
- `models/pose_model.onnx` — 10.0 MB

Both models use:
- **ONNX opset 17** — broad runtime compatibility
- **Dynamic input shapes** — accept any `(batch, 3, H, W)` at runtime
- **Simplified graph** via `onnxslim`

### Python Inference (ONNX)

```bash
cd onnx_inference
python3 infer.py --model ../models/detection_model.onnx --image ../data/images/val/val_000001.jpg
python3 infer.py --model ../models/pose_model.onnx --image ../data/images/val/val_000001.jpg
```

### Python Inference (Ultralytics, for testing)

```python
from ultralytics import YOLO
import cv2
import numpy as np

det_model = YOLO('training/runs/detection/photo-detector/weights/best.pt')
pose_model = YOLO('training/runs/pose/photo-corner-detector/weights/best.pt')

image = cv2.imread('scanned_photo.jpg')

# Step 1: Detect photo regions
det_results = det_model.predict(image, verbose=False)
boxes = det_results[0].boxes

# Step 2: Find corners in each region
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    region = image[y1:y2, x1:x2]

    pose_results = pose_model.predict(region, verbose=False)
    corners = pose_results[0].keypoints.xy[0]  # (4, 2) — LL, UL, UR, LR

    # Map corners back to full image
    corners[:, 0] += x1
    corners[:, 1] += y1

    # Step 3: Extract with perspective correction
    src = corners.cpu().numpy().astype(np.float32)
    w = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    h = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    dst = np.array([[0, h-1], [0, 0], [w-1, 0], [w-1, h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    extracted = cv2.warpPerspective(image, M, (w, h))
    cv2.imwrite('extracted_photo.jpg', extracted)
```

### Kotlin / Android Deployment

The ONNX model integrates into Kotlin using ONNX Runtime + BoofCV:

```kotlin
val detector = PhotoCornerDetector("pose_model.onnx")
val detected = detector.detect(scannedImage)         // List<DetectedPhoto>
val extracted = detector.extractPhotos(scannedImage)  // Perspective-corrected crops
```

> 📄 Full Kotlin integration guide with Gradle setup, preprocessing, and perspective transform: [`docs/KOTLIN_USAGE.md`](docs/KOTLIN_USAGE.md)

---

## Project Structure

```
photo-pose-detector/
├── download_oxford.py              # Download Oxford Buildings Dataset
├── download_textures.py            # Download & process DTD textures
│
├── data_generator/
│   ├── generate.py                  # Main generator (examples + batch modes)
│   ├── SYSTEM_DOCUMENTATION.md      # Detailed generator documentation
│   └── images/                      # 5,062 source photos (Oxford Buildings)
│
├── textures/                        # 85 processed greyscale textures (DTD)
│
├── training/
│   ├── dataset_detection.yaml       # Detection dataset config
│   ├── dataset_pose.yaml            # Pose dataset config (kpt_shape: [4, 3])
│   ├── label_links.py               # Symlink manager for label paths
│   ├── train_detection.py           # Detection training script
│   ├── train_pose.py               # Pose training script
│   ├── train_both.py               # Train both sequentially
│   ├── validate.py                  # Validation script
│   └── runs/                        # Training outputs (gitignored)
│       ├── detection/photo-detector/weights/
│       │   ├── best.pt
│       │   └── last.pt
│       └── pose/photo-corner-detector/weights/
│           ├── best.pt
│           └── last.pt
│
├── export/
│   └── export_onnx.py              # ONNX export script (both models)
│
├── models/                          # Exported ONNX models
│   ├── detection_model.onnx         # 10.2 MB
│   └── pose_model.onnx              # 10.0 MB
│
├── onnx_inference/
│   └── infer.py                    # Python ONNX inference testing
│
├── docs/
│   ├── KOTLIN_USAGE.md             # Kotlin/Android integration guide
│   └── REFACTORING_PLAN.md         # Draft: separate data pipeline plan
│
├── data/                            # Generated datasets (gitignored)
├── requirements.txt                 # Python dependency list
├── setup.sh                         # One-command environment setup
└── venv/                            # Legacy venv (gitignored, still in use)
```

---

## Known Issues & Pitfalls

| Problem | Cause | Solution |
|---------|-------|----------|
| **All training losses = 0** | Missing `data/labels` symlink → Ultralytics can't find labels | Training scripts now auto-create it via `label_links.py` |
| **MPS crash: `torchvision::nms not implemented`** | NMS not supported on MPS device | `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically |
| **Training epoch time doubles each epoch** | Images re-read from disk every epoch | Use `--cache ram` to cache in memory |
| **Pose labels: "require 13 columns" error** | `kpt_shape` was `[4, 2]` but labels have visibility (3 values per keypoint) | Set `kpt_shape: [4, 3]` in `dataset_pose.yaml` |
| **Nested output dirs** | Running from wrong working directory | Always `cd training/` before running scripts |
| **Keypoints wrong order** | Incorrect `flip_idx` | Verify `flip_idx: [2, 3, 0, 1]` in `dataset_pose.yaml` |
| **Cached labels from wrong model** | Stale `.cache` files after switching detection↔pose | `label_links.py` clears caches automatically |

---

## Requirements

- Python 3.9+ (3.12 recommended)
- See [`requirements.txt`](requirements.txt) for full dependency list
- Quick install: `./setup.sh` (CPU) or `./setup.sh --gpu` (CUDA)

## License

Apache 2.0