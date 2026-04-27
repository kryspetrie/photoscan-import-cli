# Photo Pose Detector

Detect the corners of physical photographs in camera-scanned images, extract each photo, and correct perspective distortion — powered by two custom YOLO models.

## The Problem

When you photograph multiple physical photos laid out on a table (camera scanning), the resulting image contains:

- Multiple photos at different positions and rotations
- Perspective distortion from the camera angle
- Drop shadows and glare from overhead lighting
- Varied table surface backgrounds

The goal: **automatically detect each photo, locate its four corners precisely, and extract a clean perspective-corrected image.**

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
│     4 keypoints per photo  │
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
| **Label format** | 5 columns | 13 columns |
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

### Synthetic Data Generation

Training data is generated synthetically — no manual annotation required. The generator places real photo content onto random backgrounds with realistic distortions and effects.

**Source material:**
- **5,062 source photos** from the Oxford Buildings dataset (architecture, landscapes, objects)
- **6 background textures** for surface simulation

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
| Pose | `class x y w h kp0…kp3` | `0 0.501715 0.505848 0.378018 0.590964 0.312 0.801 2 0.690 0.801 2 0.689 0.210 2 0.314 0.210 2` |

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
        ├── train/          # 13-column keypoint labels
        └── val/
```

**Same images, different labels.** Each model's dataset YAML points to the shared images with its own label directory.

---

## Training

### Setup

The project includes a `requirements.txt` and a `setup.sh` script for one-command environment creation.

```bash
# Quick setup — creates .venv with CPU-only PyTorch
./setup.sh

# With CUDA 11.8 GPU support
./setup.sh --gpu

# With CUDA 12.1 GPU support
./setup.sh --cuda12

# Or install into an already-active venv
source your_env/bin/activate
./setup.sh --existing
```

<details>
<summary>Manual setup (equivalent)</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU PyTorch:
pip install torch==2.2.2 torchvision==0.17.2

# —or— CUDA 11.8 PyTorch:
# pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

</details>

> **Note on existing environments:** The repo currently contains two older virtual environments (`venv/` and `training_env/`) that were created during early development. `venv/` holds the full set of installed packages; `training_env/` is an empty shell that was never populated. Both are gitignored and can be removed once you've migrated to the new `.venv` — but they are preserved for now since active runs may depend on them.

### Train Both Models

```bash
cd training

# Train both sequentially
python train_both.py --epochs 100 --batch 16 --device 0

# Or train individually
python train_detection.py --epochs 100 --batch 16
python train_pose.py --epochs 100 --batch 16
```

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
python export_onnx.py --model ../training/runs/pose/photo-corner-detector/weights/best.pt
```

Produces a ~3–5 MB ONNX model suitable for cross-platform deployment.

### Python Inference (testing)

```bash
cd onnx_inference
python infer.py --model ../models/photo-corner-detector/best.onnx --image ../data/images/val/val_000001.jpg
```

### Kotlin / Android Deployment

The ONNX model integrates into Kotlin using ONNX Runtime + BoofCV:

```kotlin
val detector = PhotoCornerDetector("photo-corner-detector.onnx")
val detected = detector.detect(scannedImage)         // List<DetectedPhoto>
val extracted = detector.extractPhotos(scannedImage)  // Perspective-corrected crops
```

> 📄 Full Kotlin integration guide with Gradle setup, preprocessing, and perspective transform: [`docs/KOTLIN_USAGE.md`](docs/KOTLIN_USAGE.md)

---

## Inference Pipeline (Python)

```python
from ultralytics import YOLO
import cv2
import numpy as np

det_model = YOLO('runs/detection/photo-detector/weights/best.pt')
pose_model = YOLO('runs/pose/photo-corner-detector/weights/best.pt')

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

---

## Project Structure

```
photo-pose-detector/
├── data_generator/
│   ├── generate.py                  # Main generator (examples + batch modes)
│   ├── SYSTEM_DOCUMENTATION.md      # Detailed generator documentation
│   └── images/                      # 5,062 source photos (Oxford Buildings)
│
├── training/
│   ├── dataset_detection.yaml       # Detection dataset config
│   ├── dataset_pose.yaml            # Pose dataset config
│   ├── train_detection.py           # Detection training script
│   ├── train_pose.py               # Pose training script
│   ├── train_both.py               # Train both sequentially
│   └── validate.py                  # Validation script
│
├── export/
│   └── export_onnx.py              # ONNX export script
│
├── onnx_inference/
│   └── infer.py                    # Python ONNX inference testing
│
├── docs/
│   ├── GETTING_STARTED.md          # Setup tutorial
│   ├── KOTLIN_USAGE.md             # Kotlin/Android integration
│   └── PROJECT_PLAN.md             # Original project plan
│
├── textures/                        # 6 background textures
├── data/                            # Generated datasets (gitignored)
├── runs/                            # Training runs (gitignored)
│
├── requirements.txt                 # Python dependency list
├── setup.sh                         # One-command environment setup
├── venv/                            # Legacy venv (gitignored, still in use)
└── training_env/                    # Legacy empty venv (gitignored)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Training loss NaN | Reduce learning rate (`--lr0 0.0001`) |
| Keypoints wrong order | Check `flip_idx: [2, 3, 0, 1]` in dataset YAML |
| Overfitting | Generate more data, reduce augmentation |
| ONNX export fails | `pip install onnxruntime onnx` |
| Slow inference | Use GPU provider, or try smaller input (320×320) |
| Photos overlapping in generated data | Increase `OVERLAP_THRESHOLD` or reduce `NUM_PHOTOS_MAX` |
| Generated photos too small | Increase `PHOTO_SIZE_MIN` in `generate.py` |

---

## Requirements

- Python 3.9+ (3.12 recommended)
- See [`requirements.txt`](requirements.txt) for full dependency list
- Quick install: `./setup.sh` (CPU) or `./setup.sh --gpu` (CUDA)

## License

Apache 2.0