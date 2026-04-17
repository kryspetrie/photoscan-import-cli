# Photo Pose Detector - Two-Model Architecture

Train custom YOLO models to detect the corners of physical photographs within camera-scanned images, enabling extraction and perspective correction of individual photos.

## Overview

This project trains **TWO separate YOLO models** that work together:

| Model | Output | Purpose |
|-------|--------|---------|
| **Detection Model** | Axis-aligned bounding box | Find where photos are located |
| **Pose Model** | 4 corner keypoints (LL, UL, UR, LR) | Detect precise corner locations |

From the detected corners, you can:
1. Extract the quadrilateral photo region
2. Apply perspective correction to straighten the photo
3. Create a hybrid pipeline with traditional CV + ML

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Image                           │
│          (Photos on table, captured from above)          │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Detection Model (YOLO)                         │
│  - Output: Axis-aligned bounding boxes                  │
│  - Purpose: Find where photos are located              │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Pose Model (YOLO-Pose)                         │
│  - Output: 4 corner keypoints per photo                  │
│  - kp0=LL, kp1=UL, kp2=UR, kp3=LR                       │
│  - Purpose: Precise corner locations for extraction     │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Photo Extraction                               │
│  - Build quadrilateral from corner keypoints            │
│  - Apply perspective transform                          │
│  - Output: Clean, rectangular cropped images            │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install ultralytics torch torchvision numpy opencv-python pillow
```

### 2. Generate Training Data

```bash
cd data_generator
python generate_batch.py > generation_log.txt 2>&1 &
```

This generates 5000 images with **BOTH** label formats:
- Detection labels (5 columns): bounding boxes
- Pose labels (13 columns): corner keypoints

### 3. Train Both Models

```bash
cd ../training

# Train both models
python train_both.py --epochs 100 --batch 16 --device 0

# Or train individually
python train_detection.py --epochs 100 --device 0
python train_pose.py --epochs 100 --device 0
```

## Two-Model Architecture

### Model 1: Detection Model

**Type:** Standard YOLO Object Detection

**Output Format:** 5 columns per object
```
<class_id> <x_center> <y_center> <width> <height>
```

**Use Cases:**
- Initial photo detection in a pipeline
- Fast approximate localization
- Filter/reduce search space before pose detection

**Training:**
```bash
python train_detection.py --epochs 100 --batch 16
```

---

### Model 2: Pose Model

**Type:** YOLO-Pose Keypoint Detection

**Output Format:** 13 columns per object
```
<class_id> <x_center> <y_center> <width> <height> 
<kx0> <ky0> <kc0> <kx1> <ky1> <kc1> <kx2> <ky2> <kc2> <kx3> <ky3> <kc3>
```

**Keypoint Order (CRITICAL):**
| Keypoint | Name | Description |
|----------|------|-------------|
| kp0 | LL | Lower-Left (minimum Y, minimum X) |
| kp1 | UL | Upper-Left (maximum Y, minimum X) |
| kp2 | UR | Upper-Right (maximum Y, maximum X) |
| kp3 | LR | Lower-Right (minimum Y, maximum X) |

**Horizontal Flip Augmentation:**
When the image is horizontally flipped, keypoints are swapped:
- LL (kp0) ↔ LR (kp3)
- UL (kp1) ↔ UR (kp2)

This is configured via `flip_idx: [2, 3, 0, 1]` in the dataset YAML.

**Use Cases:**
- Precise corner detection
- Photo extraction with quadrilateral crops
- Perspective correction

**Training:**
```bash
python train_pose.py --epochs 100 --batch 16
```

---

## Data Generation (v32)

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Canvas Size | 640×640 | Matches YOLO input size |
| Photos per Image | 1-4 | Clean single-image training |
| Photo Size | 60-90% of canvas | Photos fill most of frame |
| Perspective | 5-15% | **More extreme than before** |
| Rotation | ±30° | Random photo rotation |
| Edge Blur | 0-5px | Soft focus simulation |
| Shadow Offset | 0-5px | Subtle drop shadows |
| Background | 30% dark, 30% light, 40% color | Varied environments |

### Key Features

1. **Single Global Perspective Warp**
   - Applied once to entire composite
   - Simulates camera viewing table at angle
   - All photos distorted uniformly
   - Ground truth corners track correctly

2. **More Extreme Perspective**
   - 5-15% strength (was 2-5%)
   - More realistic for handheld captures
   - Greater corner variation for ML

3. **Soft Photo Edges**
   - 0-5px Gaussian blur on all edges
   - Simulates imperfect camera focus
   - Prevents overfitting to sharp edges

4. **Drop Shadows**
   - Scaled for 640x640 canvas
   - Rotates with photo
   - Adds depth realism

5. **Glare Effects**
   - Large elliptical flares
   - Screen blend mode
   - Simulates lighting

### Output Structure

```
data/
├── images/
│   ├── train/              # 4000 JPEG images (shared)
│   └── val/                # 1000 JPEG images
│
├── detection/              # DETECTION MODEL DATA
│   └── labels/
│       ├── train/          # 4000 labels (5 columns)
│       └── val/            # 1000 labels
│
└── pose/                   # POSE MODEL DATA
    └── labels/
        ├── train/          # 4000 labels (13 columns)
        └── val/            # 1000 labels
```

**Same images, different labels.**

---

## Training

### Dataset YAMLs

**Detection (`dataset_detection.yaml`):**
```yaml
path: /path/to/data
train: images/train
val: images/val
nc: 1
names:
  0: photo
```

**Pose (`dataset_pose.yaml`):**
```yaml
path: /path/to/data
train: images/train
val: images/val
nc: 1
names:
  0: photo
kpt_number: 4
kpt_shape: [4, 2]
flip_idx: [2, 3, 0, 1]
keypoint_names:
  - lower_left
  - upper_left
  - upper_right
  - lower_right
```

### Training Scripts

| Script | Purpose |
|--------|---------|
| `train_detection.py` | Train detection model |
| `train_pose.py` | Train pose model |
| `train_both.py` | Train both models sequentially |

### Recommended Hyperparameters

| Parameter | Detection | Pose |
|-----------|-----------|------|
| Model | yolo11n.pt | yolo26n-pose.pt |
| Epochs | 100 | 100 |
| Batch Size | 16 | 16 |
| Image Size | 640 | 640 |
| Mosaic | 0.5 | 0.5 |
| Degrees | 10 | 10 |
| Scale | 0.5 | 0.3 |
| Mixup | 0.1 | 0.0 |
| Copy-paste | 0.1 | 0.0 |
| Flip LR | 0.5 | 0.5 |
| Flip UD | 0.0 | 0.0 |

---

## Inference Pipeline

```python
from ultralytics import YOLO
import cv2
import numpy as np

# Load models
det_model = YOLO('runs/detection/photo-detector/weights/best.pt')
pose_model = YOLO('runs/pose/photo-corner-detector/weights/best.pt')

# Read image
image = cv2.imread('scanned_photo.jpg')

# Step 1: Detection
det_results = det_model.predict(image, verbose=False)
boxes = det_results[0].boxes

# Step 2: Pose on each detected region
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    region = image[y1:y2, x1:x2]
    
    pose_results = pose_model.predict(region, verbose=False)
    keypoints = pose_results[0].keypoints
    
    # Get corner coordinates
    # kp0=LL, kp1=UL, kp2=UR, kp3=LR
    corners = keypoints.xy[0]  # Shape: (4, 2)
    
    # Map back to original image coordinates
    offset_corners = corners.cpu().numpy()
    offset_corners[:, 0] += x1
    offset_corners[:, 1] += y1
    
    # Extract photo using perspective transform
    src_pts = offset_corners.astype(np.float32)
    # Order: LL, UL, UR, LR
    width = int(max(np.linalg.norm(src_pts[3] - src_pts[0]),
                    np.linalg.norm(src_pts[2] - src_pts[1])))
    height = int(max(np.linalg.norm(src_pts[1] - src_pts[0]),
                     np.linalg.norm(src_pts[2] - src_pts[3])))
    
    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], 
                       dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    extracted = cv2.warpPerspective(image, M, (width, height))
    
    # Save extracted photo
    cv2.imwrite('extracted_photo.jpg', extracted)
```

---

## Project Structure

```
photo-pose-detector/
├── data_generator/
│   ├── generate_dataset.py     # Example generator (v32)
│   ├── generate_batch.py      # Batch dataset generator (v32)
│   └── images/                # Source photos
│
├── training/
│   ├── dataset_detection.yaml  # Detection dataset config
│   ├── dataset_pose.yaml       # Pose dataset config
│   ├── train_detection.py      # Detection training script
│   ├── train_pose.py          # Pose training script
│   └── train_both.py          # Train both models
│
├── data/
│   ├── images/                # Generated images
│   ├── detection/labels/      # Detection labels
│   └── pose/labels/           # Pose labels
│
├── textures/                   # Background textures
│
├── runs/
│   ├── detection/            # Detection training runs
│   └── pose/                 # Pose training runs
│
└── models/                    # Trained models
```

---

## Troubleshooting

### Training Issues

| Problem | Solution |
|---------|----------|
| Loss is NaN | Reduce learning rate (`--lr0 0.0001`) |
| Overfitting | Reduce augmentation, add more data |
| Low accuracy | Verify annotations, train longer |
| Keypoints wrong order | Check `flip_idx` in YAML |

### Generation Issues

| Problem | Solution |
|---------|----------|
| Photos overlapping | Increase collision margin |
| Black borders | Adjust perspective warp strength |
| Slow generation | Reduce canvas size or photo count |
| Too extreme perspective | Lower `PERSPECTIVE_STRENGTH_MAX` |

### Label Format Verification

**Detection (5 columns):**
```
0 0.501715 0.505848 0.378018 0.590964
```

**Pose (13 columns + keypoint visibility):**
```
0 0.501715 0.505848 0.378018 0.590964 0.312194 0.801330 2 0.690212 0.801330 2 0.689914 0.210366 2 0.314542 0.210366 2
```

---

## Version History

### v32 (Current)
- **Two-model architecture**: Detection + Pose
- **640x640 canvas**: Matches YOLO input
- **1-4 photos per image**: Cleaner training
- **More extreme perspective**: 5-15% strength
- **Fixed keypoint order**: LL, UL, UR, LR
- **Same images, different labels**

### v31 (Previous)
- Single YOLO-pose model
- 1920x1080 canvas
- 5-10 photos per image
- Subtle perspective: 2-5%
- Keypoint order: TL, TR, BR, BL

---

## Requirements

- Python 3.9+
- OpenCV 4.x
- NumPy
- Ultralytics 8.x
- PyTorch 2.x
- PIL/Pillow

## License

Apache 2.0
