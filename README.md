# Photo Pose Detector

Train a custom YOLO26-pose model to detect the 4 corners of physical photographs within camera-scanned images, enabling extraction and perspective correction of individual photos.

## Overview

This project trains a machine learning model to detect the corners of photographs within scanned images (photos laid out on a table, captured from above). The model outputs 4 keypoints per detected photo:

- **Keypoint 0:** Top-Left corner
- **Keypoint 1:** Top-Right corner
- **Keypoint 2:** Bottom-Right corner
- **Keypoint 3:** Bottom-Left corner

These keypoints enable:
1. Precise photo extraction with quadrilateral crops
2. Perspective correction when photos are tilted or skewed
3. Hybrid pipeline with traditional CV for rough detection + ML for precise corners

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Image                           │
│          (Photos on table, captured from above)          │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Traditional CV (Edge detection, Contours)      │
│  - Rough bounding box detection                          │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: YOLO26-Pose Corner Detection (ML)              │
│  - Input: Cropped regions from Step 1                   │
│  - Output: 4 keypoints per photo (confidence scores)    │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Perspective Transform                          │
│  - Apply perspective correction if skew > threshold      │
│  - Output: Clean, rectangular cropped images             │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install ultralytics torch torchvision numpy opencv-python pillow
pip install onnx onnxruntime
```

### 2. Generate Training Data

```bash
cd data_generator
python generate_dataset.py
```

This generates synthetic images with realistic photo arrangements:
- **5-10 photos per frame** arranged on a table surface
- **Random backgrounds**: dark grey (30%), light grey (30%), colored (40%)
- **Random rotations** (±30°) simulating handheld photos
- **Drop shadows** with blur simulating depth
- **Glare effects** simulating lighting
- **Soft edges** simulating imperfect camera focus
- **Global perspective warp** simulating camera angle

### 3. Train the Model

```bash
cd ../training
python train.py --epochs 100 --batch 16 --device 0
```

### 4. Export to ONNX

```bash
cd ../export
python export_onnx.py --model ../runs/pose/weights/best.pt
```

## Project Structure

```
photo-pose-detector/
├── data_generator/              # Synthetic training data generation
│   ├── generate_dataset.py       # Main data generator (v31)
│   └── images/                   # Source photos (5000+ images)
├── training/                    # Model training scripts
├── export/                      # ONNX export
├── onnx_inference/              # Python inference testing
├── kotlin_integration/          # Kotlin/Android integration
├── docs/                        # Documentation
├── data/                        # Generated training data
│   ├── examples_v31/            # Example outputs
│   └── ...                      # Full datasets
└── models/                      # Trained models
```

---

## Data Generator Documentation

### Overview

The `generate_dataset.py` script creates synthetic training images that simulate real-world scanned photos. It implements a **single global perspective warp** architecture:

```
1. Pack FLAT photos on a flat background (with rotation)
2. Apply ONE perspective warp to the ENTIRE composite
3. Crop to content bounds
```

This architecture ensures:
- Photos start as perfect rectangles (ground truth = 4 corners)
- Global warp distorts all photos uniformly
- Ground truth corners track correctly through the warp

### Key Features

#### Background Generation
- **30% Dark backgrounds**: Very low saturation, brightness 10-72/255
- **30% Light backgrounds**: Very low saturation, brightness 176-245/255
- **40% Colored backgrounds**: Reduced saturation (20% lower), full brightness range

Each background includes:
- Solid base color with subtle noise (σ=1-4)
- 3 linear gradients with screen blend (0-20% opacity)
- Real texture overlay (multiply or screen blend)

#### Photo Packing
- **Spiral packing** from center outward
- **Pixel-based collision** detection with dilated masks (~20px gap)
- Photos placed without overlap
- Random rotation: ±30° (10% of photos)

#### Photo Effects
- **Soft edges**: 0-5px Gaussian blur on all edges (simulates focus)
- **Drop shadows**: Offset 0-15px, blur σ=6-15, opacity 15-60%
- **Glare**: Large elliptical flares (20-40% of photo), opacity 60-100%
- **Noise**: Subtle photo manipulation (brightness/contrast/colors)

#### Global Perspective
- Subtle perspective warp (2-5% strength) applied once to entire composite
- Simulates camera viewing table at an angle
- All photos distorted uniformly

### Output Format

#### YOLO26-Pose Format

Images: Standard JPEG/PNG (e.g., `image_001.jpg`)
Labels: One `.txt` file per image with one line per photo:

```
<class_id> <x_center> <y_center> <width> <height> <kx0> <ky0> <kc0> <kx1> <ky1> <kc1> <kx2> <ky2> <kc2> <kx3> <ky3> <kc3>
```

Where:
- `class_id`: 0 (single class for "photo")
- `x_center, y_center`: Bounding box center (normalized 0-1)
- `width, height`: Bounding box size (normalized 0-1)
- `kx0-3, ky0-3`: Keypoint coordinates (normalized 0-1)
- `kc0-3`: Keypoint confidence (1.0 for synthetic data)

### Configuration

Key parameters in `generate_dataset.py`:

| Parameter | Default | Description |
|-----------|--------|-------------|
| `CANVAS_W` | 3000 | Canvas width in pixels |
| `CANVAS_H` | 1800 | Canvas height in pixels |
| `NUM_PHOTOS` | 5-10 | Photos per frame |
| `ROTATION_RANGE` | ±30° | Max rotation angle |
| `EDGE_BLUR` | 0-5px | Soft edge blur amount |
| `SHADOW_OFFSET` | 0-15px | Shadow offset range |
| `SHADOW_BLUR` | 6-15 | Shadow blur sigma |
| `GLARE_OPACITY` | 60-100% | Glare intensity |

### Running Examples

```bash
# Generate 10 example images
python generate_dataset.py

# Images saved to ../data/examples_v31/
```

### Extending the Generator

#### Adding New Background Types

Edit `random_base_background()` function:

```python
def random_base_background(w, h):
    """Generate background with custom distribution."""
    rand_val = random.random()
    
    if rand_val < 0.30:  # Dark
        lightness = random.uniform(0.04, 0.28)
        saturation = random.uniform(0, 0.04)
    elif rand_val < 0.60:  # Light
        lightness = random.uniform(0.69, 0.96)
        saturation = random.uniform(0, 0.04)
    else:  # Colored
        lightness = random.uniform(0.19, 0.86)
        saturation = random.uniform(0.04, 0.40)
```

#### Adding New Photo Effects

Edit `fast_photo_manipulation()` function:

```python
def fast_photo_manipulation(img):
    """Add effects to individual photos."""
    # Brightness/contrast adjustments
    # Color shifts
    # Noise
    # etc.
```

## Model Training

### YOLO26-Pose Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | YOLO26n-pose (nano) |
| Input Size | 640x640 |
| Keypoints | 4 (photo corners) |
| Classes | 1 (photo) |

### Training Hyperparameters

```bash
python train.py \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --lr0 0.001 \
    --patience 20
```

### Expected Performance

| Metric | Target |
|--------|--------|
| mAP50 | >0.85 |
| mAP50-95 | >0.65 |
| Keypoint Accuracy | >95% |
| Inference Time | <50ms (CPU) |

## Kotlin Integration

```kotlin
val detector = PhotoCornerDetector("model.onnx")
val detections = detector.detect(image)

// Get corner coordinates
val topLeft = detections[0].keypoints[0]
val topRight = detections[0].keypoints[1]
val bottomRight = detections[0].keypoints[2]
val bottomLeft = detections[0].keypoints[3]

// Extract with perspective correction
val extracted = detector.extractPhoto(image, corners)
```

## Troubleshooting

### Training Issues

| Problem | Solution |
|---------|----------|
| Loss is NaN | Reduce learning rate (--lr0 0.0001) |
| Overfitting | Reduce augmentation, add more data |
| Low accuracy | Verify annotations, train longer |

### Generation Issues

| Problem | Solution |
|---------|----------|
| Photos overlapping | Increase collision dilation |
| Black borders | Adjust perspective warp strength |
| Slow generation | Reduce canvas size or photo count |

## Requirements

- Python 3.9+
- OpenCV 4.x
- NumPy
- Ultralytics
- CUDA (optional, for GPU training)

## References

- [Ultralytics YOLO26 Pose](https://docs.ultralytics.com/tasks/pose/)
- [YOLO26 Training Recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)
- [ONNX Runtime](https://onnxruntime.ai/)

## License

Apache 2.0
