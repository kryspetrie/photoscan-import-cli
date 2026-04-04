# Photo Pose Detector

Train a custom YOLO26-pose model to detect the 4 corners of physical photographs within camera-scanned images, enabling extraction and perspective correction of individual photos.

## Overview

This project trains a machine learning model to detect the corners of photographs within scanned images (photos laid out on a table, captured from above). The model outputs 4 keypoints per detected photo:

- **Top-Left** (keypoint 0)
- **Top-Right** (keypoint 1)
- **Bottom-Right** (keypoint 2)
- **Bottom-Left** (keypoint 3)

These keypoints enable:
1. Precise photo extraction with quadrilateral crops
2. Perspective correction when photos are tilted or skewed
3. Hybrid pipeline with traditional CV for rough detection + ML for precise corners

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Image                           │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Traditional CV (BoofCV/JavaCV)                 │
│  - Edge detection → Contour analysis → Rough bounding   │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: YOLO26-Pose Corner Detection                  │
│  - Input: Cropped regions from Step 1                  │
│  - Output: 4 keypoints per photo (confidence scores)   │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Perspective Transform                          │
│  - Apply perspective correction if skew > threshold    │
│  - Output: Clean, rectangular cropped images           │
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
pip install ultralytics torch torchvision numpy pillow opencv-python
pip install onnx onnxruntime
```

### 2. Generate Training Data

```bash
cd data_generator
python generate_dataset.py --num-train 800 --num-val 200
```

This generates synthetic images of photos on tables with:
- Random backgrounds (wood, marble, solid colors)
- Random photo positions and rotations
- Drop shadows, blur, and noise effects
- Ground truth keypoint annotations

### 3. Train the Model

```bash
cd ../training
python train.py --epochs 100 --batch 16 --device 0
```

For CPU-only training:
```bash
python train.py --epochs 100 --device cpu
```

### 4. Export to ONNX

```bash
cd ../export
python export_onnx.py --model ../runs/pose/photo-corner-detector/weights/best.pt
```

### 5. Test Inference

```bash
cd ../onnx_inference
python infer.py --model ../models/photo-corner-detector/best.onnx
```

## Project Structure

```
photo-pose-detector/
├── data_generator/          # Synthetic training data generation
│   ├── generate_dataset.py   # Main data generator
│   └── ...
├── training/                 # Model training scripts
│   ├── train.py              # Training script
│   ├── validate.py           # Validation script
│   └── dataset.yaml          # Dataset configuration (generated)
├── export/                   # Model export scripts
│   └── export_onnx.py       # ONNX export script
├── onnx_inference/          # Python inference testing
│   └── infer.py             # Inference script
├── kotlin_integration/      # Kotlin/Java integration (template)
│   └── src/main/kotlin/
├── docs/                    # Documentation
│   ├── PROJECT_PLAN.md      # Technical plan
│   ├── GETTING_STARTED.md   # Detailed tutorial
│   └── KOTLIN_USAGE.md      # Kotlin integration guide
├── data/                    # Generated training data
│   ├── images/train/
│   ├── images/val/
│   └── labels/
└── models/                  # Trained models
    └── photo-corner-detector/
```

## Documentation

| Document | Description |
|----------|-------------|
| [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Technical plan and architecture overview |
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Step-by-step tutorial |
| [KOTLIN_USAGE.md](docs/KOTLIN_USAGE.md) | Kotlin/Android integration guide |

## Model Details

### YOLO26-Pose Configuration

- **Base Model:** YOLO26n-pose (nano variant)
- **Pretrained on:** COCO keypoints
- **Custom Training:** Fine-tuned on synthetic photo corner data
- **Keypoints:** 4 corners per detected photo
- **Output Format:** ONNX for cross-platform deployment

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| epochs | 100 | Training epochs |
| batch | 16 | Batch size |
| imgsz | 640 | Input image size |
| lr0 | 0.001 | Initial learning rate |
| mosaic | 0.5 | Mosaic augmentation |
| degrees | 10.0 | Rotation range |
| patience | 20 | Early stopping |

### Expected Performance

| Metric | Value | Description |
|--------|-------|-------------|
| mAP50 | ~0.85 | Mean AP at IoU=0.50 |
| mAP50-95 | ~0.65 | Mean AP across IoU thresholds |
| Keypoint Accuracy | ~95% | Corner localization accuracy |
| Inference Time | ~50ms | CPU inference (640x640) |

## Kotlin Integration

See [KOTLIN_USAGE.md](docs/KOTLIN_USAGE.md) for complete integration guide.

### Dependencies

```kotlin
// build.gradle.kts
dependencies {
    implementation("org.onnxruntime:onnxruntime:1.17.0")
    implementation("boofcv:boofcv-core:0.40")
    implementation("org.bytedeco:javacv-platform:1.5.9")
}
```

### Basic Usage

```kotlin
val detector = PhotoCornerDetector("models/photo-corner-detector.onnx")

// Detect photos
val detections = detector.detect(image)

// Extract with perspective correction
val extracted = detector.extractPhotos(
    image = image,
    outputDir = "output/",
    applyPerspectiveCorrection = true
)
```

## Troubleshooting

### Training Issues

**Loss is NaN**
- Reduce learning rate: `--lr0 0.0001`
- Increase batch size

**Model overfitting**
- Reduce augmentation: `--mosaic 0.3`
- Generate more training data
- Use early stopping (patience=30)

**Low keypoint accuracy**
- Verify annotations are correct
- Increase training data
- Train for more epochs

### Export Issues

**ONNX export fails**
- Install onnxruntime: `pip install onnxruntime`
- Check model file is valid

**Inference produces wrong output**
- Verify output tensor shape matches model
- Check preprocessing (CHW format, normalization)

## Requirements

### Software

- Python 3.9+
- CUDA 11.8+ (optional, for GPU training)
- Java 17+ (for Kotlin integration)

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | None | 4GB+ |
| RAM | 8GB | 16GB |
| Storage | 10GB | 20GB |

## References

- [Ultralytics YOLO26 Pose](https://docs.ultralytics.com/tasks/pose/)
- [YOLO26 Training Recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)
- [Model Export](https://docs.ultralytics.com/modes/export/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [BoofCV](https://boofcv.org/)

## License

Apache 2.0
