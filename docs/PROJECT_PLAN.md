# Photo Pose Detector Project Plan

## Executive Summary

**Project Name:** photo-pose-detector  
**Purpose:** Train a custom YOLO26-pose model to detect the 4 corners of physical photographs within camera-scanned images, enabling extraction and perspective correction of individual photos.  
**Output Format:** ONNX model for cross-platform Kotlin deployment (via BoofCV/JavaCV)

---

## Problem Statement

When scanning multiple physical photographs placed on a table (camera scanning), images are captured with:
- Perspective distortion
- Variable rotation
- Drop shadows
- Background noise/blur
- Various table surface colors (light/dark)

The goal is to:
1. Detect individual photos within a scanned image
2. Extract each photo as a cropped image
3. Apply perspective correction when distortion exceeds a threshold
4. Output clean, rectangular images

---

## Architecture Overview

### Hybrid Approach (Traditional CV + ML)

```
┌─────────────────────────────────────────────────────────┐
│                    Input Image                           │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Traditional CV (BoofCV/JavaCV)                 │
│  - Edge detection                                        │
│  - Contour analysis                                      │
│  - Quadrilateral detection                               │
│  - Output: Rough bounding boxes for each photo           │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: YOLO26-Pose Corner Detection                   │
│  - Input: Cropped regions from Step 1                   │
│  - Output: 4 keypoints per photo (confidence scores)    │
│  - Keypoints: TopLeft, TopRight, BottomRight, BottomLeft │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Perspective Transform                          │
│  - Calculate homography matrix                           │
│  - Apply perspective correction if distortion > threshold│
│  - Output: Clean, rectangular cropped images             │
└─────────────────────────────────────────────────────────┘
```

### Why This Hybrid Approach?

| Approach | Pros | Cons |
|----------|------|------|
| **Pure ML (OBB Detection)** | Single model | Requires massive training data, overkill for simple geometry |
| **Hybrid (This Project)** | ✅ Efficient, ✅ Precise corners, ✅ Less training data | Requires two-stage pipeline |
| **Pure Traditional CV** | No training needed | Struggles with shadows, occlusions, complex backgrounds |

---

## Technical Approach

### YOLO26-Pose Model Configuration

**Model:** YOLO26n-pose (nano variant - smallest, fastest)  
**Pretrained Weight:** `yolo26n-pose.pt`  
**Keypoint Definition:** 4 corners (quadrilateral pose)

| Keypoint | Index | Description |
|----------|-------|-------------|
| Top-Left | 0 | Upper-left corner of photo |
| Top-Right | 1 | Upper-right corner of photo |
| Bottom-Right | 2 | Lower-right corner of photo |
| Bottom-Left | 3 | Lower-left corner of photo |

### Why YOLO26-Pose?

1. **Designed for keypoint detection** - outputs confidence scores per keypoint
2. **Efficient** - nano variant is ~3MB, suitable for mobile/embedded
3. **Pretrained on COCO** - good feature extraction foundation
4. **Easy to fine-tune** - Ultralytics provides excellent training pipeline
5. **ONNX export** - native support for cross-platform deployment

---

## Project Structure

```
photo-pose-detector/
├── data_generator/          # Synthetic training data generation
│   ├── generate_dataset.py  # Main data generator
│   ├── scene_renderer.py    # Renders synthetic photos on tables
│   ├── keypoint_annotator.py # Generates YOLO-format annotations
│   └── config.py            # Generation configuration
├── training/                # Model training scripts
│   ├── train.py             # Training script
│   ├── dataset.yaml         # YOLO dataset configuration
│   ├── hyperparameters.yaml  # Training hyperparameters
│   └── validate.py          # Validation script
├── export/                  # Model export scripts
│   └── export_onnx.py       # ONNX export script
├── onnx_inference/          # Python inference testing
│   ├── infer.py             # Inference script
│   └── postprocess.py       # Keypoint extraction utilities
├── kotlin_integration/      # Kotlin/Java integration
│   ├── src/main/kotlin/     # Kotlin source files
│   ├── build.gradle.kts     # Gradle build configuration
│   └── src/test/kotlin/     # Unit tests
├── docs/                    # Documentation
│   ├── PROJECT_PLAN.md      # This file
│   ├── GETTING_STARTED.md   # Step-by-step tutorial
│   ├── TRAINING_DATA.md     # Data generation details
│   └── KOTLIN_USAGE.md      # Kotlin integration guide
├── data/                    # Generated training data (gitignored)
│   ├── images/train/
│   ├── images/val/
│   └── labels/
└── models/                  # Trained models (gitignored)
    └── photo-corner-detector/
```

---

## Training Data Generation Strategy

### Synthetic Data Generation

Rather than manually annotating thousands of images, we generate synthetic training data:

1. **Background Tables:** Random textures (wood, marble, fabric, solid colors)
2. **Photo Placement:** Random position, rotation, scale, perspective
3. **Photo Content:** Random images from public datasets (COCO, ImageNet samples)
4. **Augmentations:**
   - Drop shadows (random position, blur, opacity)
   - Motion blur (direction, amount)
   - Gaussian noise
   - Color temperature variations
   - Vignette effects

### Ground Truth Generation

For each synthetic scene:
```
Image: scene_0001.png
Keypoints (YOLO pose format):
  0 0.234 0.156 0.95   # Top-Left (x_norm, y_norm, confidence)
  1 0.789 0.178 0.97   # Top-Right
  2 0.812 0.891 0.93   # Bottom-Right
  3 0.201 0.845 0.91   # Bottom-Left
```

### Dataset Size Recommendation

| Dataset Size | Use Case | Training Time |
|-------------|----------|---------------|
| ~500 images | Quick testing | ~30 min |
| ~2,000 images | Development | ~2 hours |
| ~5,000 images | Production | ~5 hours |

---

## Training Configuration

### Dataset YAML (`dataset.yaml`)

```yaml
# YOLO Pose Dataset Configuration
path: ../data
train: images/train
val: images/val

# Keypoint definitions (4 corners)
kpt_shape: [4, 2]  # 4 keypoints, each with (x, y)
flip_idx: [1, 0, 3, 2]  # Left-right flip mapping

# Number of classes (1 = photo, 0 = background)
names:
  0: photo_corner

# Pose configuration
pose:
  flip_idx: [1, 0, 3, 2]
```

### Training Command

```bash
# Python API
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device=0  # GPU, or 'cpu'
)
```

### Recommended Hyperparameters

Based on YOLO26 fine-tuning guidelines:

```yaml
# Reduced augmentation for small custom dataset
augment: true
mosaic: 0.5      # Reduce from default 0.9
mixup: 0.0       # Disable for pose detection
copy_paste: 0.0  # Disable
scale: 0.5       # Reduce scale variation
degrees: 10      # Allow some rotation
translate: 0.1   # Allow position variation

# Learning rate
lr0: 0.001       # Lower than pretraining
lrf: 0.01        # Final learning rate factor

# Training duration
epochs: 100
patience: 20     # Early stopping
```

---

## ONNX Export

### Export Command

```bash
# Python API
model = YOLO("runs/pose/train/weights/best.pt")
model.export(format="onnx", dynamic=True)
```

### Export Arguments for ONNX

| Argument | Value | Reason |
|----------|-------|--------|
| `format` | `"onnx"` | Cross-platform compatibility |
| `dynamic` | `True` | Handle varying input sizes |
| `simplify` | `True` | Optimize graph (default) |
| `opset` | `12` | Good compatibility |
| `nms` | `False` | Handle NMS in Kotlin |

### Expected Output

```
photo-corner-detector.onnx  # ~3-5 MB
```

### ONNX Output Tensor Shape

For pose model with 4 keypoints:
```
Output: (batch_size, 4 + num_classes + keypoint_dims, num_predictions)
       = (1, 4 + 1 + 3, 8400)  # 4 box coords + 1 class + (x,y,conf)*4 keypoints
```

---

## Kotlin Integration

### Dependencies

```kotlin
// build.gradle.kts
dependencies {
    // ONNX Runtime for cross-platform inference
    implementation("org.onnxruntime:onnxruntime:1.17.0")
    
    // BoofCV for image processing and perspective transforms
    implementation("boofcv:boofcv-core:0.40")
    implementation("boofcv:boofcv-android:0.40")
    
    // JavaCV for image loading
    implementation("org.bytedeco:javacv-platform:1.5.9")
}
```

### Inference Pipeline in Kotlin

```kotlin
class PhotoCornerDetector(private val modelPath: String) {
    
    // 1. Preprocess: Resize to 640x640, normalize to [0,1]
    fun preprocess(image: BufferedImage): FloatArray { ... }
    
    // 2. Run ONNX inference
    fun detect(image: BufferedImage): List<DetectedPhoto> { ... }
    
    // 3. Postprocess: Extract keypoints, filter by confidence
    fun extractCorners(output: FloatArray): List<Quadrilateral> { ... }
    
    // 4. Apply perspective transform if needed
    fun extractAndCorrect(image: BufferedImage, corners: Quadrilateral): BufferedImage { ... }
}

data class DetectedPhoto(
    val confidence: Float,
    val corners: List<Point2D>,  // 4 corners in image coordinates
    val correctedImage: BufferedImage?  // After perspective transform
)
```

### BoofCV Integration for Perspective Transform

```kotlin
import boofcv.alg.distort.DistortImageOps
import boofcv.alg.interpolate.InterpolateType
import georegression.struct.homography.ModelHomography2D_F32
import georegression.struct.point.Point2D_F32

fun applyPerspectiveTransform(
    image: BufferedImage,
    sourceCorners: List<Point2D_F32>,
    destSize: Dimension
): BufferedImage {
    // Compute homography from source to destination rectangle
    val destCorners = listOf(
        Point2D_F32(0f, 0f),
        Point2D_F32(destSize.width.toFloat(), 0f),
        Point2D_F32(destSize.width.toFloat(), destSize.height.toFloat()),
        Point2D_F32(0f, destSize.height.toFloat())
    )
    
    val homography = estimatedHomography(sourceCorners, destCorners)
    
    // Apply transform
    return distortImage(image, homography, destSize)
}
```

---

## Step-by-Step Implementation Plan

### Phase 1: Project Setup (Day 1)
- [ ] Create project directory structure
- [ ] Set up Python virtual environment with ultralytics
- [ ] Verify GPU/CUDA availability (optional but recommended)
- [ ] Test pretrained YOLO26-pose model inference

### Phase 2: Data Generation (Day 1-2)
- [ ] Implement scene renderer with random backgrounds
- [ ] Implement photo placement with perspective simulation
- [ ] Add shadow/noise/blur augmentations
- [ ] Generate YOLO-format annotations
- [ ] Generate initial dataset (500-1000 images)
- [ ] Inspect annotations visually for correctness

### Phase 3: Training (Day 2-3)
- [ ] Create dataset.yaml configuration
- [ ] Adjust hyperparameters for small dataset
- [ ] Run initial training (50 epochs)
- [ ] Evaluate validation metrics
- [ ] Tune and retrain as needed
- [ ] Export best model to ONNX

### Phase 4: Python Validation (Day 3)
- [ ] Test ONNX inference on held-out images
- [ ] Verify keypoint accuracy
- [ ] Test perspective transform pipeline
- [ ] Measure inference time

### Phase 5: Kotlin Integration (Day 4-5)
- [ ] Set up Kotlin project with Gradle
- [ ] Add ONNX Runtime dependency
- [ ] Implement pre/post-processing in Kotlin
- [ ] Integrate with BoofCV for transforms
- [ ] Write unit tests
- [ ] Build and verify Android/desktop compatibility

### Phase 6: Polish & Documentation (Day 5)
- [ ] Update getting-started guide
- [ ] Document common issues and solutions
- [ ] Create example usage code
- [ ] Verify end-to-end pipeline

---

## Verification Checklist

### Data Generation Verification
- [ ] Generated images show photos with visible perspective
- [ ] Keypoint annotations are visually accurate
- [ ] Augmentations are applied (shadows, blur visible)
- [ ] Train/val split is reasonable (80/20)

### Training Verification
- [ ] Training loss decreases over epochs
- [ ] Validation metrics improve
- [ ] Overfitting is controlled (train/val gap < 10%)
- [ ] Best model checkpoint saved

### ONNX Export Verification
- [ ] ONNX file is generated (~3-5 MB)
- [ ] ONNX model loads without errors
- [ ] Inference output matches PyTorch output
- [ ] Dynamic input size works

### Kotlin Integration Verification
- [ ] Gradle build succeeds
- [ ] ONNX Runtime loads model
- [ ] Inference produces reasonable results
- [ ] Perspective transform produces clean output
- [ ] No memory leaks on repeated inference

---

## Troubleshooting Guide

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Training loss NaN | Learning rate too high | Reduce lr0 to 0.0001 |
| Keypoints all in center | Annotations incorrect | Verify annotation generation |
| Model overfits | Too few training images | Generate more data |
| ONNX export fails | Missing onnxruntime | `pip install onnxruntime` |
| Kotlin OOM | Large images not resized | Always resize to model input size |

---

## References

- [Ultralytics YOLO26 Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [YOLO26 Training Recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)
- [Model Export Guide](https://docs.ultralytics.com/modes/export/)
- [BoofCV Perspective Transform](https://boofcv.org/)
- [ONNX Runtime Kotlin](https://onnxruntime.ai/docs/api/kotlin/)

---

## License

This project uses Apache 2.0 license. Training data generation may use images from COCO/ImageNet under their respective licenses.
