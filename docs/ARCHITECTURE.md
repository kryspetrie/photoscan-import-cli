# Architecture

## Project Overview

**photo-pose-detector** detects physical photographs in camera-scanned images, locates their four corners, and extracts perspective-corrected crops. The pipeline runs detection to find photos within the scene, pose estimation to pinpoint corners, and geometric correction to produce final output.

## Model Inventory

### 1. Detection Model (`detection_ep47`) ★ CRITICAL PRODUCTION MODEL

- **Architecture**: YOLO detection model (yolo26s)
- **Purpose**: Finds bounding boxes of photos in full-scene images
- **Training**: `training/train_detection.py`
- **Data**: `data_detection/`
- **Generator**: `data_generator/generate_detection.py`
- **Dataset config**: `training/dataset_detection.yaml`
- **Best results** (epoch 47): mAP50=0.995, mAP50-95=0.918

### 2. Pose Model (`pose_single_ep42`) ★ CRITICAL PRODUCTION MODEL

- **Architecture**: YOLO pose model (yolo26s-pose)
- **Purpose**: Detects 4 corner keypoints per photo in tightly-cropped images
- **Keypoints**: kp0=LL, kp1=UL, kp2=UR, kp3=LR (`flip_idx=[3,2,1,0]`)
- **Training**: `training/train_pose.py`
- **Data**: `data_pose/`
- **Generator**: `data_generator/generate_pose.py`
- **Dataset config**: `training/dataset_pose.yaml`
- **Best results** (epoch 42): mAP50=0.995

> ⚠️ This model is the foundation of the entire pipeline — it must not be removed.

### 3. Corner Regression Model (`corner-regression-v2`) ★ PRODUCTION

- **Architecture**: YOLO pose model (yolo26n-pose, 1 class + 1 keypoint)
- **Purpose**: Detects precise corner positions in 320×320 crops centered on approximate corners. Used by `--corner-refine` flag to recover invisible corners and improve precision.
- **Training data**: 10K train + 2K val synthetic corner crops (20-25% negative/background samples)
- **Training**: `training/train_corner_regression.py`
- **Data**: `data_corner_regression/` (gitignored, regenerateable)
- **Generator**: `data_generator/generate_corner_regression.py`
- **Dataset config**: `training/dataset_corner_regression.yaml`
- **Best results** (epoch 24): Box mAP50=0.994, Box mAP50-95=0.773, Pose mAP50=0.994, Pose mAP50-95=0.994

### 4. Fiducial-Pose Segment Model (IN DEVELOPMENT)

- **Architecture**: YOLO pose model for detecting individual visible edge segments
- **Purpose**: Each segment has keypoints at its endpoints (the visible corners of that segment). Goal is to better handle invisible/occluded corners by assembling segments geometrically.
- **Training**: `training/train_fiducial_pose.py`
- **Data**: `data_fiducial_pose/`
- **Generator**: `data_generator/generate_fiducial_pose.py`
- **Dataset config**: `training/dataset_fiducial_pose.yaml`
- **Best results** (V3, epoch 18): Pose mAP50=0.857, Pose mAP50-95=0.851

## Inference Pipeline

Implemented in `onnx_inference/photocrop.py`.

```
1. Detection     → find photo bounding boxes
2. Pose          → find 4 corners per photo (keypoints)
2b. Pose Refine  → re-derive bbox from keypoints → re-run pose (optional, --pose-refine)
3. Dedup         → remove duplicate detections
4. Warp Recovery → compute warp score; re-pose high-warp detections (default on)
5. Rescue        → Sobel edge detection recovers invisible corners (always on)
6. Corner Ref.   → corner regression model on 320×320 crops (optional, --corner-refine)
7. CV Ref.       → sub-pixel edge intersection (optional, --cv-refine)
8. Crop/Warp     → extract perspective-corrected photo
```

### Warp Recovery

After dedup, compute a **warp score** for each detected photo — a metric for how much the detected quad deviates from a perfect rectangle. A photo is flagged for recovery when:

- **Absolute threshold** (default 1.15): opposing edge lengths differ by >15%
- **Peer outlier** (default 2.0×): warp exceeds median × ratio AND exceeds the absolute threshold

Recovery iteratively re-runs the pose model with progressively larger crops (starting at 10%, increasing by 8% per iteration, max 15 iterations). Stops early when warp drops below threshold.

### Corner Refinement Modes

| Mode | Flag | Model | How it works |
|------|------|-------|--------------|
| Regression | `--corner-refine` (default) | corner-regression-v2 (320×320) | Dedicated 1-class model finds precise corner position |
| Pose refine | `--corner-refine-model pose` | pose_single_ep42 | Reuses pose model on corner crop |
| CV refinement | `--cv-refine` | None (classical CV) | Sobel edge detection + line intersection → sub-pixel |

### Presets

| Preset | Stages | Time |
|--------|--------|------|
| fast | Detect + pose + dedup + warp recovery | ~700ms |
| pose_refine | + pose-refine + adaptive margin | ~1s |
| corner_refine | + pose-refine + corner-refine (regression) + adaptive margin | ~3s |

All presets include warp recovery by default. Rescue is always on.

## Training Pipeline

Each model follows the same pattern: **generator → dataset → trainer**.

- **Generators** create synthetic training images with random photos, backgrounds, shadows, glare, and perspective warps
- **Shared utilities**: `data_generator/generate_common.py`
- **Pretrained weights**:
  - `training/yolo26s-pose.pt` — pose model (small variant)
  - `training/yolo26s.pt` — detection & fiducial models (small variant)
  - `training/yolo26n-pose.pt` — corner regression (nano variant)
  - `training/yolo26n.pt` — detection fallback (nano variant)

### Key Training Parameters

| Model | Architecture | Key Params | Notes |
|-------|-------------|-----------|-------|
| Detection | yolo26s | mosaic=0.5, mixup=0.1, scale=0.5 | Standard detection augmentation |
| Pose | yolo26s-pose | mosaic=0.3, mixup=0.0, flip_idx=[3,2,1,0] | No mixup (corrupts keypoints), no flipud |
| Corner regression | yolo26n-pose | mosaic=0.0, box=4.0, kobj=2.0, rle=1.0 | 320×320 input, 1 class + 1 keypoint |
| Fiducial-pose | yolo26s-pose | mosaic=0.0, cls=0.3, rle=0.5, box=4.0 | V3 in development |

## Export

`export/export_onnx.py` converts `.pt` models to `.onnx` for inference. All three production models have exported ONNX versions in `models/`.

## Data Flow

```
Source photos (data_generator/images/)
    + Backgrounds (data_generator/textures/)
    + Oxford/DTD datasets
         │
         ▼
    Data Generators
    ├── generate_detection.py           → data_detection/
    ├── generate_pose.py                → data_pose/
    ├── generate_corner_regression.py   → data_corner_regression/
    └── generate_fiducial_pose.py       → data_fiducial_pose/
         │
         ▼
    Training Scripts
    ├── train_detection.py          → models/detection_ep47.{pt,onnx}
    ├── train_pose.py               → models/pose_single_ep42.{pt,onnx}
    ├── train_corner_regression.py  → models/corner-regression-v2.onnx
    └── train_fiducial_pose.py     → (in progress)
         │
         ▼
    ONNX Export (export/export_onnx.py)
         │
         ▼
    Inference (onnx_inference/photocrop.py)
         │
         ▼
    Kotlin Integration (petrie-file-importer)
    ├── YoloDetectionService.kt
    ├── YoloPoseService.kt
    ├── YoloCornerRegressionService.kt
    └── YoloPhotoScanPipeline.kt
```

## Integration with petrie-file-importer

The ONNX models are also used in the Kotlin desktop app (petrie-file-importer) for YOLO-based photo detection:

- **YoloDetectionService** — loads `detection_model.onnx`, runs detection inference
- **YoloPoseService** — loads `pose_model.onnx`, runs pose inference
- **YoloCornerRegressionService** — loads `corner_regression_model.onnx`, runs corner refinement
- **YoloPhotoScanPipeline** — orchestrates the full multi-stage pipeline
- **PhotoScanDetectorService** — dual CV/YOLO mode, falls back to classical CV if models unavailable

The pipeline achieves **15/16 corners within ±10px** of the Python reference output on real-world test images, with the 1 mismatch being a rescued corner where CV edge detection produces a valid but different result.

### Corner Order Mapping

When integrating YOLO ONNX models into petrie, corner order must be mapped:

| YOLO keypoint | Petrie corner |
|---------------|---------------|
| kp0 LL (lower-left) | bottomLeft |
| kp1 UL (upper-left) | topLeft |
| kp2 UR (upper-right) | topRight |
| kp3 LR (lower-right) | bottomRight |