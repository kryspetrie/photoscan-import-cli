# Architecture

## Project Overview

**photo-pose-detector** detects physical photographs in camera-scanned images, locates their four corners, and extracts perspective-corrected crops. The pipeline runs detection to find photos within the scene, pose estimation to pinpoint corners, and geometric correction to produce final output.

## Model Inventory

### 1. Detection Model (`detection_ep47`)

- **Architecture**: YOLO detection model (yolo26s)
- **Purpose**: Finds bounding boxes of photos in full-scene images
- **Training**: `training/train_detection.py`
- **Data**: `data_detection/`
- **Generator**: `data_generator/generate_detection.py`
- **Dataset config**: `training/dataset_detection.yaml`

### 2. Pose Model (`pose_single_ep42`) ★ CRITICAL PRODUCTION MODEL

- **Architecture**: YOLO pose model (yolo26s-pose)
- **Purpose**: Detects 4 corner keypoints per photo in tightly-cropped images
- **Keypoints**: kp0=LL, kp1=UL, kp2=UR, kp3=LR (`flip_idx=[3,2,1,0]`)
- **Training**: `training/train_pose.py`
- **Data**: `data_pose/`
- **Generator**: `data_generator/generate_pose.py`
- **Dataset config**: `training/dataset_pose.yaml`

> ⚠️ This model is the foundation of the entire pipeline — it must not be removed.

### 3. Fiducial-Pose Segment Model (IN DEVELOPMENT)

- **Architecture**: YOLO pose model for detecting individual visible edge segments
- **Purpose**: Each segment has keypoints at its endpoints (the visible corners of that segment). Goal is to better handle invisible/occluded corners by assembling segments geometrically.
- **Training**: `training/train_fiducial_pose.py`
- **Data**: `data_fiducial_pose/`
- **Generator**: `data_generator/generate_fiducial_pose.py`
- **Dataset config**: `training/dataset_fiducial_pose.yaml`

## Inference Pipeline

Implemented in `onnx_inference/photocrop.py`.

```
1. Detection   → find photo bounding boxes
2. Pose        → find 4 corners per photo (keypoints)
3. Dedup       → remove duplicate detections
4. Rescue      → Sobel edge detection recovers invisible corners geometrically (always on)
5. Corner Ref. → re-run pose model on corner crops (optional)
6. CV Ref.     → sub-pixel edge intersection (optional)
7. Crop/Warp   → extract perspective-corrected photo
```

## Training Pipeline

Each model follows the same pattern: **generator → dataset → trainer**.

- **Generators** create synthetic training images with random photos, backgrounds, shadows, glare, and perspective warps
- **Shared utilities**: `data_generator/generate_common.py`
- **Pretrained weights**:
  - `training/yolo26s-pose.pt` — pose model
  - `training/yolo26s.pt` — detection & fiducial models

## Export

`export/export_onnx.py` converts `.pt` models to `.onnx` for inference.

## Data Flow

```
Source photos (data_generator/images/)
    + Backgrounds (textures/)
    + Oxford/DTD datasets
         │
         ▼
    Data Generators
    ├── generate_detection.py      → data_detection/
    ├── generate_pose.py           → data_pose/
    └── generate_fiducial_pose.py → data_fiducial_pose/
         │
         ▼
    Training Scripts
    ├── train_detection.py       → models/detection_ep47.{pt,onnx}
    ├── train_pose.py            → models/pose_single_ep42.{pt,onnx}
    └── train_fiducial_pose.py   → (in progress)
         │
         ▼
    ONNX Export (export/export_onnx.py)
         │
         ▼
    Inference (onnx_inference/photocrop.py)
```