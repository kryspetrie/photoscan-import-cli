# Refactoring Plan: Separate Data Pipelines for Detection & Pose Models

> **Status:** Draft — Awaiting decision after 3–6 more pose training epochs  
> **Date:** 2026-04-27  
> **Author:** Photo Pose Detector Project

## Background & Motivation

### The Train-Inference Distribution Mismatch

The current architecture has a fundamental problem: **the pose model trains on one visual distribution but deploys on another.**

| | Training | Inference |
|---|---|---|
| **Photo scale** | Small-to-medium (30–60% of 640×640 image) | Fills nearly the entire crop |
| **Background** | Table/textured surfaces visible | Minimal background visible |
| **Instances** | 1–4 photos per image | Exactly 1 photo per crop |
| **Keypoints** | Scattered across full image coordinates | Always near image edges |

This is the standard problem in two-stage detection pipelines. Faster R-CNN, Mask R-CNN, and all RoI-based architectures train their second stage on **cropped region proposals**, not on full scenes. Our pose model should do the same.

### Current Training Results (Epochs 1–4)

| Epoch | mAP50(B) | mAP50(P) | mAP50-95(P) | pose_loss |
|-------|----------|----------|-------------|-----------|
| 1 | 0.995 | 0.011 | 0.002 | 0.516 |
| 2 | 0.995 | 0.489 | 0.069 | 0.338 |
| 3 | 0.995 | 0.298 | 0.034 | 0.300 |
| 4 | 0.995 | 0.551 | 0.071 | 0.288 |

Bounding box detection is already converged. Pose keypoint detection is learning slowly and oscillating. mAP50-95(P) is still below 0.1 — the model struggles with precise localization.

### What We're Watching For

If pose mAP50(P) reaches **>0.85 with mAP50-95(P) >0.4** by epoch 6–10, the current approach may be good enough. If it plateaus or stays erratic, we proceed with this refactoring.

---

## Plan Overview

Separate the two models into fully independent data pipelines:

1. **Detection generator** → full scenes (1–4 photos), bounding box labels only
2. **Pose generator** → single-photo crops (1 photo, tightly framed), keypoint labels only

This eliminates:
- The symlink hack (`data/labels` → `data/detection/labels` or `data/pose/labels`)
- Shared image directories between models
- The train-inference distribution mismatch for the pose model

---

## 1. Detection Model Data Pipeline

### Generator: `generate_detection.py`

A simplified copy of the current `generate.py` with pose-related code **removed**.

**What stays (identical output):**
- 1–4 photos per 640×640 image
- Same background generation, shadow, glare, texture overlay
- Same perspective warp
- Same overlap validation and bounds checking
- Same bounding box label format: `class x_center y_center width height`
- Same visual output — images look exactly the same

**What gets removed:**
- All corner/keypoint calculation code
- Pose label writing (`pose_labels` list, `pose_labels.txt` files)
- `pose/` output directory structure
- Corner coordinate variables that were only used for pose labels

**Output structure:**
```
data_detection/
├── images/
│   ├── train/    # 640×640 composite JPEGs
│   └── val/
└── labels/
    ├── train/    # 5-column YOLO detection labels
    └── val/
```

**Dataset YAML (`dataset_detection.yaml`):**
```yaml
path: /path/to/data_detection
train: images/train
val: images/val
nc: 1
names:
  0: photo
```

No symlinks. The dataset is self-contained.

---

## 2. Pose Model Data Pipeline

### Generator: `generate_pose.py`

A new generator that produces **single-photo cropped images** with precise 4-corner keypoints.

**Key differences from current generator:**

| Aspect | Detection | Pose (New) |
|--------|-----------|------------|
| Photos per image | 1–4 | **1 only** |
| Final image size | 640×640 | **640×640** |
| Initial render size | 640×640 | **≥1280×1280** (2× minimum) |
| Crop & pad step | None | **Yes** (see below) |
| Background | Full table surface | **Only margin around photo** |
| Labels | Bounding box only | **Keypoints only** |
| Overlap checking | Multi-photo | **N/A** (1 photo) |

### High-Resolution Render → Scale → Crop → Pad Pipeline

The core innovation: render at high resolution, then scale and crop to create the "as if detected" framing.

```
Step 1: Render at 2× resolution
  ┌─────────────────────────────────────┐
  │  High-res canvas (e.g., 1280×1280)  │
  │       ┌───────────────┐             │
  │       │               │             │
  │       │   Photo with  │             │
  │       │   perspective │             │
  │       │   warp        │             │
  │       │               │             │
  │       └───────────────┘             │
  │    (axis-aligned bbox of photo)     │
  └─────────────────────────────────────┘

Step 2: Calculate bounding box of the warped photo
  bbox = axis-aligned bounding box around 4 corner keypoints

Step 3: Calculate scale factor
  scale = min(600 / bbox_width, 600 / bbox_height)
  (This makes the bounding box fit within 600×600 pixels)

Step 4: Scale entire image and all coordinates by scale factor
  - If scale < 1: image shrinks (large photo → fits in 600px)
  - If scale > 1: image grows (small photo → fills 600px)
  - All corner coordinates multiplied by scale

Step 5: Square crop with 20px top-left-aligned padding
  - Place the scaled bounding box at position (20, 20)
  - Crop a 640×640 window: (0, 0) to (640, 640)
  - All corner coordinates shifted: x_new = x_scaled - crop_x + 20
    
  ┌─────────────────────────────────────┐
  │ 20px │                              │
  │ ┌────┼──────────────────────────┐   │
  │ │    │                          │   │ 640×640
  │ │    │  Photo fills most of     │   │
  │ │    │  the frame, ~20px margin │   │
  │ │    │  on all sides            │   │
  │ └────┼──────────────────────────┘   │
  │      │ 20px                         │
  └─────────────────────────────────────┘
```

### Detailed Steps

```python
POSE_CANVAS_INIT_SIZE = 1280  # Initial high-res render size
POSE_FINAL_SIZE = 640         # Final output image size
POSE_FIT_SIZE = 600           # Bounding box target fit size
POSE_PADDING = 20             # Minimum padding around photo
```

1. **Create high-res canvas** (1280×1280 or larger, configurable)
2. **Select and place 1 photo** with rotation, shadow, glare (same effects as detection)
3. **Apply perspective warp** (same `apply_perspective_safe` logic, but only 1 photo)
4. **Get corner keypoints** after perspective transform
5. **Calculate axis-aligned bounding box** of the 4 corners:
   ```python
   bbox_x1 = min(c[0] for c in corners)
   bbox_y1 = min(c[1] for c in corners)
   bbox_x2 = max(c[0] for c in corners)
   bbox_y2 = max(c[1] for c in corners)
   bbox_w = bbox_x2 - bbox_x1
   bbox_h = bbox_y2 - bbox_y1
   ```
6. **Calculate scale** to make bbox fit within `POSE_FIT_SIZE × POSE_FIT_SIZE`:
   ```python
   scale = min(POSE_FIT_SIZE / bbox_w, POSE_FIT_SIZE / bbox_h)
   ```
7. **Scale the entire image** by that factor:
   ```python
   new_w = int(POSE_CANVAS_INIT_SIZE * scale)
   new_h = int(POSE_CANVAS_INIT_SIZE * scale)
   scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
   ```
8. **Scale all corner coordinates** by the same factor:
   ```python
   scaled_corners = corners * scale
   scaled_bbox_x1 = bbox_x1 * scale
   scaled_bbox_y1 = bbox_y1 * scale
   ```
9. **Calculate crop offset** to position the bbox at (20, 20):
   ```python
   crop_x = int(scaled_bbox_x1 - POSE_PADDING)
   crop_y = int(scaled_bbox_y1 - POSE_PADDING)
   ```
10. **Pad image if needed** to ensure the 640×640 crop is valid:
    ```python
    # If the crop would go out of bounds, pad with background
    pad_top = max(0, -crop_y)
    pad_left = max(0, -crop_x)
    pad_bottom = max(0, (crop_y + POSE_FINAL_SIZE) - new_h)
    pad_right = max(0, (crop_x + POSE_FINAL_SIZE) - new_w)
    # Use same background color/texture for padding
    ```
11. **Crop 640×640 window** and **shift all coordinates**:
    ```python
    final_corners = scaled_corners - np.array([crop_x, crop_y])
    ```
12. **Verify all corners are within bounds** (each corner should have ≥20px margin on all sides, or at minimum be within the 640×640 image)
13. **Generate YOLO-pose label** with normalized coordinates:
    ```python
    # Bounding box (required by YOLO-pose format)
    x_center = ((min_x + max_x) / 2) / POSE_FINAL_SIZE
    y_center = ((min_y + max_y) / 2) / POSE_FINAL_SIZE
    width = (max_x - min_x) / POSE_FINAL_SIZE
    height = (max_y - min_y) / POSE_FINAL_SIZE
    # Keypoints (normalized)
    for corner in final_corners:
        kp_x = corner[0] / POSE_FINAL_SIZE
        kp_y = corner[1] / POSE_FINAL_SIZE
        kp_v = 2  # visible
    ```
14. **Reorder corners to LL, UL, UR, LR** (same convention as current model)

### Output Structure

```
data_pose/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/    # 13-column YOLO-pose labels (1 object per image)
    └── val/
```

### Dataset YAML (`dataset_pose.yaml`)

```yaml
path: /path/to/data_pose
train: images/train
val: images/val
nc: 1
names:
  0: photo
kpt_shape: [4, 3]
flip_idx: [2, 3, 0, 1]
```

No symlinks. Self-contained dataset.

---

## 3. Shared Code: `generate_common.py`

Both generators share substantial logic. Extract common functions into a shared module:

**Shared functions:**
- `random_base_background(w, h)` — background generation
- `apply_texture_overlay(canvas)` — texture application
- `fast_glare(photo)` — specular glare
- `rotate_photo(photo, angle)` — rotation with alpha channel
- `apply_photo_shadow(...)` — drop shadow
- `composite_photo_at_center(canvas, photo, cx, cy)` — compositing
- `apply_perspective_safe(canvas, corners_list)` — perspective warp
- `get_rotated_polygon(...)` — corner calculation
- `create_debug_image(img, photos)` — debug visualization

**Kept separate per generator:**
- Detection: `pack_photos_validated()`, overlap checking, multi-photo placement
- Pose: single-photo placement, scaling, cropping, padding pipeline

---

## 4. Training Script Updates

### `train_detection.py`

- Remove `label_links` import and symlink management
- Update default `data` path to `dataset_detection.yaml` (pointing to `data_detection/`)
- Default `device="cpu"`, `cache="ram"` for Intel Mac training
- Default `workers=4` (matches current working config)

### `train_pose.py`

- Remove `label_links` import and symlink management
- Update default `data` path to `dataset_pose.yaml` (pointing to `data_pose/`)
- Default `device="cpu"`, `cache="ram"` for Intel Mac training
- Default `workers=4`
- Reduce `scale` augmentation since the photo already fills most of the frame (suggest `scale=0.2` — smaller range since the photo is already large)
- Consider reducing `copy_paste=0.0` — no point copying a photo onto a single-photo crop
- Keep `fliplr=0.5` — important for keypoint flip learning

### `train_both.py`

- Remove all symlink references
- Update to call both generators if data doesn't exist
- Default `device="cpu"`, `cache="ram"`, `workers=4`

### `label_links.py`

- **Delete this file.** It exists only to manage the shared symlink hack. With separate datasets, it's unnecessary.

---

## 5. Data Generation Count Recommendations

| Dataset | Training | Validation | Total |
|---------|----------|------------|-------|
| Detection | 4,000 | 1,000 | 5,000 |
| Pose | 4,000 | 1,000 | 5,000 |

Pose generation is simpler (1 photo per image) so it may be faster per image, but the high-res render + scale step adds overhead. Net effect should be similar or slightly faster than current generation.

---

## 6. Migration & Cleanup

### Files to Create
- `data_generator/generate_common.py` — shared utilities
- `data_generator/generate_detection.py` — detection-specific generator
- `data_generator/generate_pose.py` — pose-specific generator

### Files to Modify
- `training/train_detection.py` — remove label_links, update defaults
- `training/train_pose.py` — remove label_links, update defaults
- `training/train_both.py` — remove label_links, update workflow
- `training/dataset_detection.yaml` — point to `data_detection/`
- `training/dataset_pose.yaml` — point to `data_pose/`

### Files to Delete
- `training/label_links.py` — no longer needed

### Files to Keep (with updates)
- `data_generator/generate.py` — keep as reference/archive, but mark as deprecated

---

## 7. Additional Suggestions

### 7.1 Variable Render Resolution

Rather than always rendering at a fixed 1280×1280, consider:
- **Small photos** (270–350px source) → 1280×1280 initial render
- **Medium photos** (350–500px source) → 1600×1600 initial render
- **Large photos** (500–640px source) → 1920×1920 initial render

This ensures the scaling step always has enough resolution. The extra resolution is needed because we scale *down* after perspective warping — if the initial render is too small, the final 640×640 image will be pixelated.

**Simpler alternative:** Always render at 1280×1280. This gives 2× headroom which is sufficient for most cases. Only increase if quality issues are observed.

### 7.2 Padding Variation

The plan specifies 20px padding. Consider randomizing padding between 15–40px:
- Simulates variation in detection model crop accuracy
- Makes the model robust to different crop positions
- Still guarantees the photo fills most of the frame

```python
POSE_PADDING_MIN = 15
POSE_PADDING_MAX = 40
padding = random.randint(POSE_PADDING_MIN, POSE_PADDING_MAX)
```

### 7.3 Background in Margins

The margins around the photo will be whatever the high-res background was (table texture, gradients). This is fine — it simulates what the detection model would crop. But consider also:
- Occasionally using **solid color backgrounds** in margins (simulates crop from a larger table)
- Occasionally adding **other objects partially visible** at edges (simulates detection model including some background)

These are optional refinements that could be added in a v2.

### 7.4 Keypoint Visibility Edge Cases

If the perspective warp causes a corner to be very close to the 20px margin, the corner could end up at the image boundary. Two options:
1. **Clamp and set visibility=0** — mark the keypoint as invisible (more correct, but reduces training signal)
2. **Reject and regenerate** — throw away the image and try again with a different perspective (simpler, ensures all corners are visible)
3. **Increase padding** — set the minimum margin to 40px, giving more buffer

Recommend option 3 for initial implementation, with option 2 as a fallback safety check.

### 7.5 Augmentation Adjustments for Cropped Pose Model

With the photo filling most of the frame, certain augmentations need adjustment:

| Augmentation | Current | Recommended | Reason |
|---|---|---|---|
| `scale` | 0.3 | **0.2** | Photo already fills frame; less scale variation needed |
| `translate` | 0.1 | **0.05** | Less translation room in a tight crop |
| `mosaic` | 0.5 | **0.3** | Mosaic with single-photo crops is less useful |
| `copy_paste` | 0.0 | **0.0** | No second photo to paste |
| `mixup` | 0.0 | **0.0** | Blending two single-photo crops is noise |
| `degrees` | 10 | **5–10** | Small rotations are fine; large ones change the crop framing |
| `fliplr` | 0.5 | **0.5** | Keep — critical for keypoint learning with `flip_idx` |
| `hsv_h/s/v` | 0.015/0.3/0.3 | **Same** | Color augmentation is always useful |

### 7.6 Inference Pipeline Update

The inference pipeline (Python, Kotlin) must be updated to match the new training:

```python
# CURRENT (works but suboptimal due to distribution mismatch)
det_results = det_model.predict(image)
for box in det_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    region = image[y1:y2, x1:x2]  # Raw crop
    pose_results = pose_model.predict(region)

# NEW (matches training distribution)
det_results = det_model.predict(image)
for box in det_results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # Add padding like training did (20px)
    pad = 20
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(image.shape[1], x2 + pad)
    cy2 = min(image.shape[0], y2 + pad)
    region = image[cy1:cy2, cx1:cx2]
    # Scale to match training: photo bbox fills ~600px of 640px
    bbox_w, bbox_h = x2 - x1, y2 - y1
    scale = min(600 / bbox_w, 600 / bbox_h)
    if scale > 1:  # Only upscale small crops
        new_w = int(region.shape[1] * scale)
        new_h = int(region.shape[0] * scale)
        region = cv2.resize(region, (new_w, new_h))
    pose_results = pose_model.predict(region)
    # Scale keypoint coordinates back to original image space
```

This ensures inference sees the same distribution as training. The exact matching logic should be validated by comparing inference preprocessing with training's `generate_pose.py` pipeline.

---

## 8. Success Criteria

After implementing this plan, we expect:

1. **Pose mAP50(P) > 0.85** within 10 epochs (vs. current ~0.55 after 4 epochs)
2. **Pose mAP50-95(P) > 0.5** within 50 epochs (vs. current ~0.07 after 4 epochs)
3. **Corners localized to <5px error** on average (vs. current undetermined)
4. **Training convergence faster** — each epoch should show measurable improvement
5. **No symlink management** — both datasets fully self-contained

---

## 9. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| High-res render + scale is slow | Profile first; 1280→640 resize is fast |
| Padding variation causes OOB keypoints | Add safety check, regenerate if any corner is within 5px of edge |
| Inference pipeline mismatch with new crop method | Write shared `crop_for_pose()` function used by both training and inference |
| Detection model crop quality affects pose accuracy | Detection model is already very good (mAP50 = 0.995); add padding to absorb small detection errors |
| Existing trained detection model becomes invalid | Detection model should be retrained on new `data_detection/`, but data is identical so weights will be the same |

---

## 10. Implementation Order

1. Extract `generate_common.py` from existing `generate.py`
2. Create `generate_detection.py` (simpler — just remove pose code)
3. Create `generate_pose.py` (new — single photo + crop pipeline)
4. Test both generators with `--count 10` examples mode
5. Update `train_detection.py` and `train_pose.py` (remove label_links, update defaults)
6. Update `dataset_detection.yaml` and `dataset_pose.yaml`
7. Delete `label_links.py`
8. Generate full datasets for both models
9. Retrain detection model (verify it matches old results)
10. Retrain pose model (expect significantly better pose metrics)
11. Update inference pipeline (Python, then Kotlin)
12. Update knowledge base docs (ARCHITECTURE.md, TRAINING.md, etc.)