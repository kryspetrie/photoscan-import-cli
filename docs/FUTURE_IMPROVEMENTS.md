# Future Improvements

Status of the two-model pipeline after the V2 refactoring.

---

## Current Status (V2)

The V2 refactoring addressed the two root causes of the V1 pose model failure:

1. **Fixed `flip_idx`**: Changed from `[1, 0, 3, 2]` (wrong — assumed TL/TR/BR/BL order) to `[3, 2, 1, 0]` (correct for LL/UL/UR/LR order: horizontal flip swaps LL↔LR, UL↔UR).
2. **Fixed `kpt_shape`**: Changed from `[4, 2]` (wrong — read garbage from 17-column labels) to `[4, 3]` (correct — reads x, y, visibility per keypoint).
3. **Separated data pipelines**: Detection model trains on multi-photo scenes; pose model trains on tightly-cropped single-photo images matching the inference distribution.
4. **Upgraded to yolo26s-pose**: Larger model with more capacity for precise keypoint localization.

**The V2 pose model has not yet been trained.** If it achieves acceptable corner accuracy (sub-pixel or near-sub-pixel), no further architecture changes are needed.

---

## Escalation Plan: Corner Detection Models

If the V2 pose model still fails to achieve sufficient corner accuracy, the following approach offers a fundamentally different decomposition of the problem.

### Core Insight

Each corner of a photo on a contrasting background is **visually unique** — the photo interior extends in a different direction from each corner:

| Corner | Photo extends | L-shape orientation |
|--------|---------------|---------------------|
| LL (Lower-Left) | Right + Up | ┗ |
| UL (Upper-Left) | Right + Down | ┏ |
| UR (Upper-Right) | Left + Down | ┓ |
| LR (Lower-Right) | Left + Up | ┛ |

Even in a tight crop around the corner, the L-shaped boundary with photo content on one side and background on the other is visually unambiguous. This makes the corner-finding problem a well-posed detection task.

### Proposed Architecture

```
Input Image
    │
    ▼
┌──────────────┐
│  Detection   │  1 forward pass
│   Model      │  → axis-aligned bounding boxes
└──────┬───────┘
       │
       ├─ bbox[0] ──→ extract edge strips ──┐
       ├─ bbox[1] ──→ extract edge strips ──┤
       │                ...                   ├─→ 1 Corner Detection Model (4-class)
       └─ bbox[N] ──→ extract edge strips ──┘     → corner bboxes per class
                                                     │
                                                     ▼
                                              corner coordinates
                                                     │
                                                     ▼
                                          perspective warp or crop
```

### Detailed Pipeline

1. **Detection model** finds axis-aligned bounding boxes of photos in the full image (unchanged from current pipeline).

2. **Extract edge strips**: For each detected bbox, extract four narrow strips along the edges:
   - Top strip: crop from `bbox_top` to `bbox_top + strip_height`, full width
   - Bottom strip: crop from `bbox_bottom - strip_height` to `bbox_bottom`, full width
   - Left strip: crop from `bbox_left` to `bbox_left + strip_width`, full height
   - Right strip: crop from `bbox_right - strip_width` to `bbox_right`, full height

   The strip dimensions should be proportional to the bbox size (e.g., 20-30% of the shorter axis). Each strip is resized to a fixed input resolution (e.g., 640×160 for horizontal strips, 160×640 for vertical strips).

3. **Corner detection model** (single model, 4 classes):
   - Class 0: UL corner (found in top strip near left edge, or left strip near top)
   - Class 1: UR corner (found in top strip near right edge, or right strip near top)
   - Class 2: LL corner (found in bottom strip near left edge, or left strip near bottom)
   - Class 3: LR corner (found in bottom strip near right edge, or right strip near bottom)

   The model outputs small bounding boxes around each corner, providing sub-pixel localization via bbox center.

4. **Map back**: Convert corner detections from strip coordinates back to full-image coordinates.

### Training Data Generation

Generate synthetic corner strips from the existing photo compositing pipeline:

1. Render full scenes with photos on backgrounds (reuse `generate_detection.py`).
2. For each photo, compute the axis-aligned bbox.
3. Extract edge strips and label corner bboxes within each strip.
4. Augment with random offsets — shift the strip crop region by ±N pixels along the edge so the corner appears at varying positions within the strip.

### Advantages Over Pose Model

| Factor | Pose Model | Corner Detection |
|--------|-----------|-----------------|
| Precision | Regression on keypoints (single-pixel) | Bbox regression (sub-pixel via center) |
| Task difficulty | All 4 corners from full context | Single corner in small crop |
| Visual ambiguity | Must distinguish corners by position | Each corner class has unique L-shape |
| Training signal | 4 keypoints share one loss | 4 classes with separate bbox losses |
| Robustness | Sensitive to photo scale/position | Robust — always sees same visual pattern |

### Disadvantages

- **More complex pipeline**: 1 detection + N×4 corner inferences vs. 1 detection + N pose inferences
- **Edge strip strategy assumes moderate distortion**: If perspective is extreme, a corner might not be on its expected edge strip
- **No global shape constraint**: Pose model sees the full quadrilateral; corner model sees each corner independently

---

## Root Cause Analysis: V1 Pose Model Failure

The V1 pose model (mAP50-95 = 0.107) predicted all four keypoints near the top of the image, with LL and LR shifted ~270px above their ground truth positions. This was **not** a fundamental limitation of keypoint regression or architecture — it was caused by two configuration bugs that corrupted the keypoint learning signal.

### Bug 1: Wrong `flip_idx` — Horizontal Flip Augmentation Swapped Wrong Keypoints

The old `training/dataset.yaml` had:
```yaml
flip_idx: [1, 0, 3, 2]
kpt_shape: [4, 2]
```

The `flip_idx` `[1, 0, 3, 2]` is correct for TL/TR/BR/BL ordering (horizontal flip swaps TL↔TR and BL↔BR). But `generate.py` produced labels in LL/UL/UR/LR ordering.

For LL/UL/UR/LR, the correct `flip_idx` is `[3, 2, 1, 0]` (horizontal flip swaps LL↔LR and UL↔UR).

**What happened during training:**

| Keypoint | Correct swap | Actual swap (old flip_idx) | Result |
|----------|-------------|---------------------------|--------|
| LL (kp0) | ↔ LR (kp3) | ↔ UL (kp1) | Model learned LL and UL are interchangeable |
| UL (kp1) | ↔ UR (kp2) | ↔ LL (kp0) | Model learned UL and LL are interchangeable |
| UR (kp2) | ↔ UL (kp1) | ↔ LR (kp3) | Model learned UR and LR are interchangeable |
| LR (kp3) | ↔ LL (kp0) | ↔ UR (kp2) | Model learned LR and UR are interchangeable |

Over thousands of augmented training images, every horizontal flip told the model that LL and UL occupy the "same" keypoint slot (just mirrored). The model responded by **averaging their predictions** — LL was predicted near the UL position, and LR near the UR position. This exactly matches the observed error pattern: LL ~273px above ground truth (near UL), LR ~270px above (near UR).

### Bug 2: Wrong `kpt_shape` — YOLO Read Garbage Keypoint Coordinates

The old `dataset.yaml` had `kpt_shape: [4, 2]`, telling YOLO to read 2 values per keypoint (x, y only). But `generate.py` wrote labels with 3 values per keypoint (x, y, visibility):

```
0 cx cy w h kp0x kp0y kp0v kp1x kp1y kp1v kp2x kp2y kp2v kp3x kp3y kp3v
  0  1  2  3    5    6    7    8    9   10   11   12   13   14   15   16
```

With `kpt_shape [4, 2]`, YOLO reads only 2 values per keypoint (stride of 2 instead of 3):

| YOLO reads | Actual data | Correct data | Error |
|-----------|------------|--------------|-------|
| kp0.x = parts[5] | kp0x | kp0x | ✓ Correct |
| kp0.y = parts[6] | kp0y | kp0y | ✓ Correct |
| kp1.x = parts[7] | **kp0v = 2.0** | kp1x | ✗ 200% of image width! |
| kp1.y = parts[8] | **kp1x** | kp1y | ✗ Wrong coordinate |
| kp2.x = parts[9] | **kp1y** | kp2x | ✗ Wrong coordinate |
| kp2.y = parts[10] | **kp1v = 2.0** | kp2y | ✗ 200% of image height! |
| kp3.x = parts[11] | **kp2x** | kp3x | ✗ Shifted by 1 keypoint |
| kp3.y = parts[12] | **kp2y** | kp3y | ✗ Shifted by 1 keypoint |

Values of 2.0 (200% of image dimension) are off-image and would be clamped, ignored, or produce NaN losses. Three out of four keypoints had corrupted coordinates on every single training sample.

### Combined Effect

Either bug alone would catastrophically break keypoint training. Together, the model had essentially zero reliable keypoint signal. The fact that it achieved mAP50-95 = 0.107 (rather than 0.0) is surprising and likely reflects the model learning from the one uncorrupted keypoint (kp0/LL) plus whatever signal survived the garbage.

### Verification

The V2 dataset YAML (`training/dataset_pose.yaml`) now correctly specifies:
```yaml
kpt_shape: [4, 3]   # 4 keypoints, 3 values each (x, y, visibility)
flip_idx: [3, 2, 1, 0]  # Horizontal flip: LL↔LR, UL↔UR
```

**Note:** An earlier version of `dataset_pose.yaml` had `flip_idx: [2, 3, 0, 1]`, which swaps corners across *diagonals* (LL↔UR, UL↔LR) instead of across the vertical axis. This was corrected before any V2 training run.

---

## Other Potential Improvements

### Classical CV Corner Refinement

After the pose model predicts coarse corner positions, a classical refinement step could achieve sub-pixel accuracy:

1. For each predicted corner, search along a ray from the photo center through the corner
2. Find the strongest edge gradient along that ray
3. Refine the corner to the sub-pixel edge position
4. Optionally, fit lines to the edges and compute their intersection

This requires no additional training and can correct for small systematic errors in the pose model.

### Multi-Scale Pose Inference

If the pose model struggles with photos of very different sizes within the detection crop, run inference at multiple scales and merge predictions.

### Anchor-Free Detection

The current detection model uses anchor-based YOLO. An anchor-free variant (e.g., YOLO's native anchor-free mode) might produce more consistent bounding boxes for irregularly-shaped photos.