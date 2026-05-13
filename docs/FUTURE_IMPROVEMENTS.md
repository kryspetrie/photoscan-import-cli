# Architecture Evolution & Future Improvements
=============================================

## What We've Tried (and What Worked)

| Approach | Status | mAP50 | mAP50-95 | Precision | Notes |
|----------|--------|-------|----------|-----------|-------|
| Detection model | ✅ Success | 0.995 | 0.918 | N/A | Finds bounding boxes reliably |
| Pose model (single-photo) | ⚠️ Partial | 0.995 | 0.107 | N/A | Coarse corners OK, sub-pixel poor |
| Pose model (multi-photo) | ❌ Failed | — | 0.838* | 0.50 | Overfit: 0% bg images, P stuck at 0.50 |
| Fiducial 4-class | ❌ Failed | 0.906 | 0.358 | — | cls_loss ≈ random; can't classify by orientation |
| Fiducial binary (current) | 🔄 Training | 0.917 | 0.364 | 0.60 | Promising at epoch 10; still improving |

\* Best validation epoch before overfitting

### Why Multi-Pose Failed

The multi-photo pose model trained on full scenes had a fundamental data problem:
0% background-only images. Every training image contained at least one photo,
so the model learned to always fire. Result: precision stuck at 0.50 with
recall at 0.99 — essentially one false positive per true positive.

Even with background images added, the single-photo pose model's keypoint
accuracy (mAP50-95=0.107) suggests keypoint regression is inherently less
precise than detection for localizing small features like corners.

### Why 4-Class Fiducial Failed

The 4 corner types (UL, UR, LL, LR) have nearly identical visual appearance at
640×640 — they're all L-shaped photo/background boundaries differing only by
90° rotation. The model could **detect** corners (box loss converged) but
**couldn't classify** them (cls_loss stuck near random).

### Why Binary Fiducial Works

Each binary model answers a single yes/no question: "Is THIS specific corner
orientation present?" This eliminates the hardest part of the task (distinguishing
similar orientations) while preserving the easy part (detecting the L-shaped
boundary). Early results show mAP50=0.917 at epoch 10, already surpassing the
4-class model's best of 0.906 at epoch 42.

## Current Pipeline (Recommended)

```
Input Image
    │
    ▼
┌──────────────┐
│  Detection   │  ← Always runs first
│   Model      │  → bounding boxes for each photo
└──────┬───────┘
       │
       │ (optional, recommended for multi-photo scenes)
       ▼
┌──────────────┐
│  Pose Model  │  ← Single-photo, tightly-cropped
│  (single)    │  → approximate 4 corner positions
└──────┬───────┘
       │
       │ If no pose model: use bbox corners as rough positions
       │
       ▼
┌──────────────────────┐
│  4 Binary Fiducial   │  ← 4 forward passes (1 per corner type)
│  Models (iterative)  │  → precise corner positions
└──────┬───────────────┘
       │
       │ (optional)
       ▼
┌──────────────────┐
│  CV Refinement   │  ← Sobel edge detection + line intersection
│  (edge search)   │  → sub-pixel accuracy
└──────┬───────────┘
       │
       ▼
  Perspective warp → clean extracted photo
```

## Future Improvements

### Higher-Confidence Binary Fiducial (When Training Completes)

The current binary models are training with corrected hyperparameters
(lr=0.001, optimizer=AdamW, mosaic=0). Once all 4 models finish:
- Export each to ONNX
- Integrate into `photocrop.py` pipeline
- Benchmark end-to-end accuracy on real scanned images

### Classical CV Corner Refinement (Already Implemented)

After the fiducial model locates corners to ~1–2px accuracy, classical CV
can push to sub-pixel:

1. For each corner, search along two rays through the photo interior
2. Find the strongest edge gradient along each ray
3. Fit lines to the edge pixels (weighted least-squares)
4. Compute the intersection of the two lines → sub-pixel corner position

This is already implemented in `photocrop.py` as `refine_corners_cv()` and
available via `--cv-refine` or `--auto-refine`.

### Multi-Scale Fiducial Inference

If the binary fiducial models struggle with very small or very large corners,
run inference at multiple scales and merge predictions. This is a simple
post-processing step that doesn't require retraining.

### Background Image Augmentation

For any future training (pose or detection), include **10–15% background-only
images** (table surfaces, textures, gradients with no photos). This prevents
the objectness head from learning an "always fire" bias.

### Anchor-Free or Focused Detection

If bounding box precision becomes a bottleneck, consider anchor-free detection
variants or a focused detection approach that only refines boxes near the
expected corner positions rather than scanning the entire crop.

## Root Cause Analysis: V1 Pose Model Failure

The V1 pose model (mAP50-95=0.107) predicted all four keypoints near the top
of the image. This was caused by two configuration bugs, not a fundamental
architecture problem:

### Bug 1: Wrong `flip_idx`

The old config had `flip_idx: [1, 0, 3, 2]` (correct for TL/TR/BR/BL) but
labels were in LL/UL/UR/LR order. The correct value is `[3, 2, 1, 0]`.

This caused horizontal flip to swap the wrong keypoint pairs, making the model
learn that LL↔UL and LR↔UR are interchangeable. It averaged their predictions,
pushing LL up toward UL position and LR up toward UR position.

### Bug 2: Wrong `kpt_shape`

The old config had `kpt_shape: [4, 2]` (x,y only) but labels used 3 values
per keypoint (x, y, visibility). This caused YOLO to read visibility values
as coordinates, producing garbage keypoint positions for 3 of 4 keypoints.

Both bugs were fixed in the V2 config (`kpt_shape: [4, 3]`, `flip_idx: [3, 2, 1, 0]`),
but the pose approach was abandoned in favor of binary fiducial detection,
which achieves better corner localization through a simpler task decomposition.