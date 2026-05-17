# Architecture Evolution & Future Improvements
=============================================

## What We've Tried (and What Worked)

| Approach | Status | mAP50 | mAP50-95 | Notes |
|----------|--------|-------|----------|-------|
| Detection model | ✅ Success | 0.995 | 0.918 | Finds bounding boxes reliably |
| Pose model (single-photo) | ✅ Success | 0.995 | 0.107 | Finds approximate corners; low mAP50-95 is expected for keypoint tasks |
| Multi-pose model | ❌ Failed | — | 0.838* | Overfit: 0% bg images, precision stuck at 0.50 |
| 4-class fiducial | ❌ Failed | 0.906 | 0.358 | cls_loss ≈ random; can't classify L-shapes by orientation |
| Binary fiducial | ❌ Failed | 0.917 | 0.364 | Corner types are visually identical in crops; classification impossible |
| **Corner refinement (pose)** | ✅ **Current** | — | — | Reuses pose model on corner crops; recovers invisible corners |

\* Best validation epoch before overfitting

### Why Multi-Pose Failed

The multi-photo pose model trained on full scenes had 0% background-only images.
Every training image contained at least one photo, so the model learned to always
fire → precision stuck at 0.50 with recall at 0.99.

### Why Fiducial Models Failed (Both 4-Class and Binary)

The 4 corner types (UL, UR, LL, LR) have **nearly identical visual appearance**
in crops — they're all L-shaped photo/background boundaries differing only by
90° rotation. No model architecture could distinguish them because the task is
inherently ambiguous without geometric context.

**The solution**: Instead of training a new model to classify corners, use
**geometric corner refinement** — crop around each approximate corner (found by
the pose model) and run the **existing** pose or detection model on that crop.
The pose model's named keypoints directly identify the corner, and the detection
model's bounding box position can be classified geometrically.

## Current Pipeline

```
Input Image
    │
    ▼
┌──────────────┐
│  Detection   │  Find photo bounding boxes (1 forward pass)
│   Model      │  → N axis-aligned bounding boxes
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Pose Model  │  Find approximate 4 corners per photo (1 pass per photo)
│  (single)    │  → 4 keypoints (LL, UL, UR, LR) + visibility
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│  Corner Refinement       │  Recover invisible/low-vis corners (1 pass per corner)
│  (pose or detection)     │  Crops around each corner, runs model again
│  --corner-refine          │  → precise positions for vis≈1.0
└──────┬───────────────────┘
       │
       ▼
┌──────────────────┐
│  CV Refinement   │  Sobel edge detection + line intersection (optional)
│  --cv-refine     │  → sub-pixel accuracy
└──────┬───────────┘
       │
       ▼
  Perspective warp → clean extracted photo
```

### Corner Refinement Details

When the pose model reports a low-visibility corner (e.g., occluded by shadow
or glare), corner refinement crops around that approximate position and runs
the pose model again on just that small region. This recovers corners that the
full-image pose pass missed.

**Two modes:**
- **Pose model** (default): Uses named keypoints directly. Fast, single pass
  per corner. No classification needed — the matching keypoint name (UL, UR,
  etc.) IS the corner position.
- **Detection model**: Uses bounding box corners with geometric classification.
  Slower due to expand retries, less precise. Useful as a fallback when the
  pose model can't find any keypoint.

**Key optimization**: Crop size is automatically computed from the photo's
bounding box (1.2× the max dimension, minimum 640px). This ensures the photo
doesn't fill the entire crop, allowing the detection model to identify edge
contacts for classification.

### Performance (1512×2016 scan, 4 photos)

| Pipeline | Time | Notes |
|----------|------|-------|
| Detect + pose (baseline) | ~700ms | 4/4 photos found, but invisible corners stay invisible |
| + Corner refinement (pose) | ~2,500ms | Recovers all invisible corners, vis=1.0 for all |
| + Corner refinement (detection) | ~1,900ms | Recovers most corners, less precise than pose |
| + Sweep (XY) | ~6,900ms | Replaced by corner refinement — slower, same results |
| + Sweep + corner refine | ~8,900ms | Redundant — sweep is unnecessary with corner refinement |

**The "best" preset** now uses corner refinement instead of sweep:
```bash
photocrop --image scan.jpg --preset best
# Equivalent to:
photocrop --image scan.jpg \
  --corner-refine --corner-refine-model pose \
  --cv-refine --auto-refine \
  --crop warp-stretch --crop-margin 0.02 --border-fill white \
  --adaptive-margin
```

## Future Improvements

### Thread-Based Parallel Corner Refinement

Corner refinement processes 4 corners × N photos independently — embarrassingly
parallel. With a thread pool (4 threads):

| Pipeline | Sequential | 4 Threads | Speedup |
|----------|-----------|-----------|---------|
| Corner refine (pose) | ~2,500ms | ~700ms | 3.5× |
| Full pipeline | ~3,200ms | ~1,400ms | 2.3× |

In Python: `concurrent.futures.ThreadPoolExecutor` with ONNX Runtime
sessions (thread-safe for concurrent `run()` calls on CPU).

In Kotlin/C++/Rust: Thread pool with per-thread ONNX sessions, or a single
session with async inference queues.

### Batched Inference

Instead of 16 sequential pose calls (4 photos × 4 corners), batch all
corner crops into a single inference call. ONNX Runtime supports dynamic
batch sizes. This would reduce overhead and GPU utilization gaps.

### Adaptive Iteration Count

Currently runs 2 refinement iterations per corner. Since iteration 1 almost
always succeeds, we could:
- Default to 1 iteration
- Only run iteration 2 if the first iteration moved the corner significantly
- This would halve the inference calls for corner refinement

### Background Image Augmentation

For any future training (pose or detection), include 10–15% background-only
images (table surfaces, textures, gradients with no photos). This prevents
the objectness head from learning an "always fire" bias.

## Root Cause Analysis: V1 Pose Model Failure

The V1 pose model (mAP50-95=0.107) predicted all four keypoints near the top
of the image. This was caused by two configuration bugs:

1. **Wrong `flip_idx`**: Old config had `[1, 0, 3, 2]` (for TL/TR/BR/BL) but
   labels were LL/UL/UR/LR. Correct: `[3, 2, 1, 0]`.
2. **Wrong `kpt_shape`**: Old config had `[4, 2]` (x,y only) but labels used
   3 values (x,y,visibility). Correct: `[4, 3]`.

Both bugs were fixed in V2, but the pose approach was then superseded by
corner refinement, which reuses the existing pose model more effectively.