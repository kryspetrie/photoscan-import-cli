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
| Corner regression V2 | ✅ Production | 0.994 | 0.994 | 320×320 crop model recovers invisible corners (epoch 24) |
| Corner refinement (pose) | ✅ Current | — | — | Reuses pose model on corner crops; recovers invisible corners |
| Fiducial-pose segments V3 | 🔄 In Dev | 0.857 | 0.851 | Detects edge segments; assembles corners geometrically |

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

> **Note**: The previous `REMOVED/` directory that held failed approach code
> has been deleted. Those implementations are preserved only in git history.

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
│  Pose Refinement         │  (optional, --pose-refine)
│                          │  Refines approximate pose keypoints
└──────┬───────────────────┘
       │
       ▼
┌──────────────┐
│  Dedup       │  Remove duplicate detections / overlapping poses
└──────┬───────┘
       │
       ▼ (enabled by default; --no-warp-recover to disable)
┌──────────────────────────┐
│  Warp Recovery            │  Compute warp score per photo; re-pose with
│                            │  larger crops for high-warp detections
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Rescue (always on)      │  Sobel edge detection + line intersection
│                          │  Recovers invisible/low-visibility corners when
│                          │  a photo has < 3 visible corners
│                          │  → inferred corner positions from edge geometry
└──────┬───────────────────┘
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
  Crop / Warp → clean extracted photo
```

### Automatic Rescue

The **rescue stage** always runs after dedup. It uses Sobel edge detection and
line intersection to recover invisible or low-visibility corners when a photo
has fewer than 3 visible corners. This is a geometric fallback — no ML
inference required — that reconstructs missing corners from the photo's edge
structure.

Rescue is always on (no flag needed) because it has **zero cost** when all 4
corners are already visible. When a photo has < 3 visible corners, rescue adds
only ~280ms to recover the missing corners from edge geometry.

### Corner Refinement

Three corner refinement modes are available:

| Mode | Flag | Model | How it works |
|------|------|-------|--------------|
| Pose refine | `--pose-refine` | Pose model (pose_single_ep42) | Re-crop around detected photo, re-run pose for better localization |
| Corner regression | `--corner-refine` | Corner regression (corner-regression-v2) | 320×320 crop around each corner, specialized model finds precise corner position |
| CV refinement | `--cv-refine` | None (classical CV) | Sobel edge detection + line intersection, sub-pixel precision |

**Corner regression** (V2) is the recommended refinement mode. It uses a dedicated lightweight YOLO model (yolo26n-pose, 10MB ONNX) trained on 320×320 corner crops with:
- 1 class (`corner`) + 1 keypoint (precise corner position)
- 20-25% negative/background samples for robust rejection of non-corner regions
- Box mAP50=0.995, Pose mAP50-95=0.994
- Cross-photo validation: binary search on crop size avoids detecting adjacent photos in the corner crop

**Pose model** uses named keypoints directly — the matching keypoint name (UL, UR, etc.) IS the corner position. No classification needed.

**Detection model** uses bounding box corners with geometric classification. Slower due to expand retries, less precise. Useful as a fallback when the pose model can't find any keypoint.

## Fiducial-Pose Segment Model (In Development)

**Status**: V3 training completed (best: Pose mAP50=0.857, mAP50-95=0.851 at epoch 18). V4 training planned with improved hyperparameters.

The fiducial-pose segment model takes a fundamentally different approach to corner
detection. Instead of trying to detect corners directly, it detects the **visible
segments of each photo edge** (fiducial segments) using a YOLO pose model. Each
segment is represented as a pose object with keypoints at its two endpoints.

### Why Detect Segments Instead of Corners?

The core insight is that **corners are often invisible** (occluded by shadow,
glare, overlap, or cropping), but **edges are usually at least partially visible**.
A photo with 0 directly visible corners may still have 6+ visible edge segments
(2 along each visible edge). By detecting these segments and assembling them
geometrically, we can infer corner positions even when no corner is directly
visible — without relying on the current Sobel/line-intersection rescue fallback.

### How It Works

1. **Segment detection**: A YOLO pose model detects each visible edge segment as
   a separate object. Each segment has two keypoints — one at each endpoint.
2. **Geometric assembly**: Detected segments are grouped by edge (top, bottom,
   left, right) and their endpoint positions are combined to reconstruct the
   photo's corner positions. Segments along the same edge provide redundant
   evidence, making the assembly robust even when some segments are missed.
3. **Invisible corner recovery**: Where segments from two edges meet, a corner
   is inferred — even if no keypoint was directly detected at that intersection.

### Key Files

| File | Purpose |
|------|---------|
| `data_generator/generate_fiducial_pose.py` | Generates training data with edge-segment annotations |
| `training/train_fiducial_pose.py` | Trains the fiducial-pose YOLO model |

### Motivation vs. Current Pipeline

| Aspect | Current Pipeline | Fiducial-Pose Segments |
|--------|-----------------|----------------------|
| Invisible corners | Sobel rescue + corner refinement (multi-pass) | Geometric assembly from detected segments (single pass) |
| Inference passes | 1 detection + 1 pose + N corner refinements | 1 detection + 1 fiducial-pose (potentially) |
| Robustness to occlusion | Degrades when edges are also occluded | Multiple segments per edge provide redundancy |
| Speed | ~3,200ms with corner refinement | TBD — potentially much faster (fewer passes) |

### Performance (1512×2016 scan, 4 photos)

| Pipeline | Time | Notes |
|----------|------|-------|
| Detect + pose (baseline) | ~700ms | 4/4 photos found, but invisible corners stay invisible |
| + Rescue (if triggered) | ~280ms | Only when a photo has < 3 visible corners; zero cost otherwise |
| + Corner refinement (pose) | ~2,500ms | Recovers all invisible corners, vis=1.0 for all |
| + Corner refinement (detection) | ~1,900ms | Recovers most corners, less precise than pose |
| + Corner refinement (regression) | ~3,200ms | Best precision, dedicated 320×320 model |
| + Sweep (XY) | ~6,900ms | Replaced by corner refinement — slower, same results |
| + Sweep + corner refine | ~8,900ms | Redundant — sweep is unnecessary with corner refinement |

**The default preset** (`corner_refine`) uses corner regression for best precision:
```bash
photocrop --image scan.jpg
# Equivalent to:
photocrop --image scan.jpg \
  --pose-refine \
  --corner-refine --corner-refine-model regression \
  --crop warp-stretch --crop-margin 0.02 --border-fill edge-extend \
  --adaptive-margin \
  --warp-recover
```
Auto-rescue is always on (no flag needed). Warp recovery is on by default.

## Future Improvements

### Integrate Fiducial-Pose Segment Model into Inference Pipeline

Once training completes and validates well, the fiducial-pose segment model
could replace or complement the current rescue + corner refinement stages:

- **Primary corner source**: Use segment-based geometric assembly as the
  primary method for determining corner positions, reducing reliance on
  multi-pass corner refinement.
- **Hybrid pipeline**: Run fiducial-pose segments alongside the existing pose
  model. Use segments for corners where the pose model reports low visibility,
  and trust the pose model for high-visibility corners. This gives the best
  of both approaches.
- **Replace Sobel rescue**: Segment-based corner inference is more robust than
  Sobel edge detection + line intersection, especially in low-contrast or
  noisy scans. Evaluate whether fiducial segments can fully replace the
  current rescue stage.
- **Single-pass inference**: If the fiducial-pose model can detect both
  photos (via bounding boxes) and their edge segments (via keypoints) in
  one forward pass, it could replace both the detection and pose models,
  collapsing the entire pipeline to a single inference call + geometric
  assembly.

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