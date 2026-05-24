# V4 Fiducial Training Plan

## Overview

Two parallel tracks:

1. **V4 Segment Model**: Retrain the 2-keypoint segment detector with improved training strategy
2. **V4 Corner Regression Head**: New lightweight model that predicts precise corner position from a tight corner crop

---

## Part 1: V4 Segment Model — Should We Use a Larger YOLO?

### Current Model: yolo26s-pose (11.8M parameters, 23MB ONNX)

| Model | Params | ONNX Size | Notes |
|-------|--------|-----------|-------|
| yolo26n-pose | 3.7M | 7.5MB | Too small for keypoint precision |
| **yolo26s-pose** | **11.8M** | **23MB** | **Current — good balance** |
| yolo26m-pose | ~25M | ~50MB | 2× params, diminishing returns on 640×640 |
| yolo26l-pose | ~43M | ~85MB | 4× params, overkill for this task |

### Recommendation: **Stay with yolo26s-pose**

Reasons:
1. **The problem isn't model capacity — it's training dynamics.** V3 peaked at epoch 18 with 11.8M params. The model learned the task well (mAP50-95=0.843) but then forgot/overfit. A larger model would memorize even faster and overfit even harder.

2. **m (2×) or l (4×) models are overkill for 640×640 crops with 1 class and 2 keypoints.** The task is "find line segments" — not a complex multi-class detection problem. The s model has plenty of capacity.

3. **Inference speed matters.** This model runs during corner refinement, which happens per-corner per-photo. With 4 corners × multiple photos, inference time multiplies. yolo26s-pose at 23MB is already substantial; yolo26m at ~50MB would be noticeably slower.

4. **The dataset is small (3000 train images).** Larger models need more data to avoid overfitting. With only 3K training images, a larger model would overfit even faster than V3 did.

5. **Better training strategy > bigger model.** The fix for plateau/decline is better lr scheduling, data augmentation, and loss weighting — not more parameters.

### When to Consider a Larger Model

- If V4 with yolo26s-pose plateaus below mAP50-95=0.90 despite optimal training
- If the dataset grows to 10K+ images
- If we need sub-pixel precision on larger images (>640×640)

---

## Part 2: V4 Training Strategy

### What Went Wrong with V3

```
Epoch  1: mAP50-95=0.055  (just starting)
Epoch 10: mAP50-95=0.791  (rapidly learning)
Epoch 18: mAP50-95=0.843  (PEAK — best checkpoint)
Epoch 23: mAP50-95=0.517  (wild oscillation)
Epoch 43: mAP50-95=0.768  (partial recovery, still declining)
```

The model overshoots the keypoint minima because:
- lr0=0.002 is too aggressive for the fine-tuning phase
- No lr scheduling beyond cosine decay to lrf=0.01
- Training continued 28 epochs past the peak

### V4 Training Changes

| Parameter | V3 | V4 | Rationale |
|-----------|-----|-----|-----------|
| model | yolo26s-pose | yolo26s-pose | Same — not the problem |
| epochs | 150 | 50 | V3 peaked at 18; stop early |
| patience | 30 | 15 | Stop sooner when not improving |
| optimizer | AdamW | AdamW | Good for keypoint tasks |
| lr0 | 0.002 | 0.001 | Half — gentler convergence |
| lrf | 0.01 | 0.01 | Same cosine endpoint |
| warmup_epochs | 3 | 5 | Longer warmup for stability |
| box | 4.0 | 3.0 | Reduce box weight — thin segments |
| cls | 0.3 | 0.3 | Same — moderate classification |
| pose | 12.0 | 12.0 | Same |
| kobj | 1.0 | 2.0 | **Double** — emphasize keypoint confidence |
| rle | 0.5 | 0.8 | **Increase** — keypoint localization is the goal |
| scale | 0.3 | 0.5 | More size variation |
| degrees | 5.0 | 10.0 | More rotation variation |
| translate | 0.1 | 0.2 | More position variation |
| mosaic | 0.0 | 0.0 | Still OFF |
| flipud | 0.0 | 0.0 | Still OFF |

### Key Changes Explained

1. **lr0=0.001** (was 0.002): V3 overshot the keypoint minima. Halving the learning rate gives finer convergence.

2. **kobj=2.0** (was 1.0): The keypoint objectness loss determines how confident the model is about keypoint visibility. V3 had low-confidence keypoints (0.2–0.4). Doubling kobj encourages the model to be more confident about detected keypoints.

3. **rle=0.8** (was 0.5): RLE (rotation-equivariant keypoint loss) is the primary loss for keypoint localization precision. Increasing it makes the model prioritize accurate endpoint placement over box accuracy.

4. **More augmentation**: scale=0.5, degrees=10, translate=0.2 give more training diversity, reducing overfitting.

5. **50 epochs max, patience=15**: V3 peaked at 18. We shouldn't train for 150 epochs. With patience=15, training stops at epoch ~33 if it peaks at 18 again.

---

## Part 3: V4 Corner Regression Head

### Concept

Instead of finding boundary segments and clustering endpoints, train a **lightweight model that directly predicts the corner position** from a tight crop around the approximate corner.

**Pipeline**: detect → pose (approximate) → per-corner crop → regression head → precise corner

### Architecture Options

#### Option A: YOLO Pose (1 class, 1 keypoint)
- Use yolo26n-pose with 1 class `corner` and 1 keypoint (the corner position)
- Input: 256×256 or 320×320 corner crop
- Output: bounding box + 1 keypoint (corner position)
- Pros: Reuses existing infrastructure, well-understood
- Cons: Bounding box is meaningless for a single point, adds overheard

#### Option B: Simple CNN Regression Head
- Small CNN (MobileNetV3-small or custom 4-layer) that regresses (dx, dy)
- Input: 256×256 RGB crop centered on approximate corner
- Output: (dx, dy) offset from crop center to exact corner
- Pros: Tiny model (<1M params), very fast, purpose-built
- Cons: Custom training code, no YOLO infrastructure

#### Option C: YOLO Detect (1 class, predict corner as 1×1 box)
- Use yolo26n (detection only) with 1 class `corner`
- Train on 1×1 pixel "bounding box" at the corner position
- Input: 256×256 crop, output: tiny bounding box at corner
- Pros: Simple, reuses detection infrastructure
- Cons: YOLO isn't designed for 1×1 boxes; poor fit for the task

### Recommendation: Option A — YOLO Pose with 1 keypoint

This is the simplest to implement and best aligned with our existing infrastructure:
- Reuses the same data generator (with modifications for corner crops)
- Reuses the same training script (with modifications for 1 kp)
- Reuses the same inference pipeline ([`run_pose`] → just read the 1 keypoint)
- yolo26n-pose at 3.7M params is tiny and fast for this simple task
- The bounding box provides context (what part of the photo is visible)

### Training Data for Corner Regression

Generate corner crops from the existing dataset + new generator:

1. **From existing fiducial_pose data**: For each segment with a visible corner keypoint (v=2), extract a 320×320 crop centered on that keypoint. The keypoint becomes the regression target.

2. **New generator mode**: Generate 640×640 images with 1 corner visible, then extract corner crops at multiple zoom levels. This gives more training diversity.

3. **Augmentation**: Heavy augmentation on the corner crops (rotation, scale, brightness) since the model should be robust to different corner appearances.

### Label Format (YOLO-pose, 1 keypoint)

```
0 cx cy w h kpx kpy kpv
```

Where:
- class 0 = `corner`
- `(cx, cy, w, h)` = bounding box of visible photo edges in crop
- `(kpx, kpy)` = exact corner position in crop (normalized)
- `kpv` = 2 (visible) or 0 (off-screen — but these won't be in the dataset)

### Data Generator Design

```
generate_corner_regression.py
  --mode batch
  --train-count 3000
  --val-count 500
  --source ./images
  --output ../data_corner_regression
```

Scene modes:
- 50%: Single corner visible, 1-2 edges entering (L-shape)
- 25%: Single corner, 1 edge only (partial visibility)
- 15%: No corner visible, edges only (negative — predicts nearest edge intersection)
- 10%: Two corners visible in crop (harder case)

The generator will:
1. Compose a photo on a 640×640 canvas (reuse existing compositing)
2. Extract corners that are visible (visibility ≥ 1)
3. For each visible corner, create a 320×320 (or 256×256) crop centered near the corner with random jitter
4. Transform the annotation coordinates accordingly
5. Save crop + label

### Implementation Plan

1. ✅ Write V4 training strategy doc
2. ✅ Create `generate_corner_regression.py` data generator
3. ✅ Create `train_corner_regression.py` training script
4. ✅ Generate training data (10K train + 2K val)
5. ✅ Train the corner regression model (V2, best epoch 24: box mAP50-95=0.773, pose mAP50-95=0.994)
6. ✅ Integrate into pipeline (corner regression model via `--corner-refine`)
7. ✅ Benchmark on val set (cross-photo validation fix, 93 tests pass)
8. ⬜ Retrain V4 segment model with improved training strategy

---

## Timeline

| Step | Time | Notes |
|------|------|-------|
| Corner regression data generator | 1-2 hours | Based on existing generate_fiducial_pose.py |
| Training script | 30 min | Based on train_fiducial_pose.py |
| Generate data (3000 train + 500 val) | 10-30 min | CPU-bound compositing |
| Train corner regression (yolo26n-pose) | 1-2 hours | Small model, fast convergence |
| Integrate + benchmark | 30 min | Add regression model to pipeline |
| V4 segment model retraining | 2-4 hours | Can run in parallel with integration |