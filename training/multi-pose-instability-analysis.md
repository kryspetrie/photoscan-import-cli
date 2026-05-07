# Multi-Photo Pose Training Instability Analysis

**Date:** 2026-05-04  
**Status:** Analysis only — no changes made to code or running processes

---

## Summary

The multi-photo pose model (`train_pose_multi.py`) shows significant training instability starting around epoch 13, with mAP50-95 oscillating between 0.715 and 0.956 across epochs. Meanwhile, the single-photo pose model (`train_pose.py`) trains stably and has reached mAP50-95 = 0.964 (epoch 14). Three root causes have been identified, with a critical configuration bug as the primary suspect.

---

## Observables: The Instability Pattern

| Epoch | mAP50-95 | Val box | Val pose | Val cls | Pattern |
|------:|---------:|--------:|---------:|--------:|---------|
| 6 | 0.862 | 0.616 | 0.034 | 0.181 | Stable convergence |
| 13 | 0.913 | 0.446 | 0.017 | 0.149 | Peak performance |
| 16 | 0.828 | 0.710 | 0.015 | 0.210 | Degrading |
| **17** | **0.715** | **1.145** | **0.049** | **0.355** | **Near-catastrophe** |
| 19 | 0.932 | 0.398 | 0.012 | 0.143 | Recovery |
| 23 | 0.784 | 0.876 | 0.016 | 0.258 | Another severe dip |
| 25 | 0.956 | 0.336 | 0.011 | 0.120 | New best |
| 26 | 0.940 | 0.394 | 0.016 | 0.124 | Good |

The oscillation amplitude is **increasing over time**: epochs 1–12 were stable, then from epoch 13 onward, swings of ±0.15–0.24 in mAP50-95. This is not typical training noise — it's instability.

---

## Root Cause #1: Learning Rate Configuration Discrepancy (CRITICAL)

### The Bug

Both models are configured with `lr0=0.001` and `optimizer=auto`, yet they end up with drastically different effective learning rates:

| | Single-Photo | Multi-Photo |
|---|---|---|
| **Param groups** | 8 (2 distinct LRs) | 3 (1 unified LR) |
| **Weight LR (post-warmup)** | ~0.0297 | ~0.00194 |
| **Bias LR (post-warmup)** | ~0.0099 | ~0.00194 |
| **Effective LR ratio** | 15.3× higher for weights | — |
| **Warmup LR (ep 1)** | 0.00996 | 0.000664 |

The multi-photo model is training at **~15× lower effective learning rate** for its weight parameters. This is a massive, unintended difference.

### Why This Happens

In Ultralytics (v8.4.33), `optimizer=auto` creates parameter groups based on the model architecture. The number of groups (8 vs 3) suggests that the two training runs are **using different optimizer configurations despite identical `args.yaml` settings**. 

The likely cause: Ultralytics' `auto` optimizer mode creates separate param groups for backbone weights, backbone biases, and per-decoder-head groups. The difference in group count (8 vs 3) indicates that one of the models may have been initialized with a different head structure or the optimizer was constructed differently. Since both use `yolo26s-pose.pt` and `nc=1`, `kpt_shape=[4,3]`, the model architecture should be identical.

**Hypothesis:** This may be a side effect of the `deterministic: true` flag combined with different augmentation pipelines causing Ultralytics to construct the optimizer differently during the first forward pass. Or it could be a bug in how Ultralytics handles the `auto` optimizer for pose models with different augmentation configs.

### Impact

The 15× lower LR means:
- The multi-photo model learns much more slowly per gradient step
- It's more susceptible to getting stuck and then having sudden jumps when it finally escapes local minima
- The oscillations could be the model "catching up" in bursts rather than converging smoothly

---

## Root Cause #2: Aggressive Augmentation Incompatible with Pose (HIGH)

### The Bug

The multi-photo training uses augmentation settings that are **explicitly warned against** in the project's own `PITFALLS.md` and `REFACTORING_PLAN.md`:

| Parameter | Single | Multi | Known Issue |
|-----------|--------|-------|-------------|
| `mosaic` | 0.3 | **1.0** | Mosaic combines 4 images, distorting keypoint spatial relationships |
| `mixup` | 0.0 | **0.1** | **PITFALLS.md: "mixup=0.0 for pose — blending images moves keypoints to invalid positions"** |
| `copy_paste` | 0.0 | **0.1** | Copies objects between images, can create physically impossible keypoint arrangements |
| `scale` | 0.2 | **0.5** | 50% scale range on small objects dramatically changes keypoint positions |
| `translate` | 0.05 | **0.1** | More translation = more keypoint displacement |
| `hsv_s` | 0.3 | **0.7** | Very aggressive saturation augmentation |

### Why Mosaic=1.0 Is Especially harmful

With `mosaic=1.0`, **every training image** is a composite of 4 source images. For bounding box detection, this is fine — the box just moves with the crop. For **pose/keypoint models**, mosaic is problematic because:

1. Keypoint positions are relative to the final mosaic composition, not the original image
2. When a 4-photo mosaic crops a small photo, the keypoints shift by the mosaic offset
3. The model sees keypoints in "wrong" spatial contexts relative to the photo content
4. Ultralytics does handle keypoint transforms during mosaic, but the transform chain (mosaic → scale → translate → fliplr) compounds errors

### Why Mixup Is Catastrophic for Pose

The project's own documentation states clearly: **"mixup=0.0 for pose — blending images moves keypoints to invalid positions."** 

When two images are blended with alpha compositing, the keypoint locations are interpolated between the two images. But keypoints represent physical corner positions of photographs — a blended keypoint halfway between two different photos' corners is **physically meaningless**. The model receives a training signal saying "this corner is at position (0.4, 0.3)" when in reality neither photo has a corner there.

This was known and correctly set to 0.0 in the single-photo trainer. The multi-photo trainer set it to 0.1, likely copied from detection training defaults without considering the pose-specific constraint.

### Impact of copy_paste=0.1

Copy-paste randomly inserts objects from other images. For pose models, this means a photo bounding box with keypoints can appear at a random location in a different scene. While the keypoint transform should be correct, it adds another source of spatial inconsistency, especially combined with mosaic.

---

## Root Cause #3: Multi-Photo Is a Harder Task (MODERATE, but expected)

### Dataset Differences

| Metric | Single-Photo | Multi-Photo |
|--------|-------------|-------------|
| Train images | 4,000 | 4,000 |
| Val images | 1,000 | 1,000 |
| Train objects | 4,000 | 10,049 |
| Val objects | 1,000 | 2,507 |
| Avg bbox area | 73.1% of frame | 19.6% of frame |
| Photos per image | 1 (always) | 1–4 (uniform) |
| Objects/image (val) | 1.0 | 2.5 |

The multi-photo task is genuinely harder because:

1. **Photos are 3.7× smaller** relative to the frame (19.6% vs 73.1% mean area)
2. **Variable object count** requires the model to decide "how many" in addition to "where"
3. **Occlusion and overlapping** when multiple photos are placed close together
4. **Different scale regimes** — keypoints at 20% of frame need different precision than at 80%

However, the multi model has **2.5× more training objects per image** (10,049 vs 4,000), which should partially compensate. The instability is **not** primarily caused by task difficulty — the single model handles its simpler task stably, and we'd expect the multi model to converge more slowly but still monotonically, not oscillate wildly.

---

## Root Cause Assessment

| Cause | Severity | Confidence | Fix Difficulty |
|-------|----------|------------|----------------|
| LR config (3 vs 8 param groups) | CRITICAL | High | Medium (requires Ultralytics investigation) |
| Mixup=0.1 on pose model | HIGH | Very High | Trivial (set to 0.0) |
| Mosaic=1.0 on pose model | HIGH | High | Easy (reduce to 0.3) |
| Copy_paste=0.1 on pose model | MODERATE | Medium | Easy (set to 0.0) |
| Scale=0.5 on small objects | MODERATE | Medium | Easy (reduce to 0.3) |
| HSV_S=0.7 too aggressive | LOW | Low | Easy (reduce to 0.3) |
| Task difficulty | EXPECTED | High | N/A (inherent) |
| Dataset too small | LOW | Low | Medium (regenerate at 2–4×) |

---

## Is the Dataset Too Small?

**Probably not.** 1,000 validation images with 2,507 objects gives 62 batches per evaluation. The instability shows epoch-to-epoch variance of ±0.15 in mAP50-95, which is far too large for statistical noise at this sample size (expected variance ~0.005–0.01). The oscillation pattern (dips to 0.715 then recovery to 0.93+) is not statistical noise — it's training dynamics instability.

If we fix the augmentation and LR issues, 4,000/1,000 train/val split may well be sufficient. Expanding to 8,000/2,000 or 16,000/4,000 would add robustness but isn't the primary fix needed.

---

## Recommended Fixes (Priority Order)

1. **Set `mixup=0.0`** in `train_pose_multi.py`. The project's own documentation warns against this for pose models. This is the most certain fix.

2. **Reduce `mosaic` from 1.0 to 0.3** (matching the single-photo config). Mosaic is less problematic than mixup but still disruptive to keypoint spatial relationships. The single-photo model uses 0.3 and trains stably.

3. **Set `copy_paste=0.0`** in `train_pose_multi.py`. Copy-paste creates spatially inconsistent keypoints.

4. **Reduce `scale` from 0.5 to 0.3**. With photos at ~20% of frame, a 0.5 scale range means photos can be shrunk to ~10% or enlarged to ~30%, a 3× range that destabilizes keypoint regression.

5. **Reduce `hsv_s` from 0.7 to 0.3** and `hsv_v` from 0.4 to 0.3** to match the single-photo config.

6. **Investigate the LR/parameter group discrepancy.** Both models use `lr0=0.001` and `optimizer=auto`, but end up with 15× different effective LRs. This may require inspecting the Ultralytics optimizer construction or explicitly setting the optimizer to `SGD` with known LR schedules.

7. **After fixing the above, retrain and evaluate.** Only consider expanding the dataset (2× or 4×) if instability persists after configuration fixes.

---

## Why Not "Just Train Longer"?

The oscillation is **diverging**, not converging. Epochs 6–12 were stable (0.86–0.91), but epochs 17 and 23 showed severe regressions (0.715, 0.784). The model is not on a path to convergence — it's bouncing between modes of the loss landscape, likely because the aggressive augmentation and low LR prevent stable gradient descent. Training 100 or 300 epochs won't fix a misconfigured training pipeline.

---

## Comparison: Single vs Multi Training Curves

### Single-Photo (stable)
```
Epoch  mAP50-95  Val pose
  1     0.808     0.371
  6     0.874     0.035
  9     0.924     0.015
  10    0.932     0.011  ← best
  14    0.964     0.009  ← new best
```
Monotonic improvement in val pose loss. mAP50-95 climbs steadily.

### Multi-Photo (unstable)
```
Epoch  mAP50-95  Val box   Val pose
  6     0.862     0.616     0.034
  13    0.913     0.446     0.017
  17    0.715     1.145     0.049  ← crash
  19    0.932     0.398     0.012
  23    0.784     0.876     0.016  ← crash
  25    0.956     0.336     0.011  ← best
```
Wild swings in val_box_loss (0.336 → 1.145 → 0.398). mAP oscillates ±0.2. The model periodically "forgets" how to detect bounding boxes, then recovers.

---

## Appendix: Running Process State (as of analysis)

- **Detection model**: Finished (57 epochs, early stopped, mAP50-95=0.995)
- **Pose single-photo**: Epoch 15/300, mAP50-95=0.964 at best, still improving
- **Pose multi-photo**: Epoch 26/100, mAP50-95=0.956 at best, oscillating