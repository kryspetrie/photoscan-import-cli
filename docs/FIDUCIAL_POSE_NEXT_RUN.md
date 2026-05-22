# Fiducial Pose Training — Run History & V3 Plan

## Training Run History

### V1 — Baseline (Epochs 1–28, stopped)
| Parameter | Value |
|-----------|-------|
| model | yolo26s-pose |
| optimizer | auto → AdamW lr=0.002 |
| mosaic | 0.3 |
| cls | 0.5 (default) |
| rle | 1.0 (default) |
| box | 7.5 (default) |
| lr0 | 0.001 (overridden by auto to 0.002) |
| Best Pose mAP50 | **0.812** (ep 15) |
| Best Pose mAP50-95 | **0.798** (ep 15) |

**Issues**: RLE loss dominated at 47%, cls wasted 15% on nc=1, mosaic creates false positives on synthetic data.

**Backup**: `training/backups/v1-fiducial-pose-segments/`

### V2 — Aggressive Rebalance (Epochs 1–21, stopped)
| Parameter | Value | Change from V1 |
|-----------|-------|----------------|
| optimizer | AdamW | Explicit (kept) |
| lr0 | 0.001 | Half of V1's effective 0.002 |
| mosaic | 0.0 | OFF (correct) |
| close_mosaic | 0 | No mosaic to phase out (correct) |
| cls | **0.0** | Removed classification loss ❌ |
| rle | **0.3** | Over-reduced from 1.0 |
| box | 7.5 | Unchanged |

**Result**: **Failed.** Pose mAP50 stuck at ~0.24 across 21 epochs with no improvement. Val loss dropped (5.0→0.4) but detection never improved.

**Root causes**:
1. `cls=0.0` eliminated foreground/background discrimination — the cls head provides the "is this a real segment?" signal even for nc=1
2. `rle=0.3` over-reduced keypoint representation loss, causing `box` to fill the void at 41% of total gradient — worst possible loss for thin segments
3. `lr0=0.001` was 2× slower than V1, compounding the detection failure

**Backup**: `training/backups/v2-fiducial-pose-segments/`

### V3 — Corrected Rebalance (Epochs 1–42+, still running)

| Parameter | V1 | V2 | V3 | Reasoning |
|-----------|-----|-----|-----|-----------|
| `lr0` | 0.002 (auto) | 0.001 | **0.002** | Match V1's proven convergence rate |
| `mosaic` | 0.3 | 0.0 | **0.0** | Synthetic data has built-in compositing ✅ |
| `close_mosaic` | 10 | 0 | **0** | No mosaic to phase out ✅ |
| `cls` | 0.5 | 0.0 | **0.3** | Reduced but NOT zero; still needed for fg/bg discrimination |
| `rle` | 1.0 | 0.3 | **0.5** | Moderate reduction — was 47%, target ~30% |
| `box` | 7.5 | 7.5 | **4.0** | Thin segment bboxes have poor IoU; reduce box dominance |
| `optimizer` | auto | AdamW | **AdamW** | Explicit control ✅ |

**Best results**: Pose mAP50=**0.857**, mAP50-95=**0.851** (epoch 18)

**V3 vs V1 comparison**:
| Metric | V1 Best | V3 Best | Δ |
|--------|---------|---------|---|
| Pose mAP50 | 0.803 | **0.857** | +6.7% |
| Pose mAP50-95 | 0.812 | **0.851** | +4.8% |
| Epochs to best | 15 | 9 | 6 fewer |

Training plateaued after epoch 18 and was still running at epoch 42+ (early stopping at epoch 48 with patience=30).

**Backup**: `training/backups/v3-fiducial-pose-segments/`
**ONNX model**: `models/fiducial_pose_v3.onnx` (output shape [1, 300, 12])

---

## V3 Model Integration

The V3 fiducial pose model (2 keypoints per segment) has been integrated into `photocrop.py` in two ways:

### 1. Corner Refinement (`--corner-refine-model segment`)
Uses the segment model to refine corners by finding convergence clusters of segment endpoints. Added convergence-clustering logic to find where 2+ segments meet at a corner.

### 2. Pose Refinement (`--pose-refine --segment-model`)
When both `--pose-refine` and `--segment-model` are provided, the refinement stage (Stage 2b) uses the segment model instead of re-running the 4-keypoint pose model. The segment model finds boundary segments, clusters endpoints into corners, and matches them to the initial pose keypoints to refine positions.

**Current status**: Segment refinement shows promise on specific corners (e.g., 2.4px vs 8.7px for some corners) but is not yet consistently better than pose refinement overall. The matching heuristic needs further tuning for robustness.

---

## Potential Future Improvements

- More diverse training data (add `one_edge` scene mode, ~10% coverage)
- Reduce match distance threshold in segment-based keypoint matching
- Test segment model on more diverse real-world images
- Consider training a corner-specific model (4 keypoints) with V3's improved loss weights