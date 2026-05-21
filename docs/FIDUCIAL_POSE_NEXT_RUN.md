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

---

## V3 — Corrected Rebalance

### Changes from V1

| Parameter | V1 | V2 | V3 | Reasoning |
|-----------|-----|-----|-----|-----------|
| `lr0` | 0.002 (auto) | 0.001 | **0.002** | Match V1's proven convergence rate |
| `mosaic` | 0.3 | 0.0 | **0.0** | Synthetic data has built-in compositing ✅ |
| `close_mosaic` | 10 | 0 | **0** | No mosaic to phase out ✅ |
| `cls` | 0.5 | 0.0 | **0.3** | Reduced but NOT zero; still needed for fg/bg discrimination |
| `rle` | 1.0 | 0.3 | **0.5** | Moderate reduction — was 47%, target ~30% |
| `box` | 7.5 | 7.5 | **4.0** | Thin segment bboxes have poor IoU; reduce box dominance |
| `optimizer` | auto | AdamW | **AdamW** | Explicit control ✅ |

### Expected Loss Budget (approximate)

| Loss | V1 % | V2 % | V3 Target % |
|------|------|------|-------------|
| box | 14% | 41% | ~25% |
| pose | 5% | 6% | ~20% |
| cls | 15% | 0% | ~15% |
| rle | 61% | 43% | ~30% |
| kobj | 5% | 9% | ~10% |

### Execution
- Dataset: Same as V1/V2 (no regeneration needed)
- Command: `python3 train_fiducial_pose.py --epochs 150 --patience 30 --batch 16 --cache ram --workers 4`
- All V3 defaults are in the script; no CLI overrides needed