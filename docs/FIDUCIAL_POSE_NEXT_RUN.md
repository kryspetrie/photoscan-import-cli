# Fiducial Pose Model — Next Training Run Plan

> **Status**: Planning. No changes to the current training run (epoch ~24/150).
> Will execute after current run completes or is manually stopped.

---

## Why Another Run Is Needed

Analysis of the current training (24/150 epochs) reveals several configuration
and dataset issues that are limiting performance:

| Issue | Impact | Severity |
|-------|--------|----------|
| Mosaic augmentation at 0.3 | Creates false segment boundaries at composite seams; our synthetic data already has multi-photo scenes and varied backgrounds | 🔴 High |
| RLE loss weight too high (1.0) | Consumes 47–51% of total gradient budget; steals capacity from box/pose losses | 🟡 Medium |
| Classification loss weight too high (0.5) | Wasted on nc=1; accounts for 22% of loss despite having one class | 🟡 Medium |
| No single-segment scenes in dataset | 0% training images have exactly 1 visible segment; production may encounter this | 🟡 Medium |
| `optimizer=auto` overrides learning rate | Specified lr0=0.001 was overridden to AdamW lr=0.002; takes away explicit control | 🟢 Low |

---

## Changes from Current Run → Next Run

### 1. Disable Mosaic Augmentation 🔴

```yaml
# CURRENT (problematic)
mosaic: 0.3
close_mosaic: 10

# NEXT RUN
mosaic: 0.0
close_mosaic: 0   # (irrelevant when mosaic=0, but explicit)
```

**Rationale**: Our synthetic training data already includes:
- 35% grid scenes with 2–4 overlapping photos
- Varied backgrounds, textures, and glare
- Multi-photo compositions with gaps between photos

Mosaic augmentation combines 4 such images into a composite, creating
artificial boundaries where segments from different photos meet. These
boundary regions produce **false positive segment detections** that the
model then learns, degrading precision.

**Also disable all composite augmentations for the same reason**:
```yaml
mixup: 0.0       # (already 0 — keep)
copy_paste: 0.0  # (already 0 — keep)
cutmix: 0.0      # (already 0 — keep)
```

### 2. Reduce RLE Loss Weight 🟡

```yaml
# CURRENT
rle: 1.0

# NEXT RUN
rle: 0.3
```

**Rationale**: RLE (Run-Length Encoding) loss currently dominates training,
consuming 47–51% of the total loss budget. The model spends most of its
capacity learning keypoint *representation* rather than keypoint *precision*.
Since our keypoints are simple (2 per segment, on visible edges), the full
RLE weight is excessive. Reducing to 0.3 rebalances gradient allocation
toward box and pose coordinate losses.

Current loss composition (epoch 24):
| Component | Value | % of Total |
|-----------|-------|------------|
| RLE | 2.823 | **47.1%** |
| Box | 1.235 | 20.6% |
| Cls | 1.304 | 21.7% |
| Kobj | 0.437 | 7.3% |
| Pose | 0.193 | 3.2% |
| DFL | 0.008 | 0.1% |

With `rle=0.3` and `cls=0.0` (see below), the rebalanced composition
would prioritize box and pose losses — the ones that matter for our task.

### 3. Eliminate Classification Loss 🟡

```yaml
# CURRENT
cls: 0.5

# NEXT RUN
cls: 0.0
```

**Rationale**: With `nc=1` (single class `photo_segment`), the classification
head is trivially "always predict photo_segment." Yet cls_loss accounts for
22% of total training signal — pure waste. Setting `cls=0.0` redirects all
gradient budget to localization tasks (box, pose, kobj).

**Note**: This requires verifying that YOLO handles `cls=0.0` gracefully
for a single-class task. If it causes NaN or zero-gradient issues, fall
back to `cls=0.1`.

### 4. Fix Learning Rate Control 🟢

```yaml
# CURRENT
optimizer: auto   # overrides lr0=0.001, sets AdamW lr=0.002

# NEXT RUN
optimizer: AdamW
lr0: 0.001
```

**Rationale**: The `auto` optimizer selection overrode our specified lr0=0.001
and chose AdamW with lr=0.002. This is likely too aggressive. For the next
run, we explicitly set AdamW with lr=0.001 for more stable convergence.
The smaller LR also compensates for the loss weight changes, which shift
gradient magnitudes.

### 5. Add Single-Segment Scenes to Dataset 🟡

Current dataset composition (training):
| Segments/image | Count | Percentage |
|----------------|-------|-----------|
| 0 (background) | 282 | 9.4% |
| **1 segment** | **0** | **0%** ← gap! |
| 2 segments | 1516 | 50.5% |
| 3+ segments | 1202 | 40.1% |

In production, a single edge running across the crop (no corner visible)
produces exactly 1 segment. The dataset must cover this case.

**Change to `data_generator/generate_fiducial_pose.py`**:

```python
# CURRENT
MODE_WEIGHTS = {
    'one_corner':     0.45,
    'grid':           0.35,
    'two_corners':    0.15,
    'no_photo':       0.05,
}

# NEXT DATASET GENERATION
MODE_WEIGHTS = {
    'one_corner':      0.35,   # Reduced from 0.45
    'one_edge':        0.10,   # NEW: single edge across frame, 1 segment
    'grid':            0.35,
    'two_corners':     0.15,
    'no_photo':        0.05,
}
```

This adds a new scene mode `one_edge` that places a single photo edge running
across the 640×640 crop with no corner visible, producing exactly 1 segment
with 2 keypoints (the intersections of the edge with the crop boundary).

**Implementation**: Add a `generate_one_edge_scene()` function to the data
generator that:
1. Places a single photo edge (horizontal or vertical) across the crop
2. Computes the 2 intersection keypoints where the edge meets the crop boundary
3. Labels the segment with visibility=2 (both endpoints visible)
4. Generates ~10% of training data with this distribution

After regenerating the dataset, verify the distribution:
- ~9% backgrounds (0 segments)
- ~10% one-edge (1 segment) ← NEW
- ~35% one-corner (2 segments)
- ~35% grid (2–8 segments)
- ~11% two-corners (3 segments)

### 6. All Other Settings — Keep Current Values

These remain unchanged from the first run:

```yaml
model: yolo26s-pose.pt     # Small model is sufficient for 1 class + 2 keypoints
batch: 16                    # Appropriate for CPU
imgsz: 640                   # Matches inference pipeline
patience: 30                 # Standard early stopping
epochs: 150                  # Full training
lrf: 0.01                    # Standard LR schedule
momentum: 0.9                # AdamW beta1 (with explicit optimizer)
weight_decay: 0.0005         # Standard
warmup_epochs: 3.0           # Standard
box: 7.5                     # Standard
dfl: 1.5                     # Standard
pose: 12.0                   # Standard for pose tasks
kobj: 1.0                    # Standard for keypoint objectness
scale: 0.3                   # Moderate (photos can be various scales)
degrees: 5.0                 # Conservative (segments don't rotate much)
translate: 0.1               # Moderate
flipud: 0.0                  # No vertical flip (changes orientation semantics)
fliplr: 0.5                  # Horizontal flip OK with flip_idx=[1,0]
hsv_h: 0.015                 # Mild color augmentation
hsv_s: 0.3                   # Moderate saturation
hsv_v: 0.3                   # Moderate brightness
cache: ram                   # Fast for CPU training
device: cpu                  # Current hardware
```

**Why not switch to a larger model?** The task is fundamentally simple:
1 class, 2 keypoints per instance, geometrically constrained endpoints.
The `s` variant has adequate capacity. If next-run results plateau below
pose mAP50=0.85, then consider `yolo26m-pose`. But loss rebalancing and
fixing mosaic should unlock significant gains first.

---

## Complete Next-Run Training Command

```bash
python3 training/train_fiducial_pose.py \
    --model yolo26s-pose.pt \
    --epochs 150 \
    --patience 30 \
    --batch 16 \
    --imgsz 640 \
    --cache ram \
    --device cpu \
    --name fiducial-pose-v2
```

**With these changes in `train_fiducial_pose.py` defaults**:

```python
# Loss weights (rebalanced)
cls=0.0,          # Eliminated — nc=1, classification is trivial
rle=0.3,           # Reduced from 1.0 — RLE dominates gradient budget

# Augmentation (mosaic disabled)
mosaic=0.0,        # DISABLED — synthetic data already has multi-photo composites
close_mosaic=0,    # Irrelevant with mosaic=0
mixup=0.0,         # Disabled (was already)
copy_paste=0.0,    # Disabled (was already)

# Optimizer (explicit, no auto override)
optimizer="AdamW",  # Explicit — prevents lr0 override
lr0=0.001,         # Controlled LR (auto was using 0.002)
lrf=0.01,           # Standard cosine decay endpoint
momentum=0.9,       # AdamW beta1

# Everything else unchanged
scale=0.3, degrees=5.0, translate=0.1,
fliplr=0.5, flipud=0.0,
hsv_h=0.015, hsv_s=0.3, hsv_v=0.3,
box=7.5, dfl=1.5, pose=12.0, kobj=1.0
```

---

## Run Sequence

### Step 1: Let Current Training Finish
- Current run continues to epoch 150 or early stop (patience=30)
- Even if it early-stops, we get a usable `best.pt` model
- Document final metrics for comparison

### Step 2: Update Data Generator
- Add `one_edge` scene mode to `generate_fiducial_pose.py`
- Regenerate dataset:
  ```bash
  cd data_generator
  python3 generate_fiducial_pose.py --mode batch --train-count 6000 --val-count 1000 --source ./images
  ```
- Verify dataset distribution includes ~10% single-segment scenes

### Step 3: Update Training Script Defaults
- Edit `training/train_fiducial_pose.py` with the parameter changes above
- Save as a new version or update in-place with comments

### Step 4: Launch Next Training Run
```bash
nohup python3 training/train_fiducial_pose.py \
    --name fiducial-pose-v2 \
    > /tmp/train_fiducial_pose_v2.log 2>&1 &
```

### Step 5: Compare Results
- Compare v2 metrics against v1 at matching epochs
- Key metrics to watch:
  - Pose mAP50 (target: >0.85, up from peak 0.812)
  - Pose mAP50-95 (target: >0.82)
  - Box mAP50 (secondary, likely capped ~0.60)
  - Reduction in epoch-to-epoch volatility
  - Loss composition shift (RLE should drop to ~20%, cls ~0%)

---

## Expected Impact

| Change | Expected Effect | Confidence |
|--------|----------------|------------|
| Mosaic=0.0 | Fewer false positives at composite boundaries; cleaner segment learning | High |
| cls=0.0 | Redirects ~22% gradient budget from useless classification to localization | High |
| rle=0.3 | Reduces RLE from 47% to ~15% of loss; lets pose/box losses get more signal | Medium-High |
| AdamW lr=0.001 | More stable convergence; less metric volatility | Medium |
| one_edge scenes | Covers production edge case; reduces false negatives on single-edge crops | Medium |

**Combined expected improvement**: Pose mAP50 from ~0.73 (current avg) to 0.85+,
with significantly reduced epoch-to-epoch volatility.

---

## Current Run Reference (v1)

For comparison when v2 training completes:

| Metric | Epoch 15 (Best v1) | Epoch 24 (Latest v1) |
|--------|--------------------|-----------------------|
| Pose mAP50 | **0.812** | 0.785 |
| Pose mAP50-95 | **0.798** | 0.780 |
| Box mAP50 | 0.471 | 0.544 |
| Val pose loss | 0.268 | 0.160 |