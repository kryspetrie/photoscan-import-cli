# Fiducial Pose Training Strategy

## V3 Post-Mortem

**Peak**: Epoch 18 (mAP50=0.851, mAP50-95=0.843)  
**Decline**: Epoch 18→46, mAP50-95 dropped to ~0.77  
**Root cause**: Training kept minimizing RLE loss (9.17→0.5) but mAP declined because the model overfit on box/cls while keypoint predictions became unstable.

## Why Training Plateaus & Declines

1. **Learning rate too high for late-stage fine-tuning** — AdamW at lr0=0.002 converges fast but also overshoots the keypoint minima
2. **RLE loss continues to decrease but mAP doesn't correlate** — model memorizes keypoint patterns instead of generalizing
3. **Same data distribution throughout** — no curriculum or difficulty scheduling
4. **No early stopping from best mAP checkpoint** — patience=30 on box mAP, not keypoint mAP

## V4 Strategy: Keep Learning Instead of Declining

### 1. Cosine Annealing Learning Rate Schedule
```python
lr0=0.002, lrf=0.01  # cosine from 0.002 → 0.00002 over training
```
This is what we already have, but the issue is the initial rate is too aggressive. The fix is to use a **lower initial rate** with warmup, or switch to **OneCycleLR** which has a built-in super-convergence phase.

### 2. Early Stopping on Keypoint mAP (not box mAP)
The current patience=30 watches box mAP, which plateaus early. We should watch `mAP50-95(P)` (pose/keypoint mAP) which is what we actually care about.

### 3. Reduce Epochs, Train Less
V3 peaked at epoch 18. We should train **30-40 epochs max** with patience=15 on keypoint mAP. More epochs after the peak just causes overfitting.

### 4. Staged/Progressive Training
**Stage 1**: Train with higher lr0=0.002 for 15-20 epochs (fast convergence)  
**Stage 2**: Fine-tune from best checkpoint with lr0=0.0002 for 10-15 epochs (precision refinement)

### 5. Data Quality Over Quantity
The synthetic data has limited variation. We should:
- Increase background diversity (different textures/patterns)
- Add more edge cases (very thin segments, extreme angles, partial occlusion)
- Reduce exact duplicates in the dataset

### 6. Loss Weight Schedule
Start with higher box weight for detection convergence, then **reduce it** and increase pose/RLE weight for keypoint precision:
```python
# Phase 1: Focus on detection
box=7.5, cls=0.5, rle=1.0  # epochs 1-10

# Phase 2: Focus on keypoints  
box=2.0, cls=0.2, rle=2.0  # epochs 11-30
```

### 7. Recommended V4 Configuration
```python
# Two-phase training
epochs=40
patience=15  # early stop on keypoint mAP
lr0=0.001     # half of V3's rate
lrf=0.01      # cosine down to 0.00001
warmup_epochs=5  # longer warmup
optimizer='AdamW'

# Loss weights (moderate, no single loss dominates)
box=3.0        # reduced from V3's 4.0
cls=0.3        # same as V3
dfl=1.5        # same
pose=12.0      # same
kobj=2.0      # UP from 1.0 — emphasize keypoint confidence
rle=0.8       # UP from 0.5 — more keypoint localization emphasis

# Augmentation (slight increase)
scale=0.5      # UP from 0.3
degrees=10.0   # UP from 5.0
translate=0.2  # UP from 0.1
fliplr=0.5     # same
flipud=0.0     # same (semantic)
mosaic=0.0     # same (OFF)
```
