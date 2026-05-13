# Binary Fiducial Corner Detection
=================================

## Why Binary, Not Multi-Class

Each corner of a photo on a contrasting background has a distinctive L-shaped
boundary pattern, but the 4 orientations (┏ ┓ ┗ ┛) look nearly identical at
the model's input resolution. A single 4-class model **could not classify them**
(cls_loss stuck at ~1.35, near random for 4 classes). The problem isn't detection
— it's classification.

**Binary models eliminate the classification problem entirely.** Each model
answers one yes/no question: "Is this corner type present in this crop?"

| Approach | Models | Classification | Result |
|----------|--------|---------------|--------|
| 4-class fiducial | 1 | UL vs UR vs LL vs LR | **Failed** (cls_loss ≈ random) |
| Binary fiducial | 4 | Corner present? Yes/No | **Works** (mAP50 ≈ 0.92 after 10 epochs) |

## Pipeline: Detect → (Pose →) Fiducial → Crop

```
Input Image
    │
    ▼
┌──────────────┐
│  Detection   │  Find photo bounding boxes (1 forward pass)
│   Model      │  → axis-aligned bounding boxes
└──────┬───────┘
       │
       │ (optional — recommended for multi-photo scenes)
       ▼
┌──────────────┐
│  Pose Model  │  Find approximate corners per photo
│  (single)    │  → 4 keypoints (LL, UL, UR, LR)
└──────┬───────┘
       │
       │ For each photo, extract 4 corner crops:
       │   Crop UL: 640×640 centered near the UL corner position
       │   Crop UR: 640×640 centered near the UR corner position
       │   Crop LL: 640×640 centered near the LL corner position
       │   Crop LR: 640×640 centered near the LR corner position
       │
       │ If no pose model: use bbox corners as approximate positions
       │
       ▼
┌──────────────────────────────────┐
│  4 Binary Fiducial Models        │  4 forward passes (1 per corner type)
│  fiducial-ul: "Is UL present?"  │  → corner bbox (if detected)
│  fiducial-ur: "Is UR present?"  │
│  fiducial-ll: "Is LL present?"  │
│  fiducial-lr: "Is LR present?"  │
└──────┬───────────────────────────┘
       │
       │ (optional: iterate for sub-pixel accuracy)
       ▼
┌────────────────────────┐
│  Iterative Refinement  │  Re-crop around detected corner position
│  (1–2 more iterations) │  → sub-pixel corner localization
└──────┬─────────────────┘
       │
       │ (optional)
       ▼
┌──────────────────┐
│  CV Refinement   │  Sobel edge detection + line intersection
│  (edge search)   │  → sub-pixel corner from edge geometry
└──────┬───────────┘
       │
       ▼
  4 precise corners per photo
       │
       ▼
┌──────────────────┐
│  Perspective     │  Warp or crop using 4 corner positions
│  Warp / Crop     │  → clean extracted photo
└──────────────────┘
```

### When to Skip the Pose Step

| Scenario | Use Pose? | Why |
|----------|-----------|-----|
| Single photo in frame | No | Bbox corners are good enough for initial crops |
| Multiple photos, well-separated | No | Bbox corners still adequate |
| Multiple photos, touching/overlapping | Yes | Pose provides better initial corner estimates than bbox corners |
| Any scene (best quality) | Yes | Better initial positions → tighter fiducial crops → more accurate corners |

The pose model is **always optional**. The fiducial models can find corners
starting from bbox corners alone. But the pose model provides better initial
positions, especially when photos are close together and the bbox corner doesn't
correspond well to the actual photo corner.

### Iterative Fiducial Refinement

The fiducial model is most accurate when the corner is near the center of the
640×640 crop. Iterative refinement exploits this:

1. **Pass 1**: Crop around approximate corner (from pose or bbox) → fiducial detects corner → map back to full image coords
2. **Pass 2**: Re-crop centered on the detected position → fiducial runs again with the corner closer to center → more precise detection
3. **(Optional) Pass 3**: Repeat for sub-pixel accuracy

Each iteration narrows the crop, giving the model a better view of the corner.
In practice, 2 iterations are sufficient for pixel-accurate corners.

### CV Refinement (Optional Final Step)

After fiducial refinement locates corners to ~1–2px accuracy, classical CV
can push to sub-pixel:

1. For each corner, search along two rays (toward the photo edges)
2. Find the strongest gradient along each ray (Sobel edge detection)
3. Fit lines to the edge pixels
4. Compute the intersection of the two lines → sub-pixel corner position

This costs almost nothing (pure CV, no ML inference) and can correct
systematic offsets in the neural network predictions.

## Binary Model Details

| Detail | Value |
|--------|-------|
| Architecture | YOLO26n (nano) |
| Classes | 1 per model (binary: "this corner type" vs background) |
| Input | 640×640 crop centered near the corner |
| Output | Bounding box(es) of corner region + confidence |
| Corner position | bbox center ≈ precise corner location |
| No flip augmentation | Flipping changes corner orientation |
| No mosaic augmentation | Crops are already tightly framed |

### Corner Models

| Model | Question | Looks For |
|-------|----------|-----------|
| `fiducial-ul.onnx` | "Is an UL corner (┏) present?" | Photo extends RIGHT and DOWN from corner |
| `fiducial-ur.onnx` | "Is a UR corner (┓) present?" | Photo extends LEFT and DOWN from corner |
| `fiducial-ll.onnx` | "Is an LL corner (┗) present?" | Photo extends RIGHT and UP from corner |
| `fiducial-lr.onnx` | "Is a LR corner (┛) present?" | Photo extends LEFT and UP from corner |

### Training Data

Each model trains on a balanced binary dataset:

| Component | Ratio | Description |
|-----------|-------|-------------|
| Positive | ~57% | Crops containing the target corner type with the L-shaped boundary visible |
| Hard negative | ~43% | Crops containing a DIFFERENT corner type (the model must reject them) |
| Background | 0% | No pure-background crops available; hard negatives serve this role |

Hard negatives are the most valuable: they look almost identical to positives
(the same L-shape boundary) but in a different orientation. The model must
learn the subtle orientation cue to distinguish its corner type from the others.

### Dataset Generation

```bash
# Step 1: Generate source scenes (reuse existing fiducial data generator)
cd data_generator
python3 generate_fiducial.py --mode batch --train-count 4000 --val-count 1000

# Step 2: Split into 4 binary datasets (one per corner type)
cd ../training
python3 split_binary_datasets.py
```

This creates `data_fiducial_binary/{ul,ur,ll,lr}/` with balanced train/val splits.

### Training

```bash
# Train all 4 models sequentially
python3 train_fiducial_binary.py

# Train a specific corner type
python3 train_fiducial_binary.py --corner ul

# Force retrain
python3 train_fiducial_binary.py --force
```

**Critical hyperparameters** (different from YOLO defaults):

| Parameter | Value | Why |
|-----------|-------|-----|
| `lr0` | **0.001** | Default 0.01 causes violent oscillation (mAP50 bouncing 0.26→0.86) |
| `optimizer` | **auto** (AdamW) | SGD with lr=0.01 oscillates; AdamW with lower LR converges smoothly |
| `mosaic` | **0.0** | Crops are tightly framed; mosaic creates artificial composites that don't match inference |
| `hsv_s` | **0.3** | Consistent with detection/pose models |
| `batch` | **16** | Smaller batches for more gradient updates per epoch |
| `fliplr` | **0.0** | Flipping changes corner orientation — would create mislabeled data |
| `flipud` | **0.0** | Same reason |

### ONNX Export

After training, export each model:

```bash
# From Python
from ultralytics import YOLO
for corner in ['ul', 'ur', 'll', 'lr']:
    model = YOLO(f'runs/binary_fiducial/fiducial-{corner}/weights/best.pt')
    model.export(format='onnx')
```

Output models: `runs/binary_fiducial/fiducial-{corner}/weights/best.onnx`

## Integration with photocrop

The binary fiducial models integrate into the existing `photocrop.py` pipeline
through the `--fiducial-model` flag:

```bash
# Full pipeline: detect → pose → fiducial refine → warp
photocrop --image scan.jpg --preset best --fiducial-model fiducial_corner.onnx

# Without pose model: detect → fiducial (bbox corners as initial positions)
photocrop --image scan.jpg --crop warp --fiducial-model fiducial_corner.onnx
```

**Note**: The current `photocrop.py` uses the 4-class fiducial model format
(class_id 0–3 = UL/UR/LL/LR). The binary fiducial models each output class 0
("corner detected") and need to be run as 4 separate inferences. The inference
code will need updating to support the binary model approach (see Inference
Integration below).

## Inference Integration (Binary Models)

The current `refine_corners_fiducial()` in `photocrop.py` runs a single
4-class model and uses `class_id` to determine corner type. The binary models
require a different approach:

```python
# Pseudocode for binary fiducial inference
def refine_corners_binary_fiducial(image, result, fiducial_sessions, iterations=2):
    """
    fiducial_sessions: dict mapping corner name → ONNX session
                       {'ul': session_ul, 'ur': session_ur, ...}
    """
    kps = {kp["name"]: kp for kp in result.get("keypoints", [])}
    box = result["detection"]["box"]

    bbox_corners = {
        "UL": (box["x1"], box["y1"]),
        "UR": (box["x2"], box["y1"]),
        "LL": (box["x1"], box["y2"]),
        "LR": (box["x2"], box["y2"]),
    }

    # Use pose keypoints as starting positions, fall back to bbox
    approx_corners = {}
    for name in ["UL", "UR", "LL", "LR"]:
        if name in kps and kps[name]["visibility"] >= 0.1:
            approx_corners[name] = (kps[name]["x"], kps[name]["y"])
        else:
            approx_corners[name] = bbox_corners[name]

    refined_corners = {}
    for corner_name, (ax, ay) in approx_corners.items():
        session = fiducial_sessions[corner_name.lower()]
        best_detection = None

        for iteration in range(iterations):
            crop, offset_x, offset_y = extract_fiducial_crop(image, ax, ay)
            detections = run_fiducial(session, crop, conf_threshold=0.5)

            if not detections:
                break

            # Binary model only outputs class 0 — take highest confidence
            best = max(detections, key=lambda d: d["confidence"])

            # Map back to full image coordinates
            ax = best["center_x"] + offset_x
            ay = best["center_y"] + offset_y
            best_detection = best

        if best_detection is not None:
            refined_corners[corner_name] = (ax, ay)
        else:
            refined_corners[corner_name] = approx_corners[corner_name]

    # Update keypoints in result
    for name in ["UL", "UR", "LL", "LR"]:
        if name in refined_corners:
            rx, ry = refined_corners[name]
            if name in kps:
                kps[name]["x"] = rx
                kps[name]["y"] = ry
                kps[name]["visibility"] = 1.0
            else:
                result.setdefault("keypoints", []).append({
                    "name": name, "x": rx, "y": ry, "visibility": 1.0,
                })

    return result
```

### Kotlin Integration (petrie-file-importer)

The binary fiducial models need to be loaded and run separately in Kotlin:

```kotlin
data class FiducialSessions(
    val ul: OrtSession,
    val ur: OrtSession,
    val ll: OrtSession,
    val lr: OrtSession,
)

fun refineCornersBinaryFiducial(
    image: Bitmap,
    result: DetectedPhoto,
    sessions: FiducialSessions,
    iterations: Int = 2
): DetectedPhoto {
    val approxCorners = mapOf(
        "UL" to result.getApproxCorner("UL"),
        "UR" to result.getApproxCorner("UR"),
        "LL" to result.getApproxCorner("LL"),
        "LR" to result.getApproxCorner("LR"),
    )

    val refined = mutableMapOf<String, PointF>()
    for ((name, approx) in approxCorners) {
        val session = when (name) {
            "UL" -> sessions.ul
            "UR" -> sessions.ur
            "LL" -> sessions.ll
            "LR" -> sessions.lr
            else -> continue
        }
        var (ax, ay) = approx
        var bestDet: Detection? = null

        for (i in 0 until iterations) {
            val crop = extractFiducialCrop(image, ax, ay)
            val dets = runFiducial(session, crop)
            if (dets.isEmpty()) break
            val best = dets.maxBy { it.confidence }
            ax = best.centerX + crop.offsetX
            ay = best.centerY + crop.offsetY
            bestDet = best
        }

        refined[name] = if (bestDet != null) PointF(ax, ay) else approx
    }
    return result.copy(corners = refined)
}
```

## Current Training Status

| Model | Status | Epochs | Best mAP50 | Notes |
|-------|--------|--------|------------|-------|
| fiducial-ul | Training | 11/50 | 0.917 | lr=0.001, mosaic=0, AdamW |
| fiducial-ur | Queued | 0/50 | — | Awaits UL completion |
| fiducial-ll | Queued | 0/50 | — | Awaits UR completion |
| fiducial-lr | Queued | 0/50 | — | Awaits LL completion |

## File Structure

```
data_fiducial_binary/
├── ul/
│   ├── dataset_ul.yaml          # nc: 1, names: ['corner']
│   ├── images/
│   │   ├── train/                # ~4500 images (57% UL positive, 43% hard neg)
│   │   └── val/                  # ~1125 images
│   └── labels/
│       ├── train/                # YOLO format, class 0 or empty
│       └── val/
├── ur/ (same structure)
├── ll/ (same structure)
└── lr/ (same structure)

training/
├── train_fiducial_binary.py     # Train all 4 binary models
├── split_binary_datasets.py      # Split 4-class data into 4 binary datasets
├── dataset_fiducial_binary.yaml # (per-corner YAMLs in data_fiducial_binary/)
└── runs/binary_fiducial/
    ├── fiducial-ul/              # UL model training output
    ├── fiducial-ur/              # (pending)
    ├── fiducial-ll/              # (pending)
    └── fiducial-lr/              # (pending)
```