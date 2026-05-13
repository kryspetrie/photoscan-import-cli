# Fiducial Corner Refinement — Current Approach
==============================================

**Status**: Active — binary fiducial models training

This document describes the ORIGINAL 4-class fiducial plan (now replaced by
binary models). See [`FIDUCIAL_BINARY.md`](FIDUCIAL_BINARY.md) for the current
approach.

## Why the 4-Class Model Was Replaced

A single 4-class fiducial model (UL/UR/LL/LR) trained for 50 epochs with
classification loss stuck at ~1.35 — near random for 4 classes (ln(4)≈1.39).
The 4 corner orientations look nearly identical at the model's input resolution;
they are L-shaped boundaries that differ only by 90° rotation.

**4 binary models** (one per corner type) solve this by eliminating the
classification problem. Each model answers one yes/no question: "Is THIS corner
type present in this crop?"

## Original 4-Class Plan (Archived)

### Architecture: Single 4-Class Fiducial Model

| Class | Corner | L-Shape | Photo Extends |
|-------|--------|---------|---------------|
| 0 | UL | ┏ | → right, ↓ down |
| 1 | UR | ┓ | ← left, ↓ down |
| 2 | LL | ┗ | → right, ↑ up |
| 3 | LR | ┛ | ← left, ↑ up |

- **Type**: YOLO detection (4 classes: ul, ur, ll, lr)
- **Architecture**: yolo26n (nano)
- **Input**: 640×640 crop centered near the approximate corner position
- **Output**: Bounding box with class indicating corner orientation
- **NO flip augmentation**: flipping changes corner orientation

### Pipeline (unchanged for binary models)

```
Input Image
    │
    ▼
┌──────────────┐
│  Detection   │  → axis-aligned bounding boxes
└──────┬───────┘
       │ for each bbox, extract 4 corner crops
       │
       ├─ UL crop ──→ fiducial-ul model (or 4-class model, UL class)
       ├─ UR crop ──→ fiducial-ur model (or 4-class model, UR class)
       ├─ LL crop ──→ fiducial-ll model (or 4-class model, LL class)
       └─ LR crop ──→ fiducial-lr model (or 4-class model, LR class)
       │
       ▼
  4 precise corners per photo
       │
       ▼ (optional: iterate 1–2 times)
  Re-crop around detected position → even more precise
       │
       ▼
  Perspective warp → clean extracted photo
```

The pipeline structure is the same whether using 1 four-class model or 4 binary
models. The difference is in how inference works: 4-class uses 1 model × 4 crops,
binary uses 4 models × 1 crop each. Both approaches make 4 forward passes total.