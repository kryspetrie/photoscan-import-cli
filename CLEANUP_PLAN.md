# Photo Pose Detector — Repo Cleanup Plan

## Guiding Principle

**Do not remove anything that is part of the active production pipeline.** The current pipeline is:

```
detect → pose (single-photo corners) → dedup → rescue → crop/warp
```

Both the **detection model** (`detection_ep47`) and the **pose model** (`pose_single_ep42`) are production models actively used in inference. Their data generators, training scripts, dataset configs, and training data must all be preserved — they are the reproducible provenance of the models the entire system depends on.

The fiducial-pose segment model is **additive** (on top of the pose model), not a replacement for it.

---

## Active Pipeline — KEEP Everything

### Inference

| File | Purpose | Status |
|------|---------|--------|
| `onnx_inference/photocrop.py` | Main inference pipeline | ✅ Active |
| `onnx_inference/__init__.py` | Package init | ✅ Active |
| `onnx_inference/__main__.py` | `python -m onnx_inference` entry point | ✅ Active |

### Production Models

| File | Purpose | Status |
|------|---------|--------|
| `models/detection_ep47.onnx` | Detection model (ONNX) | ✅ Active |
| `models/detection_ep47.pt` | Detection model (PyTorch) | ✅ Active |
| `models/pose_single_ep42.onnx` | Pose model (ONNX) | ✅ Active |
| `models/pose_single_ep42.pt` | Pose model (PyTorch) | ✅ Active |

### Detection Pipeline (Data → Train → Export)

| File | Purpose | Status |
|------|---------|--------|
| `data_generator/generate_common.py` | Shared generation utilities | ✅ Active |
| `data_generator/generate_detection.py` | Detection data generator | ✅ Active |
| `data_detection/` | Detection training dataset (4000 train + 1000 val) | ✅ Active |
| `training/train_detection.py` | Detection model training | ✅ Active |
| `training/dataset_detection.yaml` | Detection dataset config | ✅ Active |

### Pose Pipeline (Data → Train → Export)

| File | Purpose | Status |
|------|---------|--------|
| `data_generator/generate_pose.py` | Single-photo pose data generator | ✅ Active |
| `data_pose/` | Pose training dataset (4000 train + 1000 val) | ✅ Active |
| `training/train_pose.py` | Pose model training | ✅ Active |
| `training/dataset_pose.yaml` | Pose dataset config | ✅ Active |

### Fiducial-Pose Pipeline (In Development)

| File | Purpose | Status |
|------|---------|--------|
| `data_generator/generate_fiducial_pose.py` | Fiducial-pose segment generator | ✅ Active (in dev) |
| `data_fiducial_pose/` | Fiducial-pose training dataset | ✅ Active (in dev, currently empty/regenerating) |
| `training/train_fiducial_pose.py` | Fiducial-pose segment training | ✅ Active (in dev) |
| `training/dataset_fiducial_pose.yaml` | Fiducial-pose dataset config | ✅ Active (in dev) |

### Export & Utilities

| File | Purpose | Status |
|------|---------|--------|
| `export/export_onnx.py` | ONNX model export | ✅ Active |
| `download_oxford.py` | Downloads Oxford Buildings Dataset (used by generate_common) | ✅ Active |
| `download_textures.py` | Downloads DTD textures (used by generate_common) | ✅ Active |
| `pyproject.toml` | Package config | ✅ Active |
| `requirements.txt` | Python dependencies | ✅ Active |
| `setup.sh` | Setup script | ✅ Active |
| `.gitignore` | Git ignore rules | ✅ Active |
| `README.md` | Project readme | ⚠️ Needs update |
| `real_world_example.jpg` | Test image | ✅ Active |
| `real_world_example.json` | Expected output for test image | ✅ Active |

### Tests

| File | Purpose | Status |
|------|---------|--------|
| `tests/benchmark_presets.py` | Benchmarks photocrop presets | ✅ Active |
| `tests/test_adaptive_crop.py` | Adaptive crop tests | ✅ Active |
| `tests/test_cross_photo_validation.py` | Cross-photo validation tests | ✅ Active |
| `tests/test_preset_crop_separation.py` | Preset/crop separation tests | ✅ Active |
| `tests/test_refine_corners_cv.py` | CV corner refinement tests | ✅ Active |

### Other Keepers

| File | Purpose | Status |
|------|---------|--------|
| `bench_corner_refine.py` | Corner refinement benchmark utility | ✅ Active |
| `data_generator/images/` | Source photos for synthetic data | ✅ Active |
| `textures/` | Background textures (gitignored) | ✅ Active |
| `training/yolo26s-pose.pt` | YOLOv6s-pose pretrained weights (used by train_pose.py) | ✅ Active |
| `training/yolo26s.pt` | YOLOv6s pretrained weights (used by train_detection/fiducial) | ✅ Active |
| `training/yolo26n-pose.pt` | YOLOv6n-pose pretrained weights | ✅ Active |
| `training/yolo26n.pt` | YOLOv6n pretrained weights | ✅ Active |
| `training/yolo11n.pt` | YOLOv11n pretrained weights | ✅ Active |

---

## Obsolete / Deletable Files

### ❌ Dead Generator (Explicitly Deprecated)

| File | Why obsolete | Action |
|------|-------------|--------|
| `data_generator/generate.py` | Original combined detection+pose generator. Explicitly marked **DEPRECATED** in its own docstring. Replaced by separate `generate_detection.py` and `generate_pose.py`. | **Delete** |

### ❌ Dead Training Scripts (Non-functional or Superseded)

| File | Why obsolete | Action |
|------|-------------|--------|
| `training/train.py` | Old single-photo pose trainer. Superseded by `train_pose.py` (better config). References `generate_dataset.py` which doesn't exist. Hardcoded paths. | **Delete** |
| `training/train_pipeline.py` | Orchestration script that trains detection → pose → **binary fiducial**. Binary fiducial models failed and are abandoned. Imports `train_fiducial_binary` which doesn't exist. Won't run. | **Delete** |
| `training/validate.py` | Generic validation script using dead code patterns. The `val=True` flag in ultralytics already handles validation during training. | **Delete** |
| `training/split_binary_datasets.py` | Splits data for the 4 binary fiducial classifiers. Binary fiducial approach was abandoned. References `data_fiducial/` which no longer exists. | **Delete** |

### ❌ Stale Training Logs

| File | Why obsolete | Action |
|------|-------------|--------|
| `training/detect_train.log` | Old detection training log. Training already completed, model exported. | **Delete** |
| `training/pose_train.log` | Old pose training log. Training already completed, model exported. | **Delete** |
| `training/resume_training.log` | Old resume training log. | **Delete** |

### ❌ Stale Models (Superseded)

| File | Why obsolete | Action |
|------|-------------|--------|
| `models/pose_single_ep27.onnx` | Superseded by ep42 (better mAP). ep42 is the production model. | **Delete** |
| `models/pose_single_ep27.pt` | Superseded by ep42. | **Delete** |

### ❌ Stale Documentation

| File | Why obsolete | Action |
|------|-------------|--------|
| `docs/FIDUCIAL_PLAN.md` | Describes the abandoned 4-class fiducial approach. Header says "Status: Active — binary fiducial models training" but that's also abandoned. | **Delete** |
| `docs/FIDUCIAL_BINARY.md` | Describes the abandoned binary fiducial approach. Both 4-class and binary approaches failed — the project has moved to **fiducial-pose segments**. | **Delete** |

### ❌ Stale Data Directories (No longer exist on disk)

| Path | Status | Action |
|------|--------|--------|
| `data/` | Does not exist on disk | Ensure gitignored |
| `data_pose_multi/` | Does not exist on disk | Ensure gitignored |
| `data_fiducial/` | Does not exist on disk | Ensure gitignored |
| `data_fiducial_binary/` | Does not exist on disk | Ensure gitignored |
| `REMOVED/` | Does not exist on disk | Ensure gitignored |

### ❌ Stale/Ephemeral Directories on Disk

| Path | Why obsolete | Action |
|------|-------------|--------|
| `data_generator/data/` | Empty directory (only `.DS_Store`). No generator writes here. | **Delete** |
| `training/runs/pose/runs/` | Nested runs directory (double-nesting bug). The active fiducial-pose training is at `training/runs/pose/runs/pose/fiducial-pose-segments/` — should be flattened. | **Review** — may contain active training output |
| `training/__pycache__/` | Python bytecode cache | **Delete** |
| `__pycache__/` | Python bytecode cache (root) | **Delete** |

### ❌ Mac Junk

| Path | Action |
|------|--------|
| Various `.DS_Store` files | **Delete and add to .gitignore** |

---

## .gitignore Updates

Add these entries:

```gitignore
# Training data (regenerateable but large)
data_detection/
data_pose/
data_fiducial_pose/
data_pose_multi/
data_fiducial/
data_fiducial_binary/
data/

# Training outputs (regenerateable)
training/runs/

# ONNX cache files
*.npy

# Mac junk
.DS_Store

# Python cache
__pycache__/
```

Remove stale entries that are already covered or no longer relevant:
```gitignore
# These can be removed:
test/
REMOVED/
```

---

## Proposed New Directory Structure

```
photo-pose-detector/
├── README.md                          # Updated with current pipeline
├── pyproject.toml                      # Package config
├── requirements.txt                    # Dependencies
├── setup.sh                            # Setup script
├── .gitignore                          # Updated gitignore
│
├── data_generator/
│   ├── generate_common.py              # Shared generation utilities
│   ├── generate_detection.py           # Detection model data generator
│   ├── generate_pose.py                # ★ Single-photo pose data generator
│   ├── generate_fiducial_pose.py       # Fiducial pose segment generator
│   └── images/                         # Source photos (gitignored)
│
├── data_detection/                     # Detection training data (gitignored, regenerateable)
├── data_pose/                          # ★ Pose training data (gitignored, regenerateable)
├── data_fiducial_pose/                 # Fiducial-pose training data (gitignored, in dev)
│
├── training/
│   ├── train_detection.py              # Detection model trainer
│   ├── train_pose.py                   # ★ Single-photo pose trainer
│   ├── train_fiducial_pose.py          # Fiducial pose segment trainer
│   ├── dataset_detection.yaml          # Detection dataset config
│   ├── dataset_pose.yaml              # ★ Pose dataset config
│   ├── dataset_fiducial_pose.yaml      # Fiducial-pose dataset config
│   ├── yolo26s-pose.pt                 # Pretrained weights (pose)
│   ├── yolo26s.pt                      # Pretrained weights (detection/fiducial)
│   ├── yolo26n-pose.pt                 # Pretrained weights (nano pose)
│   ├── yolo26n.pt                      # Pretrained weights (nano detection)
│   └── yolo11n.pt                      # Pretrained weights (v11)
│
├── onnx_inference/
│   ├── __init__.py
│   ├── __main__.py
│   └── photocrop.py                    # Main inference pipeline
│
├── export/
│   └── export_onnx.py                  # ONNX model export
│
├── models/
│   ├── detection_ep47.onnx             # Active detection model
│   ├── detection_ep47.pt               # Active detection model (PyTorch)
│   ├── pose_single_ep42.onnx           # ★ Active pose model
│   └── pose_single_ep42.pt             # ★ Active pose model (PyTorch)
│
├── docs/
│   ├── ARCHITECTURE.md                 # New: current pipeline architecture
│   ├── FUTURE_IMPROVEMENTS.md          # Updated: remove abandoned approaches
│   └── PHOTOCROP_USAGE.md              # Updated: add fiducial pose pipeline
│
├── tools/
│   └── corner_annotator.html           # (if still needed)
│
├── tests/
│   ├── benchmark_presets.py
│   ├── test_adaptive_crop.py
│   ├── test_cross_photo_validation.py
│   ├── test_preset_crop_separation.py
│   └── test_refine_corners_cv.py
│
├── bench_corner_refine.py              # Benchmark utility
├── download_oxford.py                  # Downloads Oxford dataset
├── download_textures.py                # Downloads DTD textures
├── real_world_example.jpg              # Test image
└── real_world_example.json             # Expected test output
```

★ = files that were nearly deleted in the old plan but are **critical production pipeline components**

---

## Execution Order

1. **Delete truly dead Python scripts** (safest, no side effects)
   - `data_generator/generate.py` — deprecated, replaced by separate generators
   - `training/train.py` — old, references nonexistent files
   - `training/train_pipeline.py` — imports nonexistent modules
   - `training/validate.py` — superseded by ultralytics built-in validation
   - `training/split_binary_datasets.py` — for abandoned binary fiducial approach

2. **Delete stale training logs**
   - `training/detect_train.log`
   - `training/pose_train.log`
   - `training/resume_training.log`

3. **Delete superseded models**
   - `models/pose_single_ep27.onnx`
   - `models/pose_single_ep27.pt`

4. **Delete stale docs**
   - `docs/FIDUCIAL_PLAN.md`
   - `docs/FIDUCIAL_BINARY.md`

5. **Clean up ephemeral files**
   - `data_generator/data/` (empty dir, only `.DS_Store`)
   - `training/__pycache__/`
   - `__pycache__/` (root)
   - All `.DS_Store` files

6. **Update .gitignore** — add `data_fiducial_pose/`, `training/runs/`, `*.npy`, `.DS_Store`, `__pycache__/`; remove stale `test/`, `REMOVED/`

7. **Review `training/runs/pose/runs/` nesting** — flatten if fiducial-pose training is not actively running, otherwise wait until training completes

8. **Update documentation** (do not delete, just update)
   - Update `docs/FUTURE_IMPROVEMENTS.md` — remove binary fiducial section, add fiducial-pose segment section
   - Update `docs/PHOTOCROP_USAGE.md` — already mostly current, add fiducial pose when integrated
   - Create `docs/ARCHITECTURE.md` — document current pipeline
   - Update `README.md`

9. **After fiducial-pose training completes**
   - Export model to ONNX: `python export/export_onnx.py`
   - Add to `models/` directory
   - Integrate fiducial pose inference into `photocrop.py`

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|-------------|
| Delete `generate.py` | None — explicitly deprecated, replaced by separate generators | Already replaced |
| Delete `train.py` | Very low — superseded by `train_pose.py` | `train_pose.py` is better |
| Delete `train_pipeline.py` | None — imports nonexistent modules | Won't run anyway |
| Delete `validate.py` | Very low — ultralytics has built-in validation | `val=True` in train scripts |
| Delete `split_binary_datasets.py` | None — binary approach abandoned | N/A |
| Delete `pose_single_ep27.*` | None — ep42 is strictly better | ep42 has higher mAP50-95 |
| Delete `FIDUCIAL_PLAN.md`, `FIDUCIAL_BINARY.md` | Low — approaches abandoned | History in git |
| **KEEP `data_pose/`** | Would be **high risk** to delete | Gitignored, not recoverable from git |
| **KEEP `generate_pose.py`** | Would be **high risk** to delete | Needed to regenerate training data |
| **KEEP `train_pose.py`** | Would be **high risk** to delete | Only way to retrain pose model |
| **KEEP `dataset_pose.yaml`** | Would be **high risk** to delete | Required by `train_pose.py` |
| **KEEP `pose_single_ep42.*`** | Would be **critical** to delete | Production model used in every inference |

All proposed deletions are recoverable from git history. The key principle: **preserve everything needed for the production detection+pose pipeline, and the in-development fiducial-pose pipeline. Only remove truly dead code from abandoned approaches.**