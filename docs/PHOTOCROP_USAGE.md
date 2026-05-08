# photocrop — Usage Guide

Detect and extract individual photographs from multi-photo scans. Feed it a scanned image of one or more physical photos on a flatbed scanner, and it finds each photo, locates its four corners, and outputs clean perspective-corrected crops.

## Quick Start

```bash
# Just detect and output coordinates (~5s)
photocrop --image scan.jpg --preset quick

# Crop photos with auto-refine (~7s, recommended starting point)
photocrop --image scan.jpg --preset crop

# Perspective warp with white fill (~5s, clean edges)
photocrop --image scan.jpg --preset warp

# Best quality for tight layouts (~15s)
photocrop --image scan.jpg --preset best

# Show annotated detection image (not saved by default)
photocrop --image scan.jpg --preset warp --debug
```

> If `photocrop` isn't on your PATH yet, you can also run it as a module:
> ```bash
> python -m onnx_inference --image scan.jpg --preset warp
> ```

---

## How It Works

The pipeline processes a scanned image (one or more photos on a scanner bed) through several stages:

```
Input Image
    │
    ▼
┌─────────────────────┐
│  1. Detection Model  │  Find where photos are (bounding boxes)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  2. Pose Model       │  Find 4 corners per photo (keypoints)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  3. Dedup            │  Remove duplicate detections
└─────────────────────┘
    │
    ▼ (optional stages)
┌─────────────────────┐
│  4. CV Refinement    │  Edge detection + line intersection
│     (auto or manual) │  Fixes inaccurate NN-predicted corners
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  5. Crop / Warp      │  Extract each photo individually
└─────────────────────┘
```

### Keypoint Order

The pose model detects 4 corners per photo in a fixed order:

| Index | Name | Position |
|-------|------|----------|
| kp0 | LL | Lower-Left |
| kp1 | UL | Upper-Left |
| kp2 | UR | Upper-Right |
| kp3 | LR | Lower-Right |

### When Things Go Wrong

The most common issues and their solutions:

| Problem | Cause | Fix |
|---------|-------|-----|
| **Photos too close together** | Default 15% crop expand includes adjacent photo | Use `--pose-sweep-xy` or `--preset best` |
| **Tiny "strip" crops** | Fewer than 3 visible corners → crop collapses | Use `--auto-refine` (included in crop/warp/best presets) |
| **Inaccurate corners** | NN prediction is approximate | Use `--cv-refine` (included in best preset) |
| **Detects wrong photo** | Detection box too large, captures neighbor | Use `--pose-sweep-xy` |

---

## Presets

Presets bundle common settings. Any individual flag can override a preset value.

| Preset | What it does | Time | Use when |
|--------|-------------|------|----------|
| **quick** | Detect + pose only, no cropping | ~5s | You only need coordinates, no crops |
| **crop** | + auto-refine + simple-corners crop | ~7s | You want rectangular crops with margin |
| **warp** | + auto-refine + perspective warp, white fill | ~5s | You want perspective-corrected crops |
| **best** | + sweep + cv-refine + auto-refine + warp | ~15s | Photos are close together; maximum quality |

```bash
# Crop photos out with 20px margin, auto-fixing bad corners
photocrop --image scan.jpg --preset crop

# Perspective warp with clean white borders
photocrop --image scan.jpg --preset warp

# Tight layout — best quality with adaptive crop sizing
photocrop --image scan.jpg --preset best

# Combine: best preset but override the margin
photocrop --image scan.jpg --preset best --crop-margin 30
```

---

## All Options

### Input / Output

| Flag | Default | Description |
|------|---------|-------------|
| `--image`, `-i` | *(required)* | Path to a single image or directory of images |
| `--output`, `-o` | auto | Output path for annotated image(s, only written with `--debug`). Single image defaults to `{stem}_detected.jpg` next to input. Directory defaults to an `output/` subdirectory |
| `--detection-model`, `-d` | `../models/detection_model.onnx` | Path to detection ONNX model |
| `--pose-model`, `-p` | `../models/pose_model_v2.onnx` | Path to pose ONNX model |
| `--limit`, `-n` | 0 | Process at most N images from a directory (0 = all) |

### Presets

| Flag | Description |
|------|-------------|
| `--preset {quick,crop,warp,best}` | Apply a named preset. Individual flags override preset values |

### Detection

| Flag | Default | Description |
|------|---------|-------------|
| `--det-conf` | 0.5 | Detection confidence threshold |
| `--iou` | 0.45 | NMS IoU threshold for detection |
| `--imgsz` | 640 | Model input image size |

### Pose

| Flag | Default | Description |
|------|---------|-------------|
| `--pose-conf` | 0.5 | Pose confidence threshold |
| `--pose-crop-expand` | 0.15 | How much to expand the detection box before passing to pose model (fraction of larger dimension). Larger = more context, but may include adjacent photos |

### Pose Refinement

| Flag | Default | Description |
|------|---------|-------------|
| `--pose-refine` | off | Run a second pose pass with a tighter crop derived from the first pass keypoints. Helps when the initial detection box is loose |
| `--pose-refine-expand` | 0.05 | Expansion for the refine crop (fraction). Only used with `--pose-refine` |

### Sweep (Adaptive Crop Sizing)

When photos are close together, the default 15% expand can cause the pose model to latch onto an adjacent photo. Sweep tries multiple expand values and picks the one with the best corner visibility.

| Flag | Default | Description |
|------|---------|-------------|
| `--pose-sweep` | off | Try a grid of (crop-expand, refine-expand) values and pick the best per photo |
| `--sweep-crop-expands` | `0.05,0.10,0.15,0.20` | Crop-expand values to try in sweep |
| `--pose-sweep-xy` | off | Try per-axis (X/E-W, Y/N-S) expand values independently. **Recommended for tight layouts** |
| `--sweep-xy-expands` | `0.05,0.10,0.15,0.20,0.25` | Per-axis expand values to try |
| `--center-bias` | off | Bias crop expansion toward image center (less expansion toward edges where there's only background) |

### CV Corner Refinement

After the neural network finds approximate corners, CV refinement uses edge detection and line intersection to find the exact corner positions.

| Flag | Default | Description |
|------|---------|-------------|
| `--cv-refine` | off | Apply CV refinement to all photos. Uses orientation-aware Sobel edge filtering + neighbor-anchored projection + strip search |
| `--cv-refine-radius` | 40 | Search radius (pixels) around each NN-predicted corner |
| `--auto-refine` | off | Apply CV refinement only to photos with fewer than 3 visible corners. Prevents "strip" crops without the cost of refining every photo |

### Cropping

| Flag | Default | Description |
|------|---------|-------------|
| `--crop` | *(none)* | Crop mode: `simple` (detection bbox), `simple-corners` (keypoint bbox + margin), `warp` (perspective, inward), `warp-stretch` (perspective, outward, preserves all content) |
| `--crop-dir` | `crops/` | Output directory for cropped photos |
| `--crop-margin` | 0 | Extra pixels to expand each side of the crop. For warp modes, pushes corners outward from quad center |
| `--crop-transparent` | off | Save crops as transparent PNG. Area outside keypoint quad is transparent |
| `--border-fill` | grey | Fill color for warp areas outside source image. Accepts `R,G,B`, `#RRGGBB`, `#RGB`, or named colors: `white`, `black`, `grey`/`gray`, `red`, `green`, `blue` |


### Output Control

| Flag | Default | Description |
|------|---------|-------------|
| `--coords {json,text}` | *(none)* | Output corner coordinates to **stdout**. `json` prints a nested array `[[[x,y], ...], ...]` with corners in LL→UL→UR→LR order. `text` prints one corner per line: `PHOTO CORNER X Y VIS`. All other diagnostic output goes to **stderr** |
| `--debug` | off | Save annotated detection image (boxes, keypoints, corners drawn on the original). Not saved by default |
| `--no-image` | off | Don't save cropped photo files. Useful with `--coords` when you only need coordinates without extracting image files |

### Deduplication

| Flag | Default | Description |
|------|---------|-------------|
| `--dedup-dist` | 0.08 | Minimum center distance for dedup, as fraction of image minimum dimension |

---

## Common Workflows

### Basic photo extraction

```bash
# Simplest: detect and annotate (no crops)
photocrop --image scan.jpg

# Recommended: perspective-corrected crops with auto-fix
photocrop --image scan.jpg --preset warp
```

### Tight layouts (photos close together)

When photos are close together on the scanner, the pose model can incorrectly detect an adjacent photo's edges. The sweep option tries multiple crop sizes and picks the best:

```bash
# Best quality for tight layouts (~15s)
photocrop --image scan.jpg --preset best

# Add sweep to any preset
photocrop --image scan.jpg --preset warp --pose-sweep-xy
```

### Batch processing

```bash
# Process a folder of scans
photocrop --image ./scans/ --preset warp --output ./extracted/

# Process just the first 5 images
photocrop --image ./scans/ --preset warp --limit 5
```

### Coordinates only (scripting)

Get corner coordinates without writing any image files — useful for downstream processing, automation, or piping to other tools.

```bash
# JSON coordinates to stdout (one array per photo, corners in LL→UL→UR→LR order)
photocrop --image scan.jpg --coords json --no-image

# Text format — one corner per line: PHOTO CORNER X Y VIS
photocrop --image scan.jpg --coords text --no-image

# Extract just the coordinates, silence all progress output
photocrop --image scan.jpg --coords json --no-image 2>/dev/null

# Coordinates only — no file output at all
photocrop --image scan.jpg --preset crop --coords json --no-image

# Use in a pipeline
photocrop --image scan.jpg --coords json --no-image 2>/dev/null | python3 -c "
import sys, json
data = json.loads(sys.stdin.read())
for i, corners in enumerate(data):
    print(f'Photo {i+1}: {len(corners)} corners')
"
```

### Custom crop settings

```bash
# Simple corner crop with generous margin (good for manual trimming)
photocrop --image scan.jpg \
  --crop simple-corners --crop-margin 30

# Perspective warp with white border + larger margin
photocrop --image scan.jpg \
  --crop warp-stretch --crop-margin 20 --border-fill white

# Transparent PNG for compositing in Photoshop etc
photocrop --image scan.jpg \
  --crop simple-corners --crop-margin 10 --crop-transparent
```

### Manual refinement (advanced)

```bash
# Full manual control: sweep + cv-refine + warp
photocrop --image scan.jpg \
  --pose-sweep-xy --cv-refine \
  --crop warp-stretch --crop-margin 20 --border-fill white

# Just CV refinement, no sweep (faster, good for well-separated photos)
photocrop --image scan.jpg \
  --cv-refine --crop warp-stretch --crop-margin 20 --border-fill white

# Pose refinement only (re-runs pose model with tighter crop)
photocrop --image scan.jpg \
  --pose-refine --crop simple-corners --crop-margin 20
```

---

## Crop Modes Compared

| Mode | How it works | Best for | Quality |
|------|-------------|----------|---------|
| `simple` | Axis-aligned bbox from detection model | Quick preview | Low — includes background |
| `simple-corners` | Axis-aligned bbox from detected corners + margin | Rectangular crops | Medium — tight but axis-aligned |
| `warp` | Perspective transform (inward, average edge lengths) | Nearly-rectangular photos | Good — may clip small edge portions |
| `warp-stretch` | Perspective transform (outward, max edge lengths) | All photos | Best — preserves all content, most resistant to corner errors |

### Why `warp-stretch` is usually best

- `simple` and `simple-corners` produce axis-aligned rectangles, so rotated or skewed photos get unnecessary background included
- `warp` uses the average of opposite edge lengths, which can clip small portions of the photo at corners
- `warp-stretch` uses the maximum of opposite edge lengths, ensuring no content is lost. Combined with `--crop-margin` and `--border-fill white`, this gives the cleanest results

```bash
# The recommended crop command
photocrop --image scan.jpg \
  --crop warp-stretch --crop-margin 20 --border-fill white
```

---

## Understanding Auto-Refine vs CV-Refine

| | `--cv-refine` | `--auto-refine` |
|---|---|---|
| **What** | Run CV edge refinement on every photo | Run CV refinement only on photos with <3 visible corners |
| **When** | You want maximum accuracy on all corners | You want to fix broken crops without slowing down all photos |
| **Cost** | Always processes every photo (~3s overhead) | Only runs when needed (0s on well-detected images) |
| **Included in** | `--preset best` | `--preset crop`, `--preset warp`, `--preset best` |

The `--auto-refine` option is a safety net: if any photo has fewer than 3 visible corners (meaning the crop could collapse to a degenerate shape like a thin "strip"), it automatically runs the full CV refinement pass. On well-separated photos where all 4 corners are detected, it adds zero overhead.

---

## Understanding Sweep

The pose model needs some surrounding context to find corners, so it expands the crop from the detection box by `--pose-crop-expand` (default 15%). When photos are close together, this expansion can include parts of an adjacent photo, causing the pose model to detect the wrong photo's corners.

Sweep solves this by trying multiple expand sizes and picking the one where the pose model is most confident (measured by corner visibility).

| Mode | What it tries | When to use |
|------|---------------|-------------|
| `--pose-sweep` | Grid of (crop-expand × refine-expand) values | General search for best crop sizing |
| `--pose-sweep-xy` | Per-axis (X/E-W, Y/N-S) expand values independently | **Tight layouts where photos are side by side** |

```bash
# Add sweep to any preset (adds ~3x processing time)
photocrop --image scan.jpg --preset warp --pose-sweep-xy
```

---

## Output Format

### Annotated Image (requires `--debug`)

By default, photocrop does **not** save the annotated detection image. Use `--debug` to save a visualization showing:
- Bounding boxes (from detection model)
- Corner keypoints (from pose model) with visibility scores
- Crop regions (if `--crop` is set)

Default naming: `{original_stem}_detected.jpg`

### Cropped Photos

When cropping is enabled (`--crop`), each detected photo is saved separately:

```
crops/
├── scan_001_1.jpg    # First detected photo
├── scan_001_2.jpg    # Second detected photo
├── scan_001_3.jpg    # Third detected photo
└── scan_001_4.jpg    # Fourth detected photo
```

Naming: `{original_stem}_{photo_id}.{ext}`

---

## Performance Guide

Approximate processing times for a 1512×2016 scan with 4 photos:

| Command | Time | Quality |
|---------|------|---------|
| `--preset quick` | ~5s | Annotated output only |
| `--preset crop` | ~5-7s | Good crops with auto-refine safety net |
| `--preset warp` | ~5s | Good perspective warps with auto-refine |
| `--preset best` | ~15-20s | Maximum quality, sweep + cv-refine |

### Speed Tips

- **For well-separated photos:** `--preset warp` is fast and sufficient. Sweep adds little value when photos aren't close together.
- **For tight layouts:** `--preset best` (which includes sweep) is worth the extra time.
- **For batch processing:** Use `--preset warp` for speed, and inspect results. Re-run any failures with `--preset best`.
- **Skip detection:** If you know the image contains exactly one photo, you can skip the detection model and run pose directly (currently requires API usage, not CLI).

---

## Python API

For programmatic use, you can import and call the pipeline directly:

```python
from onnx_inference.photocrop import pipeline, load_onnx_model, save_crops
from onnx_inference.photocrop import format_coords
from PIL import Image

# Load models
det_session = load_onnx_model("models/detection_model.onnx")
pose_session = load_onnx_model("models/pose_model_v2.onnx")

# Process an image
image = Image.open("scan.jpg")
results = pipeline(
    detection_session=det_session,
    pose_session=pose_session,
    image=image,
    cv_refine=True,          # Optional: CV corner refinement
    auto_refine=True,         # Optional: auto-fix broken corners
    pose_sweep_xy=True,       # Optional: adaptive crop sizing
)

# Each result has:
#   result["detection"]    - detection box and confidence
#   result["pose_confidence"] - overall pose confidence
#   result["keypoints"]    - list of 4 corners with name, x, y, visibility
#   result["center"]       - (x, y) center of the photo

# Get corner coordinates as JSON or text
json_coords = format_coords(results, fmt="json")
# json_coords = '[[[x,y],[x,y],...], ...]'  — LL, UL, UR, LR order

text_coords = format_coords(results, fmt="text")
# text_coords = '1 LL x y vis\n1 UL x y vis\n...'

# Save crops
save_crops(
    image=image,
    results=results,
    image_path="scan.jpg",
    crop_mode="warp-stretch",
    crop_margin=20,
    border_fill=(255, 255, 255),  # white
    output_dir="crops/",
)
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|----------|-------|----------|
| "Strip" crop (e.g., 64×888) | Only 2 corners detected, crop collapses | Use `--auto-refine` or `--preset crop/warp/best` |
| Corners on wrong photo | Crop expand includes adjacent photo | Use `--pose-sweep-xy` or `--preset best` |
| Missing corners (low visibility) | Photo edges unclear or cropped | Use `--cv-refine` for edge-based detection |
| Too many duplicate detections | Detection model finds overlapping boxes | Lower `--det-conf` or adjust `--dedup-dist` |
| Crops include too much background | Corners detected inside actual edge | Use `--crop-margin` to add padding, or `--cv-refine` for precise corners |
| Crops clip photo edges | Corners detected outside actual edge | Use `warp-stretch` mode (preserves all content) or reduce `--crop-margin` |