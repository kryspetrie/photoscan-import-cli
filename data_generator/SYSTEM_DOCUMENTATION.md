# Photo Pose Detector — Synthetic Data Generator

## System Documentation

**Version:** 2.0  
**Date:** 2026-04-21  
**File:** `data_generator/generate.py` (1084 lines, 25 functions)

---

## 1. Purpose

The generator produces synthetic training images for **two YOLO models** that work together in a pipeline:

| Model | YOLO Variant | Output | Purpose |
|-------|-------------|--------|---------|
| **Detection** | YOLO (standard) | Axis-aligned bounding boxes | Find where photos are located |
| **Pose** | YOLO-Pose | 4 corner keypoints per photo | Detect precise corner locations for extraction |

Each generated image produces **three outputs**:
1. The composite JPEG image
2. A detection label (5 columns: `class x_center y_center width height`)
3. A pose label (13 columns: `class x_center y_center width height kp0x kp0y kp0v … kp3x kp3y kp3v`)

Both label formats are derived from the **exact same geometry** — no inconsistency is possible.

---

## 2. Pipeline Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│  1. BACKGROUND                                                         │
│     random_base_background() → apply_texture_overlay()                 │
│     Solid color + noise + 3 gradient overlays + optional texture blend  │
├────────────────────────────────────────────────────────────────────────┤
│  2. PLACEMENT                                                          │
│     pack_photos_validated() → _generate_placements()                   │
│     Pick photo count (1–4), generate positions/sizes/rotations,        │
│     validate bounds + overlaps, retry if invalid                       │
├────────────────────────────────────────────────────────────────────────┤
│  3. COMPOSITING (per photo)                                            │
│     Load source → resize → glare → rotate → shadow → composite        │
│     Drop shadow drawn BEFORE photo so it appears behind                 │
├────────────────────────────────────────────────────────────────────────┤
│  4. PERSPECTIVE WARP                                                    │
│     apply_perspective_safe()                                           │
│     5% strength applied to entire composite, corner-safe validation    │
├────────────────────────────────────────────────────────────────────────┤
│  5. LABEL GENERATION                                                   │
│     Transform corners through perspective matrix                       │
│     Emit detection labels (bbox) and pose labels (4 keypoints)         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Configuration Constants

All tuning knobs are at the top of `generate.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `CANVAS_SIZE` | 640 | Image dimensions (matches YOLO input) |
| `PHOTO_SIZE_MIN` | 270 | Absolute minimum photo dimension (1.5× the v1 value of 180) |
| `PHOTO_SIZE_MAX` | 640 | Absolute maximum photo dimension |
| `ROTATION_RANGE` | 30 | Maximum rotation for single-photo layouts (degrees) |
| `NUM_PHOTOS_MIN` | 1 | Minimum photos per image |
| `NUM_PHOTOS_MAX` | 4 | Maximum photos per image |
| `EDGE_MARGIN` | 50 | Canvas edge margin (used in some legacy checks) |
| `OVERLAP_THRESHOLD` | 0.05 | Max overlap fraction (5% of smaller photo's area) |
| `BOUND_MARGIN` | 5 | Minimum margin from canvas edge for rotated corners |
| `MAX_PACK_ATTEMPTS` | 50 | Retry limit (used in reference; `pack_photos_validated` does 10) |

### Layout-specific parameters (hardcoded in `_generate_placements`)

| Parameter | 1 Photo | 2 Photos | 3–4 Photos (Grid) |
|-----------|---------|----------|-------------------|
| Rotation range | ±30° | ±5° | ±5° |
| Size range | 55–85% canvas | Separation-limited | 88–98% of cell |
| Aspect ratio | 0.8–1.2 | 0.9–1.05 | 0.88–1.05 |
| Center positions | Center ± 30px | 0.27/0.73 of canvas | Grid cell center ± 3% |
| `min_side` for shrink | 270 | 270 | 270 |

---

## 4. Background Generation

### 4.1 Base Color — `random_base_background(w, h)`

Produces a solid-color image with subtle noise, then applies gradient overlays.

**Color selection** uses three tiers:

| Probability | Lightness | Saturation | Result |
|-------------|-----------|------------|--------|
| 30% | 0.04–0.28 | 0–0.04 | Dark / near-black |
| 30% | 0.69–0.96 | 0–0.04 | Light / near-white |
| 40% | 0.19–0.86 | 0.04–0.40 | Medium with color |

Hue is fully random (`0–1`). Conversion uses `colorsys.hls_to_rgb`.

**Noise**: Gaussian, sigma 1–4 per channel, added to the float32 image before clamping.

### 4.2 Gradient Overlays — `apply_3_linear_gradients(img)`

Applies **three** gradient overlays using screen blend (`1 - (1-a)(1-b)`):

- **Direction**: chosen randomly from `{horizontal, vertical, diagonal_tl, diagonal_tr}`
- **Opacity**: `random.uniform(0, 0.20)` per gradient
- **Effect**: subtle directional shading that breaks the flat color

Each gradient is a normalized `[0, 1]` ramp along the chosen direction, multiplied by opacity, and screen-blended.

### 4.3 Texture Overlay — `apply_texture_overlay(canvas)`

If `/photo-pose-detector/textures/` exists and contains `.jpg`/`.png` files:

- Randomly selects one texture
- Resizes to canvas dimensions
- Randomly flips (horizontal, vertical, both, or none)
- Blends at `opacity = random.uniform(0, 0.40)` using:
  - 50% chance: **screen blend** → `1 - (1-canvas)(1-texture)`
  - 50% chance: **multiply blend** → `canvas × texture`
- Final: `canvas × (1 - opacity) + blended × opacity`

The project includes 85 texture images in the `textures/` directory, sourced from the [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) and processed to 1200×1200 greyscale with brightness normalization.

---

## 5. Placement Engine

This is the most complex subsystem. It solves the geometric problem of placing 1–4 rotated rectangles on a 640×640 canvas without overlaps and without corners leaving the canvas.

### 5.1 Entry Point — `pack_photos_validated(canvas_size)`

```
for 10 attempts:
    target_count = random.randint(1, 4)
    placements = _generate_placements(target_count, canvas_size)
    if check_bounds(placements) AND check_overlaps(placements):
        return placements

# Fallback: single centered photo, shrink_to_fit
```

The retry loop naturally produces a distribution of photo counts. If all 10 attempts fail (extremely rare), a single large photo is placed at center.

### 5.2 Layout Generation — `_generate_placements(num_photos, canvas_size)`

#### Single Photo (num_photos == 1)

- **Rotation**: ±30° (the widest range — single photos have the most clearance)
- **Center**: `(320 ± 30, 320 ± 30)` — near canvas center with slight jitter
- **Size**: 55–85% of canvas size (352–544px), with random aspect 0.8–1.2
- **Bounds enforcement**: `_shrink_to_fit` iteratively reduces until all 4 rotated corners fit within `[BOUND_MARGIN, CANVAS_SIZE - BOUND_MARGIN]`
- **Result**: photos are typically 355–571px, filling most of the canvas

#### Two Photos (num_photos == 2)

- **Rotation**: ±5° each — tight rotation keeps the expansion factor low (cos5° + sin5° ≈ 1.083)
- **Layout**: 50% horizontal (side-by-side), 50% vertical (stacked)

**Horizontal layout:**
- Photo 1 center: `(0.27 × 640 ± 8, 0.50 × 640 ± 8)` → approximately `(173, 320)`
- Photo 2 center: `(0.73 × 640 ± 8, 0.50 × 640 ± 8)` → approximately `(467, 320)`
- Separation = ~294px between centers (horizontal axis)

**Vertical layout:**
- Photo 1 center: `(0.50 × 640 ± 8, 0.27 × 640 ± 8)`
- Photo 2 center: `(0.50 × 640 ± 8, 0.73 × 640 ± 8)`
- Separation = ~294px between centers (vertical axis)

**Separation-aware sizing** (critical for avoiding overlaps):

The dimension **along the separation axis** is limited by:

```
max_from_sep = separation / expansion_factor
```

where `expansion_factor = |cos(max_rotation)| + |sin(max_rotation)|`.

At 5° max rotation: `expansion = cos(5°) + sin(5°) ≈ 1.083`, so `max_from_sep ≈ 294 / 1.083 ≈ 271px`.

This means each photo's rotated bounding box along the separation axis is at most 271px, fitting within the 294px gap.

The perpendicular dimension uses the same value with a tighter aspect ratio (0.9–1.05).

Both photos then go through `_shrink_to_fit` for final bounds enforcement.

#### Grid Layout (3–4 Photos)

- **Rotation**: ±5°
- **Grid**: always 2×2 (cells are ~305×305px with `BOUND_MARGIN=5`)
- **Positions**: random subset of the 4 grid cells
- **Size**: 88–98% of `min(cell_w, cell_h)` — starting large before shrink
- **Aspect**: 0.88–1.05 — tight aspect ratios prevent wide photos from overflowing cells
- **Center jitter**: ±3% of cell dimensions from cell center
- **Bounds enforcement**: `_shrink_to_fit` with `min_side=270`

Typical grid photo sizes: 270–320px.

### 5.3 Iterative Shrink-to-Fit — `_shrink_to_fit(width, height, cx, cy, rotation, canvas_size, margin, min_side)`

This function is the geometric backbone of the placement engine. It solves the problem: *"given a photo of size (w, h) at position (cx, cy) with rotation θ, what's the largest size that keeps all 4 rotated corners within `[margin, canvas_size - margin]`?"*

**Algorithm:**

```
for up to 30 iterations:
    corners = get_rotated_polygon(w, h, cx, cy, rotation)
    if all corners within [margin, canvas_size - margin]:
        return w, h  # Success — fits!
    
    # Measure overflow
    x_overflow = max(0, margin - corners.x_min, corners.x_max - (canvas_size - margin))
    y_overflow = max(0, margin - corners.y_min, corners.y_max - (canvas_size - margin))
    max_overflow = max(x_overflow, y_overflow)
    
    # Compute shrink factor
    bbox_diag = max(bbox_width, bbox_height, 1)
    shrink = 1.0 - (max_overflow / bbox_diag) * 1.1  # 1.1 overshoot for faster convergence
    shrink = max(shrink, 0.7)  # Never shrink more than 30% per iteration
    
    w = max(int(w * shrink), min_side)
    h = max(int(h * shrink), min_side)
    
    if w and h didn't change:
        return w, h  # Floor reached — accept current size

return w, h  # Converged or exhausted iterations
```

**Key design decisions:**
- **`min_side` floor**: Set to 270 (PHOTO_SIZE_MIN). If a photo can't fit at 270px, the margin will be slightly violated rather than producing a tiny photo. This is the correct tradeoff — a few pixels of margin violation is invisible, but undersized photos produce bad training data.
- **1.1 overshoot factor**: Without this, the shrink would barely dent the overflow each iteration (we'd need far more iterations). The 10% overshoot ensures convergence in ~2–5 iterations.
- **0.7 floor on shrink ratio**: Prevents catastrophic overshooting where a 30° rotation might cause a 50%+ overflow, and without the floor we'd overshoot and produce a tiny photo.

### 5.4 Corner Geometry — `get_rotated_polygon(width, height, center_x, center_y, rotation)`

Returns the 4 corner coordinates of a rotated rectangle in **photo space** (relative to canvas origin).

**Corner order** (critical for label generation):

| Index | Name | Position |
|-------|------|----------|
| 0 | LL | Lower-Left (min X, max Y) |
| 1 | UL | Upper-Left (min X, min Y) |
| 2 | UR | Upper-Right (max X, min Y) |
| 3 | LR | Lower-Right (max X, max Y) |

**Algorithm:**
1. Uses `cv2.getRotationMatrix2D` around the photo's own center `(w/2, h/2)`
2. Computes the expanded bounding box dimensions after rotation: `new_w = h×|sinθ| + w×|cosθ|`, `new_h = w×|sinθ| + h×|cosθ|`
3. Adjusts the rotation matrix translation to center the expanded image
4. Transforms each corner from photo coordinates through the matrix
5. Offsets by the top-left position on the canvas: `top_left_x = center_x - new_w/2`

**Optimization**: If `|rotation| < 1°`, the pure-rectangle coordinates are returned directly (no rotation math needed).

### 5.5 Bounds Checking — `check_bounds(placements, canvas_size, margin)`

For each placement, computes all 4 rotated corners via `get_rotated_polygon` and verifies every corner coordinate lies within `[margin, canvas_size - margin]`. Returns `True` only if ALL corners of ALL placements are within bounds.

### 5.6 Overlap Checking — `check_overlaps(placements, threshold)`

Uses **pixel-based polygon rasterization** for accuracy (not approximate box overlap):

1. For each placement, compute rotated polygon corners
2. Compute exact polygon area via shoelace formula
3. For each pair, rasterize both polygons onto 640×640 binary masks using `cv2.fillPoly`
4. Count overlapping pixels: `np.count_nonzero(mask1 & mask2)`
5. Compare to `threshold × min(area_i, area_j)`

If any pair exceeds 5% overlap of the smaller photo's area, returns `False`.

**Why pixel-based instead of geometric?** Geometric polygon clipping (Sutherland-Hodgman) is accurate but complex and requires Shapely. Pixel rasterization at 640×640 is fast enough for 1–4 polygons, exact for convex shapes, and trivially correct — no edge cases with degenerate intersections.

### 5.7 Rotation Expansion Mathematics

Understanding why rotation kills multi-photo layouts:

A square photo of side `s` at angle `θ` has a rotated bounding box with diagonal:

```
expanded_side = s × (|cos θ| + |sin θ|)
```

| Rotation | Expansion Factor | 270px photo expands to |
|----------|-----------------|----------------------|
| 0° | 1.000 | 270px |
| 5° | 1.083 | 292px |
| 10° | 1.159 | 313px |
| 15° | 1.225 | 331px |
| 20° | 1.282 | 346px |
| 25° | 1.329 | 359px |
| 30° | 1.366 | 369px |

At 30°, a 270px photo expands to 369px — exceeding the ~305px clearance available in a 2×2 grid cell. This is why multi-photo layouts are limited to ±5° rotation.

---

## 6. Photo Compositing

### 6.1 Source Loading and Sizing

```python
photo = cv2.imread(random_source)
scale = min(placement['width'] / w_orig, placement['height'] / h_orig)
photo = cv2.resize(photo, (int(w_orig * scale), int(h_orig * scale)))
```

The scale factor is the **minimum** of width-ratio and height-ratio, ensuring the photo fits within the target dimensions while maintaining its original aspect ratio. The photo is never stretched — only scaled down.

The **5,062 source images** in `data_generator/images/` are from the [Oxford Buildings Dataset](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) (Oxford5k), providing diverse content (architecture, landscapes, objects). Download and extract them into this directory before running the generator.

### 6.2 Glare Effect — `fast_glare(img)`

Applied **before rotation**, so glare is fixed to the photo content.

- **Probability**: 50% chance of applying
- **Flares**: 2–4 elliptical gaussian flares per photo
- **Position**: random within 15–85% of width, 10–70% of height (biased toward top)
- **Ellipse radii**: 20–40% of dimensions
- **Opacity**: 60–100%
- **Blend mode**: screen blend (`1 - (1-img)(1-flare)`)
- **Blur**: `cv2.GaussianBlur` with 15×15 kernel on the flare mask

This simulates specular reflections from overhead lighting on glossy photo paper.

### 6.3 Rotation — `rotate_photo(photo, angle)`

- Uses `cv2.getRotationMatrix2D` around photo center
- Expands the output dimensions to contain the full rotated image (no clipping)
- Uses `cv2.INTER_LINEAR` interpolation
- **Border**: transparent black `(0, 0, 0, 0)` — essential for alpha compositing
- Converts to BGRA if not already (adds alpha=255)

**Optimization**: If `|angle| < 1°`, returns the photo unchanged.

### 6.4 Drop Shadow — `apply_photo_shadow(canvas, photo, cx, cy, offset_x, offset_y, blur_sigma, opacity, orig_w, orig_h, rotation)`

Applied **before** the photo is composited, so the shadow appears behind it.

**Random parameters per photo:**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `shadow_offset` | 2–5 px | Distance from photo center |
| `angle` | 0–2π | Random direction |
| `offset_x, offset_y` | Computed from offset + angle | Cartesian offset |
| `opacity` | 0.15–0.35 | Shadow darkness |
| `blur_sigma` | 1.5–3.0 | Gaussian blur radius |

**Rendering algorithm:**

1. **Create shadow mask at ORIGINAL photo dimensions** (`orig_w × orig_h`), not the expanded post-rotation dimensions. This is critical — using the post-rotation bounding box would create an axis-aligned rectangle shadow that doesn't match the photo shape.

2. **Blur BEFORE rotation** using `cv2.GaussianBlur` with sigma=`blur_sigma`. Pre-rotation blur ensures omnidirectional softness (a post-rotation blur would be stretched along the rotation axis).

3. **Padding**: `blur_pad = int(3 × sigma) + 1` pixels on each side, giving the blur room to fade to zero.

4. **Rotate the blurred mask** using the same rotation matrix and expansion formula as `rotate_photo`.

5. **Normalize** the mask so peak = 1.0 (blur reduces the peak).

6. **Rotate the offset direction** by the photo's rotation angle, so the shadow always falls in the same physical direction relative to the scene (e.g., always down-right regardless of photo orientation). The offset is halved (`× 0.5`) for subtlety.

7. **Composite onto canvas**: For each pixel in the shadow region, darken the canvas: `canvas_channel *= (1 - shadow_value × opacity)`. This is a multiply-blend darkening operation.

### 6.5 Alpha Compositing — `composite_photo_at_center(canvas, photo, cx, cy)`

Standard Porter-Duff "over" compositing:

```
result = photo × photo_alpha + canvas × (1 - photo_alpha)
result_alpha = max(canvas_alpha, photo_alpha)
```

- Works with BGRA canvas and BGRA photo
- Clips to canvas bounds (photos partially off-canvas are handled gracefully)
- All arithmetic in float32, converted back to uint8

### 6.6 Processing Order Per Photo

```
1. Load source → resize to target dimensions
2. fast_glare()          ← content-fixed specular highlights
3. Convert to BGRA
4. rotate_photo()        ← expand canvas, transparent border
5. apply_photo_shadow()  ← darken canvas BEHIND photo
6. composite_photo_at_center()  ← paint photo ON canvas
7. Record corners for labels
```

Steps 5 and 6 must be in this order: shadow first (behind), then photo (on top).

---

## 7. Perspective Warp

### 7.1 `apply_perspective_safe(canvas, corners_list)`

Applies a single perspective transformation to the **entire composite** (all photos + background at once). This simulates a camera viewing angle.

**Algorithm:**

1. Start with `max_strength = 0.05` (5% of canvas dimensions ≈ 32px max displacement)
2. Try 25 strength levels from 5% down to 0% (`np.linspace(5%, 0%, 25)`)
3. At each level, generate random corner displacements:
   ```
   TL: (-disp, -disp)   TR: (+disp, -disp)
   BL: (-disp, +disp)   BR: (+disp, +disp)
   ```
4. Compute perspective matrix `M = cv2.getPerspectiveTransform(src_pts, dst_pts)`
5. **Validate**: transform ALL photo corners through `M` and check they remain within `[15, canvas_size - 15]`
6. If valid: apply `cv2.warpPerspective` with gray border fill `(128, 128, 128)` and return
7. If no strength level works: return the un-warped canvas with identity matrix

**Safety margin**: `safety_margin = 15px`. This ensures that even after perspective, no photo corner is within 15px of the edge — giving the pose model room to detect corners without them being at the very edge of the image.

**Border fill**: Gray `(128, 128, 128)` — neutral color that doesn't introduce false edges.

### 7.2 Corner Transformation

After `apply_perspective_safe` returns the homography matrix `M`, all photo corners are transformed:

```python
for each corner (x, y):
    pt = [x, y, 1]
    result = M @ pt
    warped_x = result[0] / result[2]
    warped_y = result[1] / result[2]
```

This is a homogeneous coordinate transformation — the third component `result[2]` handles the perspective foreshortening.

---

## 8. Label Generation

### 8.1 Detection Labels

For each photo, the axis-aligned bounding box is computed from the (perspective-warped) corners:

```python
min_x = min(c[0] for c in corners)
max_x = max(c[0] for c in corners)
min_y = min(c[1] for c in corners)
max_y = max(c[1] for c in corners)

x_center = ((min_x + max_x) / 2) / CANVAS_SIZE   # Normalized [0, 1]
y_center = ((min_y + max_y) / 2) / CANVAS_SIZE
width    = (max_x - min_x) / CANVAS_SIZE
height   = (max_y - min_y) / CANVAS_SIZE
```

**Format**: `0 x_center y_center width height` (5 columns, space-separated)

Example: `0 0.501715 0.505848 0.378018 0.590964`

All values are **normalized** to `[0, 1]` by dividing by `CANVAS_SIZE` (640).

### 8.2 Pose Labels

Same bounding box as detection, plus the 4 corner keypoints:

```python
corners_str = " ".join([
    f"{corners[i, 0] / CANVAS_SIZE:.6f} "
    f"{corners[i, 1] / CANVAS_SIZE:.6f} 2"
    for i in range(4)
])
```

**Format**: `0 x_center y_center width height kp0x kp0y kp0v kp1x kp1y kp1v kp2x kp2y kp2v kp3x kp3y kp3v` (13 columns)

Example: `0 0.501715 0.505848 0.378018 0.590964 0.312194 0.801330 2 0.690212 0.801330 2 0.689914 0.210366 2 0.314542 0.210366 2`

**Keypoint visibility**: Always `2` (visible). Value `2` in YOLO-Pose means "labeled and visible" (vs `0` = not labeled, `1` = labeled but not visible).

**Keypoint order** (CRITICAL — must match training YAML `flip_idx`):

| Index | Name | Description | Coordinate Convention |
|-------|------|-------------|----------------------|
| kp0 | LL | Lower-Left | (min X, max Y) |
| kp1 | UL | Upper-Left | (min X, min Y) |
| kp2 | UR | Upper-Right | (max X, min Y) |
| kp3 | LR | Lower-Right | (max X, max Y) |

**Horizontal flip mapping**: `flip_idx: [3, 2, 1, 0]` — when the image is flipped horizontally:
- LL (kp0) ↔ LR (kp3)
- UL (kp1) ↔ UR (kp2)

---

## 9. Execution Modes

### 9.1 Examples Mode (default)

```bash
python generate.py --count 10 --source ./images --output ./data/examples
```

Generates `--count` images with debug overlays:
- `example_01.jpg` — the composite image
- `example_01_debug.jpg` — same image with white polygon outlines, colored corner dots (10px radius), and corner labels (LL, UL, UR, LR)
- `example_01_det.txt` — detection labels
- `example_01_pose.txt` — pose labels

Debug colors:
| Corner | Color |
|--------|-------|
| LL | Green `(0, 255, 0)` |
| UL | Red `(255, 0, 0)` |
| UR | Yellow `(0, 255, 255)` |
| LR | Magenta `(255, 0, 255)` |

### 9.2 Batch Mode

```bash
python generate.py --mode batch --train-count 4000 --val-count 1000 --source ./images --output ../data
```

Generates a full dataset with train/val split:

```
data/
├── images/
│   ├── train/              # train_000001.jpg ... train_004000.jpg
│   └── val/                # val_000001.jpg ... val_001000.jpg
├── detection/
│   └── labels/
│       ├── train/          # 5-column YOLO detection labels
│       └── val/
└── pose/
    └── labels/
        ├── train/          # 13-column YOLO-Pose labels
        └── val/
```

**Same images, different labels** — the JPEGs are shared between both model training pipelines.

Progress is logged every 20 images with elapsed time, generation rate, and ETA.

---

## 10. Debug and Verification

### 10.1 Bounds Verification (runtime)

After compositing, each image is checked for out-of-bounds corners:

```python
for each photo:
    for each corner:
        if corner outside [0, CANVAS_SIZE]:
            count as OOB
if oob_count > 0:
    print("WARNING: N corners out of bounds")
```

This is a sanity check — `pack_photos_validated` should prevent this, but the check catches any bugs in the placement engine.

### 10.2 Composited Pixel Verification — `verify_composited_pixels(canvas, expected_areas)`

Compares the total opaque pixel count (alpha > 128) against the expected sum of photo areas. Returns `True` if within 80–110% tolerance (allows for clipping at edges and anti-aliasing).

Currently defined but **not called** in the main generation path — available for quality assurance runs.

### 10.3 Debug Image — `create_debug_image(img, photos)`

Overlays on the composite image:
- White polygon outline (2px) around each photo
- Colored filled circles (10px) at each corner
- Text labels: "LL", "UL", "UR", "LR" offset 12px from each corner

---

## 11. Complete Function Reference

| Function | Lines | Purpose |
|----------|-------|---------|
| `compute_rotated_bbox` | 55–63 | Axis-aligned bbox of rotated rectangle |
| `check_bounds` | 63–77 | Verify all rotated corners within canvas margin |
| `polygon_area` | 77–88 | Shoelace formula for polygon area |
| `polygon_intersection_area` | 88–124 | Pixel-rasterization overlap between two polygons |
| `corners_to_int_points` | 124–130 | Convert float corners to int32 for OpenCV |
| `bounding_box` | 130–137 | Min/max bounding box of polygon corners |
| `check_overlaps` | 137–170 | Pairwise overlap check with 5% threshold |
| `verify_composited_pixels` | 170–192 | Post-composite pixel count verification |
| `pack_photos_validated` | 192–221 | Retry loop: generate placements, validate, return |
| `_shrink_to_fit` | 221–263 | Iterative photo shrinking for bounds compliance |
| `_generate_placements` | 263–402 | Layout-specific position/size/rotation generation |
| `get_rotated_polygon` | 402–440 | Compute 4 corner coordinates of rotated rectangle |
| `rotate_photo` | 440–466 | Rotate BGRA photo with transparent border |
| `apply_photo_shadow` | 466–560 | Mask-blur-rotate-composite drop shadow |
| `composite_photo_at_center` | 560–631 | Porter-Duff alpha compositing |
| `random_base_background` | 631–662 | Color + noise background generation |
| `apply_3_linear_gradients` | 662–695 | 3× screen-blend gradient overlays |
| `fast_glare` | 695–724 | Elliptical gaussian flare screen-blend |
| `apply_texture_overlay` | 724–767 | Random texture blend with screen/multiply |
| `apply_perspective_safe` | 767–825 | Corner-safe perspective warp with fallback |
| `generate_image` | 825–944 | Full pipeline: background → placement → composite → warp → labels |
| `create_debug_image` | 944–963 | Overlay polygons and corner labels |
| `main` | 963–988 | Argparse entry point |
| `_example_generate` | 988–1018 | Generate debug images with overlays |
| `_batch_generate` | 1018–1084 | Generate train/val dataset split |

---

## 12. Known Limitations and Design Tradeoffs

### 12.1 Minimum Photo Size vs. Bounds Compliance

`_shrink_to_fit` has a `min_side` floor of 270px. If a photo at 270px still exceeds the boundary margin, it is accepted as-is (the margin is violated). This is a deliberate tradeoff: slightly out-of-bounds corners are invisible in training, but undersized photos produce bad training data. In practice, the 5px margin is tiny enough that violations of 1–3 pixels are imperceptible.

### 12.2 Overlap Threshold

The 5% overlap threshold means photos can overlap by up to 5% of the smaller photo's area. This is intentional — zero-tolerance overlap checking would reject most multi-photo layouts, especially after rotation. A small overlap simulates real-world photos that slightly overlap on a table.

### 12.3 Grid Layout Always 2×2

The grid layout for 3–4 photos always uses a 2×2 grid, regardless of whether there are 3 or 4 photos. For 3 photos, one cell is left empty. A 3×1 horizontal or 1×3 vertical layout is not implemented — the 2×2 grid provides the most reliable sizing.

### 12.4 No Photo Edge Blur

The current implementation does NOT apply edge blur to photos. Previous versions had a Gaussian blur on photo edges to simulate soft focus / depth-of-field, but this was removed during the consolidation. It can be re-added as a post-rotation, pre-composite step if needed.

### 12.5 Single Perspective Warp

Perspective is applied once to the entire composite, not individually per photo. This correctly simulates a single camera viewing angle — all photos on the same surface would share the same perspective distortion. However, it means the warp strength is limited by the least-tolerant photo (the one nearest the edge).

### 12.6 Rasterization Overlap Check

`check_overlaps` uses 640×640 pixel rasterization, which has ~1px quantization error. For 270px+ photos, this is negligible (< 0.5% area error). For smaller photos it would be a concern, but the 270px minimum prevents this.

---

## 13. Configuration Tuning Guide

### Increasing photo sizes further

1. Increase `PHOTO_SIZE_MIN` (e.g., 300)
2. Reduce multi-photo rotation (e.g., `rot_range = 3` for 2-photo)
3. May need to increase `BOUND_MARGIN` slightly to compensate
4. Expect fewer multi-photo images (tighter packing → more validation failures)

### Increasing rotation variety

1. Increase `rot_range` for multi-photo layouts
2. Photos will be smaller (rotation expansion consumes available space)
3. May need to decrease `PHOTO_SIZE_MIN` to compensate
4. Grid layouts with >5° rotation will have very small photos

### Adding more photos per image

1. Increase `NUM_PHOTOS_MAX` (e.g., 6 or 8)
2. Would need a new layout strategy — 2×2 grid only supports up to 4
3. Consider a 3×2 or 3×3 grid with correspondingly smaller cells
4. At 3×3, cell size ≈ 203px, well below `PHOTO_SIZE_MIN = 270` — would need to relax minimum

### Disabling perspective warp

Set `max_strength = 0.0` in `apply_perspective_safe`, or comment out the call in `generate_image`. Labels will use the identity matrix for corner transformation (no change).

---

## 14. Source Assets

| Asset | Location | Count | Description |
|-------|----------|-------|-------------|
| Source photos | `data_generator/images/` | 5,062 | [Oxford Buildings Dataset](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) — architecture, objects, scenes (`.jpg`). Download and extract into `data_generator/images/` |
| Background textures | `textures/` | 85 | [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) — processed to 1200×1200 greyscale, brightness-normalized (`.jpg`) |

**Oxford Buildings Dataset:** The 5,062 source images are from the [Oxford Buildings Dataset](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) (Oxford5k), created by the Visual Geometry Group at the University of Oxford. It is available under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Use `python download_oxford.py` to download and extract into `data_generator/images/`.

**Describable Textures Dataset (DTD):** The 85 background textures are from the [Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/), created by the Visual Geometry Group at the University of Oxford. Available under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. Use `python download_textures.py` to download, process, and install into `textures/`.

Source photos are randomly selected for each placement (with replacement), so the same photo may appear multiple times in one image or across images. This is intentional — the model must learn to detect photos regardless of content.