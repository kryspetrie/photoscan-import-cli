# Photo Pose Detector - System Documentation

**Date:** 2026-04-17

## Overview

This document describes the synthetic training data generator for a YOLO-based photo pose detection system. The system generates training images for two models:
1. **Detection Model**: Standard YOLO detection with axis-aligned bounding boxes
2. **Pose Model**: Keypoint detection for 4 corners of each photo (LL, UL, UR, LR)

---

## Coordinate Spaces

The pipeline transforms photos through several coordinate spaces:

```
Photo Space (e.g., 400x300)
    ↓ rotate_photo()
Rotated Photo Space (e.g., 500x400)  
    ↓ composite_photo_at_center()
Padded Canvas Space (1240x1240) ← padding of 300px on each side
    ↓ apply_global_perspective() + crop
Cropped Canvas Space (1120x1120) ← removes 60px margin
    ↓ resize
Output Space (640x640) ← final YOLO input
```

---

## Configuration Constants

```python
CONFIG = {
    'CANVAS_SIZE': 640,           # Output canvas size
    'CANVAS_PADDING': 300,         # Extra padding for perspective warp headroom
    'CROP_MARGIN': 60,             # Margin removed after perspective warp
    'PHOTO_SIZE_MIN': 180,         # Minimum photo dimension after scaling
    'PHOTO_SIZE_MAX': 480,         # Maximum photo dimension
    'ROTATION_RANGE': 30,          # Max rotation in degrees
    'PERSPECTIVE_STRENGTH_MIN': 0.05,  # 5% perspective
    'PERSPECTIVE_STRENGTH_MAX': 0.20,  # 20% perspective
}
```

Derived values:
- `PADDED_CANVAS_SIZE = CANVAS_SIZE + 2 * CANVAS_PADDING = 1240`
- `OUTPUT_SIZE = 640`

---

## Core Pipeline Functions

### 1. `rotate_photo(photo, angle)`

Rotates a photo around its center and expands the canvas to fit.

```python
def rotate_photo(photo, angle):
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo, (w, h), None
    
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new canvas size to fit rotated image
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Shift matrix to account for expanded canvas
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(photo, M, (new_w, new_h), ...)
    
    return rotated, (new_w, new_h), M
```

**Returns:**
- Rotated photo
- Rotated dimensions (new_w, new_h)
- Transformation matrix M (includes canvas expansion offset)

---

### 2. `composite_photo_at_center(canvas, photo, cx, cy)`

Places a photo on a canvas with its center at (cx, cy).

```python
def composite_photo_at_center(canvas, photo, cx, cy):
    ph, pw = photo.shape[:2]
    
    top_left_x = int(cx - pw / 2)
    top_left_y = int(cy - ph / 2)
    
    # Clip to canvas bounds
    # ... boundary checking ...
    
    canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
        photo[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
    
    return canvas, (top_left_x, top_left_y)
```

**Key insight:** The photo's top-left corner in canvas space is at:
```
top_left = (cx - pw/2, cy - ph/2)
```

---

### 3. `apply_global_perspective(canvas, canvas_w, canvas_h, crop_margin)`

Applies a random perspective warp to the entire canvas and crops to output size.

```python
def apply_global_perspective(canvas, canvas_w, canvas_h, crop_margin=60):
    max_disp = int(canvas_w * 0.20)  # 20% max displacement
    
    # Random corner displacements
    tl = (random.randint(-max_disp, 0), random.randint(-max_disp, 0))
    tr = (random.randint(0, max_disp), random.randint(-max_disp, 0))
    bl = (random.randint(-max_disp, 0), random.randint(0, max_disp))
    br = (random.randint(0, max_disp), random.randint(0, max_disp))
    
    # Source: full canvas corners
    src_pts = np.array([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]], dtype=np.float32)
    
    # Destination: corners with displacement
    dst_pts = np.array([
        [tl[0], tl[1]],
        [canvas_w + tr[0], tr[1]],
        [canvas_w + br[0], canvas_h + br[1]],
        [bl[0], canvas_h + bl[1]]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(canvas, M, (canvas_w, canvas_h), ...)
    
    # Crop to output size
    result = warped[crop_margin:crop_margin+640, crop_margin:crop_margin+640]
    
    return result, (tl, tr, bl, br), M
```

---

## Polygon Calculation

### `get_rotated_polygon(width, height, center_x, center_y, rotation)`

Calculates the 4 corner coordinates of a rotated rectangle in **placement space** (padded canvas coordinates).

```python
def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """
    Returns corners in placement space:
    [0] = LL (Lower-Left)
    [1] = UL (Upper-Left)
    [2] = UR (Upper-Right)
    [3] = LR (Lower-Right)
    """
    if abs(rotation) < 1:
        return np.array([
            [center_x - width/2, center_y - height/2],  # LL
            [center_x + width/2, center_y - height/2],  # UL
            [center_x + width/2, center_y + height/2],  # UR
            [center_x - width/2, center_y + height/2]   # LR
        ], dtype=np.float32)
    
    # Use M_raw - rotation matrix WITHOUT canvas expansion offset
    photo_center = (width / 2, height / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    # Canvas offset: where photo origin (0,0) lands in placement space
    canvas_offset = (center_x - width / 2, center_y - height / 2)
    
    # Original photo corners
    corners_photo = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ], dtype=np.float32)
    
    # Transform each corner
    corners_final = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        corners_final[i, 0] = canvas_offset[0] + rotated[0]
        corners_final[i, 1] = canvas_offset[1] + rotated[1]
    
    return corners_final
```

**Critical:** Uses `M_raw` (NOT M with canvas expansion). The canvas expansion is already baked into where the rotated photo is placed via `composite_photo_at_center`.

---

## Transformation Pipeline for Corners

To get corners in output space (640x640):

```python
# 1. Corners in placement space (padded canvas, before perspective)
corners_placement = get_rotated_polygon(w, h, center_x, center_y, rotation)

# 2. Transform through perspective warp
persp_M = cv2.getPerspectiveTransform(src_pts, dst_pts)
corners_persp = apply_perspective_transform(corners_placement, persp_M)

# 3. Crop (subtract margin)
corners_cropped = corners_persp.copy()
corners_cropped[:, 0] -= CROP_MARGIN  # 60
corners_cropped[:, 1] -= CROP_MARGIN  # 60

# 4. Scale to output size
scale = OUTPUT_SIZE / (PADDED_CANVAS_SIZE - 2 * CROP_MARGIN)
# scale = 640 / (1240 - 120) = 640 / 1120
corners_output = corners_cropped * scale
```

---

## Key Insights & Pitfalls

### 1. Rotation Direction and Translation (CRITICAL - DISCOVERED 2026-04-17)

**The Bug:** The `get_rotated_polygon` function had multiple issues:

1. **Direction:** OpenCV's `getRotationMatrix2D(angle)` performs **clockwise** rotation. The function was using counterclockwise (-angle).

2. **Translation:** The function was using `canvas_offset = (center_x - photo_w/2, center_y - photo_h/2)` (based on original dimensions), but the rotated photo has DIFFERENT dimensions.

3. **Separate offset:** Adding a separate `canvas_offset` was double-counting the translation that M already includes.

**The Fix:** Use the rotation matrix M with its translation, then add the top_left offset:
```python
def get_rotated_polygon(width, height, center_x, center_y, rotation):
    if abs(rotation) < 1:
        return np.array([
            [center_x - width/2, center_y + height/2],  # LL
            [center_x - width/2, center_y - height/2],  # UL
            [center_x + width/2, center_y - height/2],  # UR
            [center_x + width/2, center_y + height/2]   # LR
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    
    # Get rotation matrix (clockwise = positive angle, matching warpAffine)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    # Calculate new dimensions after rotation
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(height * sin + width * cos)
    new_h = int(width * sin + height * cos)
    
    # Add translation to center the rotated photo (same as rotate_photo)
    M[0, 2] += (new_w - width) / 2
    M[1, 2] += (new_h - height) / 2
    
    # Where does the top-left of the rotated photo land in canvas?
    top_left_x = center_x - new_w / 2
    top_left_y = center_y - new_h / 2
    
    # Corners in photo space (clockwise from bottom-left)
    corners_photo = np.array([
        [0, height],      # LL - bottom-left
        [0, 0],            # UL - top-left
        [width, 0],        # UR - top-right
        [width, height]   # LR - bottom-right
    ], dtype=np.float32)
    
    # Transform: M @ corner gives position in rotated photo
    # Then add top_left for canvas position
    corners_final = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M @ pt
        corners_final[i, 0] = top_left_x + rotated[0]
        corners_final[i, 1] = top_left_y + rotated[1]
    
    return corners_final
```

**Key insight:** The matrix M already includes translation to center the rotated photo, but in PHOTO space. We need to add the top_left offset to place it in CANVAS space.

### 2. Canvas Expansion Offset (CRITICAL - Previous version)

**Historical bug:** Originally, `get_rotated_polygon` used `M` (with canvas expansion) AND then applied centering formula, counting the offset TWICE. This was fixed by using `M_raw` but then that introduced the current bugs. The corrected approach above fixes both issues.

### 3. Photo Scaling

Real photos must be scaled to generator working size (180-480px) before processing:
```python
size = random.randint(180, 480)
height = int(size * 0.8)
width = int(height * photo_aspect_ratio)
scaled_photo = cv2.resize(photo, (width, height))
```

### 4. Bow-Tie Issue (Perspective Inversion)

When perspective displacements are too extreme, corners can cross over each other, creating a "bow-tie" or degenerate quadrilateral. The 20% max displacement should prevent this for most cases.

### 5. Boundary Clipping

When a rotated photo extends beyond canvas bounds, `composite_photo_at_center` clips the photo and adjusts source coordinates. This means some corners may be lost. The 300px padding helps prevent this.

### 6. Coordinate System Confusion (OpenCV vs Math)

OpenCV uses (x, y) where x increases right, y increases DOWN. This is different from standard math where y increases UP. Always be careful when converting between coordinate systems.

### 7. Corner vs Centroid Verification (IMPORTANT)

When verifying the corner formula using colored markers:
- If using **squares/rectangles** as markers, the detected centroid is **NOT** at the corner
- The centroid is offset by half the marker size from each edge
- Example: A 30x30 pixel marker at the bottom-left corner has its centroid at (15, h-15), not (0, h)
- To verify against corners, either use **1-pixel corner markers** or **account for the offset**

The corrected formula gives **<2px accuracy** when comparing detected centroid to expected marker position.

---

## YOLO Label Formats

### Detection (class, x_center, y_center, width, height)
Normalized to image size:
```
0 0.500 0.500 0.400 0.300
```

### Pose (keypoints)
```
# For each keypoint: x, y, visibility
# Visibility: 0=hidden, 1=visible
# Order: LL, UL, UR, LR

# Example (all visible):
0.350 0.300 1 0.650 0.300 1 0.650 0.600 1 0.350 0.600 1
```

---

## Testing & Verification

### Test Script: `test_final_verification.py`

Creates synthetic photos with known corner positions and verifies the polygon calculation through the full pipeline.

**Test Results:**
- 32 corner measurements
- Average error: 2.89px
- Maximum error: 4.33px
- 100% within 5px accuracy

### Verification Script: `verify_scaled.py`

Uses real photographs and overlays calculated corner positions for visual verification.

---

## File Structure

```
data_generator/
├── generate_dataset.py      # Main generator (v34 - FIXED)
├── test_final_verification.py  # Unit tests for polygon math
├── verify_scaled.py         # Visual verification with real photos
├── images/                  # Source photographs
│   ├── all_souls_*.jpg
│   ├── ashleian_*.jpg
│   └── ...
├── textures/               # Background textures
│   └── texture_*.jpg
└── verification_scaled/    # Generated verification images
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| ~10-14px systematic offset | Double-counting canvas expansion | Use `M_raw` not `M` |
| Corners outside output bounds | Photo too large for canvas | Scale photos to 180-480px |
| Bow-tie polygon | Extreme perspective | Limit to 20% max displacement |
| Black edges after warp | Insufficient padding | Use 300px padding |
| Poor corner detection | Markers too small | Use 7px+ marker size |

---

## Version History

- **v34 (Current)**: Fixed canvas size and shadow scaling, corrected polygon calculation
- Previous versions had double-counted canvas expansion offset in `get_rotated_polygon()`

---

## References

- OpenCV `getRotationMatrix2D`: Creates 2D rotation matrix
- OpenCV `warpAffine`: Applies affine transformation
- OpenCV `getPerspectiveTransform`: Creates 3x3 perspective matrix
- OpenCV `warpPerspective`: Applies perspective transformation
