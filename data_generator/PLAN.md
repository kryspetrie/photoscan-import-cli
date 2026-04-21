# Plan: Fix Photo Overlaps and Off-Screen Placement in generate_v2.py

## Problems

1. **Photo overlaps**: Multiple photos can overlap each other on the canvas. `pack_photos_simple()` uses heuristic positions that don't account for rotated bounding boxes.

2. **Photos off-screen**: Photos (especially after rotation) can extend beyond the canvas borders. The `EDGE_MARGIN` of 50px is too small for 30° rotations on large photos.

3. **No validation**: There is no post-placement check to verify placement quality.

## Approach: Pixel-based overlap and bounds checking

### Step 1: Add `compute_rotated_bbox()` helper
For a photo with `(width, height, center_x, center_y, rotation)`, compute the axis-aligned bounding box of the rotated rectangle. This is exactly what `get_rotated_polygon` already does — extract min/max from its corners.

### Step 2: Add `check_overlaps()` function
Given a list of placements (each with `width, height, center_x, center_y, rotation`), compute rotated bounding boxes and check if any pair overlaps by more than a threshold percentage.

- Compute the rotated polygon for each photo
- For each pair, compute the intersection area as a fraction of the smaller photo's area
- If any pair exceeds `OVERLAP_THRESHOLD` (e.g., 5%), reject the placement

### Step 3: Add `check_bounds()` function
For each placement, compute the rotated bounding box and check that it stays within the canvas with a margin.

- Use `get_rotated_polygon()` to get all 4 corners
- Check that every corner is within `[MARGIN, CANVAS_SIZE - MARGIN]`
- The margin should account for rotation expansion: for the largest expected photo (480px) at max rotation (30°), the bounding box can expand by ~72px, so MARGIN should be ~80px

### Step 4: Add pixel-based `compute_composited_pixel_count()`
As a final validation after compositing:
- For each photo loaded, compute the expected number of opaque pixels (area of the rotated photo after clipping)
- After compositing all photos, count actual composited pixels using the alpha channel
- If the actual count is less than expected (minus a threshold), some photos went off-screen or were overlapped

### Step 5: Add `pack_photos_validated()` function
Replace `pack_photos_simple()` with a function that:
1. Generates candidate placements using the same logic
2. Checks bounds for each placement (rotated bbox within canvas margin)
3. Checks overlaps between all pairs
4. Retries with different random placements if validation fails (up to MAX_ATTEMPTS)
5. Falls back to fewer photos if can't place all without overlap

### Step 6: Update `generate_image()` to use validated packing and pixel verification

### Step 7: Regenerate 10 examples and visually verify

## Implementation Details

- `OVERLAP_THRESHOLD = 0.05` — allow up to 5% overlap
- `BOUND_MARGIN = 80` — minimum margin from canvas edge for rotated photo corners  
- `MAX_ATTEMPTS = 50` — retry placements this many times before reducing photo count
- Pixel verification: after compositing, check that total composited area is within 90-110% of expected

## Files to modify
- `generate_v2.py` — add validation functions, replace `pack_photos_simple` with validated version, update `generate_image`