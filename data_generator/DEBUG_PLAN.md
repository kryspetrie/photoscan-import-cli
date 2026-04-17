# Coordinate Transform Debug Plan - COMPLETED

## Problem Identified
Bounding boxes and corner keypoints didn't match actual photo positions due to:
1. **Crop logic bug**: Original code cropped based on photo corners, but photo corners could be scattered
2. The crop was based on `all_corners` min/max instead of full warped canvas bounds

## Bug Location
In `apply_global_perspective()` function:
```python
# BUG: Crop based on photo corners (scattered, wrong)
if warped_photo_corners:
    all_corners = np.vstack(warped_photo_corners)
    crop_x1 = max(0, int(all_min_x) - margin)  # WRONG: all_min_x might be photo corner, not canvas edge
```

## Fix Applied
```python
# FIXED: Crop based on full warped canvas bounds
warped_h, warped_w = warped.shape[:2]
crop_x1 = crop_margin
crop_y1 = crop_margin
crop_x2 = warped_w - crop_margin
crop_y2 = warped_h - crop_margin

if crop_x2 > crop_x1 + 200 and crop_y2 > crop_y1 + 200:
    warped = warped[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # CORRECTION: Adjust corners for crop offset
    for corners in warped_photo_corners:
        corners[:, 0] -= crop_offset_x
        corners[:, 1] -= crop_offset_y
```

## Test Results

### Debug Test (debug_transform.py)
- Created 5 test cases with known inputs
- Verified coordinate transformation is correct
- Final corners match original corners after transform
- Transformation pipeline: Place → Perspective → Crop → Resize

### Generated Images (examples_v33)
- 10 images generated successfully
- 30 total photos (1-5 per image)
- 12/30 with rotation
- Perspective displacement: 58-245px (good range)

### Verification Images
Created `verify_01.png` - `verify_10.png` showing:
- Green bbox: Detection bounding box
- Blue dot: LL (Lower-Left)
- Yellow dot: UL (Upper-Left)
- Green dot: UR (Upper-Right)
- Orange dot: LR (Lower-Right)
- Cyan lines: Quadrilateral from corners

## Files Created

| File | Purpose |
|------|---------|
| `debug_transform.py` | Isolated coordinate transform test |
| `verify_labels.py` | Creates annotated verification images |
| `DEBUG_PLAN.md` | This document |

## Output Location
```
/Users/krys.petrie/dev/photo-pose-detector/data/examples_v33/
├── example_01.jpg - example_10.jpg    # Generated images
├── example_01_det.txt - ..._det.txt   # Detection labels
├── example_01_pose.txt - ..._pose.txt # Pose labels
├── verify_01.png - verify_10.png      # Annotated verification
└── comparison_2x2.png                # Debug comparison
```

## Next Steps
1. **User verifies** the `verify_*.png` images match actual photo corners
2. If correct: Regenerate full 5000 image dataset
3. If incorrect: Debug further using `debug_transform.py`

## Key Insight
The transformation math was always correct. The bug was in **what we crop** - we were cropping based on photo corners (which scatter after perspective) instead of the full warped canvas bounds.
