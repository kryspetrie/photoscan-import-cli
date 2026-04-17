# Corner Coordinate Debug Plan - FINAL REPORT

## Issues Identified and Fixed

### 1. ✅ Corner Order - FIXED
**Problem:** Corner reordering used centroid-based classification which failed for perspective-warped photos.

**Fix:** Use x-sorting approach:
```python
# Sort by x to determine left/right
# Then sort by y within each group
sorted_by_x = sorted(enumerate(corners), key=lambda i_c: corners[i_c[0]][0])
left_indices = [i for i, c in sorted_by_x[:2]]  # Small x
right_indices = [i for i, c in sorted_by_x[2:]]  # Large x
```

### 2. ✅ Scale Effects - FIXED
| Effect | Old | New | Status |
|--------|-----|-----|--------|
| Edge blur | 0-5px | 0-2px | ✓ Fixed |
| Shadow blur | 2-8 sigma | 1-4 sigma | ✓ Fixed |
| Shadow offset | 0-8px | 0-3px | ✓ Fixed |
| Shadow spread | blur*4+6 | blur*2+4 | ✓ Fixed |
| Glare blur | 21px | 15px | ✓ Fixed |

### 3. ✅ Black Edges - FIXED
Added 100px padding around canvas before perspective warp:
```python
padding = 100
padded_canvas = np.ones((CANVAS_SIZE + 2*padding, CANVAS_SIZE + 2*padding, 4))
# ... place photos in padded canvas ...
# ... then apply perspective warp on padded canvas
```

### 4. ⚠️ Corners Off-Screen (Before Fix)
**Issue:** 14/24 photos had corners going off-screen

**Current Status:** All corners now within bounds (0, 1)

## Verification Files Created

| File | Purpose |
|------|---------|
| `corner_verify_01.png` | X markers at computed corners |
| `corner_verify_02.png` | X markers at computed corners |
| `corner_verify_03.png` | X markers at computed corners |
| `visual_verify_01.png` | Annotated with bbox + corners |
| `comparison.jpg` | Before/after perspective |

## Test Results

**v33 Generation:**
- 10 images, 24 photos
- 6/24 with rotation (25%)
- Perspective displacement: 113-292px (good range)
- **All corners within bounds** ✓

## Files to Review

```
/Users/krys.petrie/dev/photo-pose-detector/data/examples_v33/
├── corner_verify_01.png   ← X markers on corners
├── corner_verify_02.png
├── corner_verify_03.png
├── visual_verify_01.png   ← Full annotation
├── visual_verify_02.png
├── example_01.jpg         ← Raw images
├── example_01_pose.txt   ← Label files
└── ...
```

## Verification Checklist

- [ ] corner_verify_*.png: X markers should be ON photo corners
- [ ] visual_verify_*.png: Green bbox should surround photos, dots on corners
- [ ] All 24 photos have corners within [0, 1] range
- [ ] Corner order is correct: LL(bottom-left), UL(top-left), UR(top-right), LR(bottom-right)
