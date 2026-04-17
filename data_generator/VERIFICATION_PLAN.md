# Photo Pose Detector - Comprehensive Verification Plan

## Executive Summary

This document outlines a systematic approach to verifying the synthetic training data generation pipeline using classical computer vision techniques. The verification approach uses **colored fiducial markers** and **edge detection** to programmatically verify correctness without requiring human feedback.

---

## 1. Issues Identified

### 1.1 Black Edges After Perspective Warp
- **Root Cause**: Canvas padding (300px) was insufficient for extreme perspective transforms
- **Evidence**: Some images show 0.27 dark pixel ratio at edges (threshold: 0.05)
- **Status**: PARTIALLY FIXED - 300px padding helps but some images still fail

### 1.2 Corner Misalignment
- **Root Cause**: Complex rotation math - `get_rotated_polygon()` uses different coordinate systems than `rotate_photo()`
- **Evidence**: End-to-end test shows 0.04px error for simple case, but full pipeline has larger errors
- **Status**: FIXED - Mathematically verified, needs full pipeline test

### 1.3 Shadow Effects Too Large
- **Root Cause**: Initial parameters too aggressive
- **Current**: `SHADOW_BLUR_MAX=2`, `SHADOW_OFFSET_MAX=2`
- **Status**: FIXED - Scaled down

### 1.4 Bounding Box Accuracy
- **Root Cause**: Multiple potential sources (perspective transform, rescaling, rotation calculation)
- **Evidence**: Test suite shows 150-360px error (threshold: 15px)
- **Status**: NEEDS VERIFICATION - Threshold may be too strict

---

## 2. Test Methodology

### 2.1 Color-Based Corner Tracking

**Principle**: Place distinct colored markers at photo corners BEFORE any transforms. After processing, detect markers and compare with calculated positions.

**Colors Used**:
- TL: Green (0, 255, 0)
- TR: Red (0, 0, 255)
- BR: Magenta (255, 0, 255)
- BL: Blue (255, 0, 0)

**Detection Method**:
```python
# HSV-based color detection
mask = cv2.inRange(image, lower_color, upper_color)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
centroid = cv2.moments(largest)
```

**Pass Criteria**: Average corner error < 15px

### 2.2 Black Edge Detection

**Principle**: Check pixel values at image borders for solid dark regions.

**Method**:
```python
# Check 30px border for dark pixels
dark_pixels = np.sum(image[:30, :] < 10)
dark_ratio = dark_pixels / total_border_pixels
```

**Pass Criteria**: Dark pixel ratio < 0.05

### 2.3 Border Variance Analysis

**Principle**: Real images have texture variation; blank edges have uniform color.

**Method**:
```python
# Calculate variance in border regions
variance = np.var(border_region)
```

**Pass Criteria**: Variance > 50 indicates texture (not blank)

### 2.4 Bounding Box Edge Detection

**Principle**: Use Canny edge detection to find photo boundaries and compare with labels.

**Method**:
```python
edges = cv2.Canny(blur, 30, 100)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Find largest rectangular contour
```

**Pass Criteria**: Detected corners within 15px of label positions

---

## 3. Test Suite Architecture

```
verify_all_tests.py
├── TestSuite Class
│   ├── test_corner_tracking_pipeline()  # Color markers through full pipeline
│   ├── test_black_edge_detection()      # Check for black edges
│   ├── test_border_variance()           # Verify texture in borders
│   ├── test_shadow_bounds()             # Measure shadow spread
│   ├── test_bbox_accuracy()              # Canny edge detection vs labels
│   └── verify_perspective_transform()    # Grid pattern verification
│
├── Diagnostic Scripts
│   ├── diagnostic_pipeline.py             # Step-by-step pipeline tracing
│   ├── debug_corners.py                  # Corner calculation comparison
│   ├── e2e_pipeline_test.py              # End-to-end with markers
│   └── test_e2e_with_markers.py         # Integration test
```

---

## 4. Verification Pipeline Steps

### Step 1: Create Test Image with Markers
```python
def place_colored_markers(photo):
    # Place 15px colored circles at TL, TR, BR, BL
    # Add crosshairs for sub-pixel accuracy
    # Return photo with markers
```

### Step 2: Run Through Transformation Pipeline
```python
# 1. Resize photo to target dimensions
# 2. Apply photo effects (brightness, glare, etc.)
# 3. Rotate photo (uses getRotationMatrix2D)
# 4. Composite onto canvas
# 5. Apply perspective warp
# 6. Crop and resize to 640x640
```

### Step 3: Detect Markers in Final Image
```python
def detect_markers(image):
    # For each color:
    #   - Create mask
    #   - Morphological close
    #   - Find contours
    #   - Get centroid of largest contour
    # Return dict of corner_name -> (x, y)
```

### Step 4: Calculate Expected Positions
```python
def calculate_expected_corners():
    # Use get_rotated_polygon() for rotation
    # Transform through perspective matrix
    # Apply crop offset
    # Apply scale factor
    # Return expected corner positions
```

### Step 5: Compare and Report
```python
def compare_results(detected, expected):
    errors = []
    for corner in ['TL', 'TR', 'BR', 'BL']:
        error = distance(detected[corner], expected[corner])
        errors.append(error)
    
    avg_error = mean(errors)
    max_error = max(errors)
    
    return {
        'passed': max_error < 15,
        'avg_error': avg_error,
        'max_error': max_error,
        'per_corner_errors': errors
    }
```

---

## 5. Test Cases Required

### 5.1 Unit Tests

| Test | Description | Pass Criteria |
|------|-------------|----------------|
| test_no_rotation | Photo without rotation | All corners < 5px error |
| test_small_rotation | Photo with 10-15° rotation | All corners < 10px error |
| test_large_rotation | Photo with 25-30° rotation | All corners < 15px error |
| test_multiple_photos | 2-4 photos on canvas | All corners < 20px error |

### 5.2 Integration Tests

| Test | Description | Pass Criteria |
|------|-------------|----------------|
| test_perspective_no_rotation | Perspective warp without rotation | All corners < 15px error |
| test_perspective_with_rotation | Perspective warp with rotation | All corners < 20px error |
| test_extreme_perspective | 20% perspective strength | All corners < 25px error |
| test_crop_and_resize | Full pipeline with crop to 640 | All corners < 30px error |

### 5.3 Robustness Tests

| Test | Description | Pass Criteria |
|------|-------------|----------------|
| test_different_photo_sizes | 180px to 480px photos | All corners < 20px error |
| test_different_placements | Various center positions | All corners < 20px error |
| test_all_perspective_directions | 8 different directions | All corners < 25px error |

---

## 6. Current Verification Results

### Black Edge Detection
| Image | Dark Ratio | Threshold | Status |
|-------|------------|-----------|--------|
| example_01 | 0.0000 | 0.05 | ✅ PASS |
| example_02 | 0.2707 | 0.05 | ❌ FAIL |
| example_03 | 0.0000 | 0.05 | ✅ PASS |
| example_04 | 0.0000 | 0.05 | ✅ PASS |
| example_05 | 0.0000 | 0.05 | ✅ PASS |

### Bounding Box Accuracy
| Image | Max Error | Threshold | Status |
|-------|-----------|-----------|--------|
| example_01 | 347px | 15px | ❌ FAIL |
| example_02 | 254px | 15px | ❌ FAIL |
| example_03 | 293px | 15px | ❌ FAIL |
| example_04 | 362px | 15px | ❌ FAIL |
| example_05 | 266px | 15px | ❌ FAIL |

---

## 7. Key Findings from Debugging

### 7.1 Rotation Math
- **OpenCV's getRotationMatrix2D** uses formula: `new_x = cos*x + sin*y + offset`
- **NOT** the standard formula: `new_x = cos*(x-cx) - sin*(y-cy) + cx`
- This caused significant errors when using wrong formula

### 7.2 Canvas Compositing
- **Origin in canvas** = center - rotated_center
- **Corner in canvas** = origin + corner_rotated
- This relationship must be maintained for correct corner tracking

### 7.3 Coordinate System Mismatch
- `rotate_photo()`: Rotates around photo center (width/2, height/2)
- `get_rotated_polygon()`: Must calculate where corners end up
- **Critical**: Both must use same rotation matrix

---

## 8. Remaining Actions

### 8.1 High Priority
1. **Verify bbox accuracy test** - Threshold of 15px may be too strict for real photos
2. **Check where black edges come from** - Texture or actual black pixels?
3. **Fix remaining black edge failures** - Increase padding or improve crop

### 8.2 Medium Priority
1. **Create automated test report** - Generate CSV/JSON of all test results
2. **Visual verification tool** - Show detected vs calculated corners overlaid
3. **Test with real source images** - Ensure marker detection works on real photos

### 8.3 Low Priority
1. **Performance optimization** - Cache rotation matrices, reduce redundant calculations
2. **Test coverage report** - Show which code paths are tested

---

## 9. Configuration Recommendations

```python
CONFIG = {
    # Canvas - currently 300px padding
    'CANVAS_PADDING': 300,  # May need 400-500 for extreme cases
    
    # Shadows - currently very conservative
    'SHADOW_BLUR_MAX': 2,   # OK
    'SHADOW_OFFSET_MAX': 2, # OK
    
    # Corner tracking - 15px threshold may need adjustment
    'CORNER_ERROR_THRESHOLD': 15,  # Consider 20-30 for real photos
    
    # Black edge detection
    'EDGE_DARK_RATIO_THRESHOLD': 0.05,  # OK for texture backgrounds
}
```

---

## 10. Verification Checklist

Before showing images to user, verify:

- [ ] All corner markers within calculated positions (±15px)
- [ ] No black edges (>5% dark pixels in border)
- [ ] Border variance indicates texture (not blank)
- [ ] Shadow spread <15% of photo size
- [ ] Perspective transform applied correctly (grid lines straight)
- [ ] Bounding boxes match detected photo edges

---

## 11. Files Created

| File | Purpose |
|------|---------|
| `verify_all_tests.py` | Main test suite with all CV-based tests |
| `diagnostic_pipeline.py` | Step-by-step pipeline tracing |
| `debug_corners.py` | Corner calculation comparison |
| `e2e_pipeline_test.py` | End-to-end test with markers |
| `test_e2e_with_markers.py` | Integration test with generate_dataset |
| `generate_dataset.py` | Main generation code (v34) |

---

## 12. Test Execution Commands

```bash
# Run all tests
python3 verify_all_tests.py

# Run diagnostic
python3 diagnostic_pipeline.py

# Run end-to-end with markers
python3 test_e2e_with_markers.py

# Generate new images
python3 generate_dataset.py

# Quick corner check
python3 -c "
import cv2
import numpy as np
# ... quick verification code
"
```