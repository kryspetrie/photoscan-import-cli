#!/usr/bin/env python3
"""
Corner Verification via Color Detection - Complete Pipeline Test

This script verifies that corner coordinates are correctly calculated by:
1. Creating a test photo with 4 DISTINCT colored corners (Blue, Red, Green, Yellow)
2. Processing through entire pipeline (effects, rotation, perspective, crop, resize)
3. Detecting ACTUAL pixel locations of those colors in final image
4. Comparing to CALCULATED corner coordinates
5. FAIL if >5px difference
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import generate_dataset as gd

# =============================================================================
# Test Configuration
# =============================================================================
CANVAS_SIZE = 640
PADDING = 100
WORKING_SIZE = CANVAS_SIZE + 2 * PADDING
TEST_SEED = 42

# Colors for each corner (in BGR for OpenCV)
CORNER_COLORS = {
    'LL': (255, 0, 0),      # Blue - Lower-Left
    'UL': (0, 0, 255),      # Red - Upper-Left  
    'UR': (0, 255, 0),      # Green - Upper-Right
    'LR': (0, 255, 255),    # Yellow - Lower-Right
}

def create_test_photo(size=300):
    """Create a test photo with 4 colored corners."""
    # Create gradient background (easy to see)
    photo = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        photo[:, i] = [i * 255 // size, 100, 255 - i * 255 // size]
    
    margin = 20
    radius = 35
    
    # LL (Lower-Left) - Blue
    cv2.circle(photo, (margin, size - margin), radius, (255, 0, 0), -1)
    
    # UL (Upper-Left) - Red
    cv2.circle(photo, (margin, margin), radius, (0, 0, 255), -1)
    
    # UR (Upper-Right) - Green
    cv2.circle(photo, (size - margin, margin), radius, (0, 255, 0), -1)
    
    # LR (Lower-Right) - Yellow
    cv2.circle(photo, (size - margin, size - margin), radius, (0, 255, 255), -1)
    
    return photo

def detect_color_position(img, bgr_color, tolerance=40):
    """Find center of a colored region."""
    # Convert to BGR if needed (handle BGRA)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    b, g, r = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])
    tol = tolerance
    
    # Create mask for this color
    lower = np.array([max(0, b-tol), max(0, g-tol), max(0, r-tol)])
    upper = np.array([min(255, b+tol), min(255, g+tol), min(255, r+tol)])
    
    mask = cv2.inRange(img, lower, upper)
    
    # Clean up mask
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find largest contour
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    
    if M['m00'] > 100:  # Must be substantial area
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    
    return None

def test_rotation_only():
    """Test rotation without perspective warp."""
    print("\n" + "="*60)
    print("TEST 1: Rotation Only (No Perspective)")
    print("="*60)
    
    np.random.seed(TEST_SEED)
    
    # Create test photo
    photo_orig = create_test_photo(300)
    cv2.imwrite('test1_original.jpg', photo_orig)
    
    # Apply effects
    photo = gd.fast_photo_manipulation(photo_orig.copy())
    photo = gd.fast_glare(photo)
    photo = gd.add_rgba_alpha(photo)
    photo = gd.blur_alpha_edges(photo)
    
    # Rotate by fixed amount
    rotation = 25  # degrees
    photo_rotated = gd.rotate_photo(photo, rotation)
    
    cv2.imwrite('test1_after_rotation.jpg', photo_rotated)
    
    # Use get_rotated_polygon for CORRECT corner calculation
    # This now includes the offset adjustment
    cx_orig, cy_orig = 300 // 2, 300 // 2
    expected_corners = gd.get_rotated_polygon(300, 300, cx_orig, cy_orig, rotation)
    
    print(f"Expected corners after rotation (from get_rotated_polygon):")
    for i, (label, expected) in enumerate(zip(['LL', 'UL', 'UR', 'LR'], expected_corners)):
        print(f"  {label}: ({expected[0]:.1f}, {expected[1]:.1f})")
    
    # Detect actual positions
    detected = {}
    for label, color in CORNER_COLORS.items():
        pos = detect_color_position(photo_rotated, color)
        detected[label] = pos
        if pos:
            print(f"  Detected {label}: ({pos[0]}, {pos[1]})")
        else:
            print(f"  Detected {label}: NOT FOUND")
    
    # Compare
    print("\nComparison:")
    max_error = 0
    for i, label in enumerate(['LL', 'UL', 'UR', 'LR']):
        if detected[label] is None:
            print(f"  {label}: FAILED - could not detect")
            continue
        
        calc = expected_corners[i]
        det = detected[label]
        error = np.sqrt((calc[0] - det[0])**2 + (calc[1] - det[1])**2)
        max_error = max(max_error, error)
        
        status = "✓" if error <= 5 else "✗"
        print(f"  {label}: calc=({calc[0]:.1f},{calc[1]:.1f}) det=({det[0]},{det[1]}) err={error:.1f}px {status}")
    
    # Draw on image
    vis = photo_rotated.copy()
    if vis.shape[2] == 4:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR)
    for label, pos in detected.items():
        if pos:
            color = CORNER_COLORS[label]
            cv2.circle(vis, pos, 20, color, -1)
            cv2.putText(vis, label, (pos[0]+25, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite('test1_result.jpg', vis)
    
    return max_error <= 5

def test_with_composite():
    """Test with full composite and perspective warp."""
    print("\n" + "="*60)
    print("TEST 2: Full Composite + Perspective Warp")
    print("="*60)
    
    np.random.seed(TEST_SEED)
    
    # Create working canvas with padding
    canvas = np.ones((WORKING_SIZE, WORKING_SIZE, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0  # Transparent
    
    # Create and prepare test photo
    photo_orig = create_test_photo(300)
    cv2.imwrite('test2_original.jpg', photo_orig)
    
    photo = gd.fast_photo_manipulation(photo_orig.copy())
    photo = gd.fast_glare(photo)
    photo = gd.add_rgba_alpha(photo)
    photo = gd.blur_alpha_edges(photo)
    
    rotation = 15  # degrees
    photo = gd.rotate_photo(photo, rotation)
    
    cv2.imwrite('test2_pre_composite.jpg', photo)
    
    # Place at center of working canvas
    center_x = WORKING_SIZE // 2
    center_y = WORKING_SIZE // 2
    
    canvas = gd.composite_photo_at_center(canvas, photo, center_x, center_y)
    
    cv2.imwrite('test2_pre_warp.jpg', cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR))
    
    # Use get_rotated_polygon for CORRECT corner calculation (includes offset adjustment)
    # Original photo dimensions
    orig_w, orig_h = 300, 300
    expected_canvas_corners = gd.get_rotated_polygon(orig_w, orig_h, center_x, center_y, rotation)
    
    print(f"Expected corners in canvas (before warp):")
    for i, label in enumerate(['LL', 'UL', 'UR', 'LR']):
        print(f"  {label}: ({expected_canvas_corners[i][0]:.1f}, {expected_canvas_corners[i][1]:.1f})")
    
    # Apply global perspective warp
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    
    warped, global_corners, M, content_bounds, warped_photo_corners = gd.apply_global_perspective(
        canvas_bgr, CANVAS_SIZE, CANVAS_SIZE,
        photo_corners=[expected_canvas_corners],
        crop_margin=80
    )
    
    print(f"\nWarped image size: {warped.shape}")
    print(f"Content bounds: {content_bounds}")
    
    # Resize to 640x640 if needed
    scale_x, scale_y = 1, 1
    if warped.shape[1] != CANVAS_SIZE or warped.shape[0] != CANVAS_SIZE:
        scale_x = CANVAS_SIZE / warped.shape[1]
        scale_y = CANVAS_SIZE / warped.shape[0]
        warped = cv2.resize(warped, (CANVAS_SIZE, CANVAS_SIZE))
        
        if warped_photo_corners:
            warped_photo_corners = [
                np.array([[p[0] * scale_x, p[1] * scale_y] for p in corners])
                for corners in warped_photo_corners
            ]
    
    cv2.imwrite('test2_post_warp.jpg', warped)
    
    # Get calculated corners from pipeline
    if warped_photo_corners and len(warped_photo_corners) > 0:
        calculated = warped_photo_corners[0]
        
        print(f"\nCalculated corners after warp:")
        for i, label in enumerate(['LL', 'UL', 'UR', 'LR']):
            print(f"  {label}: ({calculated[i][0]:.1f}, {calculated[i][1]:.1f})")
    
    # Detect actual positions in warped image
    detected = {}
    for label, color in CORNER_COLORS.items():
        pos = detect_color_position(warped, color, tolerance=50)
        detected[label] = pos
        if pos:
            print(f"  Detected {label}: ({pos[0]}, {pos[1]})")
        else:
            print(f"  Detected {label}: NOT FOUND")
    
    # Compare calculated vs detected
    print("\n" + "="*60)
    print("COMPARISON: Calculated vs Detected")
    print("="*60)
    
    if warped_photo_corners and len(warped_photo_corners) > 0:
        calculated = warped_photo_corners[0]
        max_error = 0
        
        for i, label in enumerate(['LL', 'UL', 'UR', 'LR']):
            if detected[label] is None:
                print(f"  {label}: ✗ FAILED - could not detect color in image")
                continue
            
            calc = calculated[i]
            det = detected[label]
            error = np.sqrt((calc[0] - det[0])**2 + (calc[1] - det[1])**2)
            max_error = max(max_error, error)
            
            status = "✓ PASS" if error <= 5 else "✗ FAIL"
            print(f"  {label}:")
            print(f"    Calculated: ({calc[0]:.1f}, {calc[1]:.1f})")
            print(f"    Detected:   ({det[0]}, {det[1]})")
            print(f"    Error:      {error:.1f}px {status}")
        
        # Draw visualization
        vis = warped.copy()
        for label, pos in detected.items():
            if pos:
                color = CORNER_COLORS[label]
                cv2.circle(vis, pos, 25, color, -1)
                cv2.circle(vis, pos, 25, (255, 255, 255), 5)
                cv2.putText(vis, label, (pos[0]+30, pos[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        
        # Draw calculated corners in different color
        if warped_photo_corners:
            for i, pt in enumerate(calculated):
                cv2.drawMarker(vis, (int(pt[0]), int(pt[1])), (255, 255, 255), 
                              cv2.MARKER_DIAMOND, 20, 3)
        
        cv2.imwrite('test2_result.jpg', vis)
        
        print(f"\n{'='*60}")
        if max_error <= 5:
            print("✅ RESULT: ALL TESTS PASSED - Corners within 5px tolerance")
        else:
            print(f"❌ RESULT: TESTS FAILED - Max error {max_error:.1f}px (>5px threshold)")
        print(f"{'='*60}")
        
        return max_error <= 5
    else:
        print("❌ No warped corners returned from pipeline!")
        return False

if __name__ == "__main__":
    print("="*60)
    print("CORNER VERIFICATION - COLOR DETECTION METHOD")
    print("="*60)
    print("\nMethodology:")
    print("1. Create photo with 4 colored corner markers")
    print("2. Process through pipeline")
    print("3. Detect actual color positions in final image")
    print("4. Compare to calculated corner coordinates")
    print("5. FAIL if >5px difference")
    
    test1_pass = test_rotation_only()
    test2_pass = test_with_composite()
    
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    print(f"Test 1 (Rotation Only): {'PASS ✓' if test1_pass else 'FAIL ✗'}")
    print(f"Test 2 (Full Pipeline): {'PASS ✓' if test2_pass else 'FAIL ✗'}")
    
    if test1_pass and test2_pass:
        print("\n✅ ALL TESTS PASSED - Corner tracking is correct!")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED - Corner tracking has errors!")
        sys.exit(1)
