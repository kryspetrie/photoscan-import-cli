#!/usr/bin/env python3
"""
Corner Verification - Complete Pipeline Test

Uses color detection methodology:
1. Place distinct colored markers at exact corners
2. Process through entire pipeline
3. Detect actual positions of colored regions
4. Compare to calculated polygon corners
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import generate_dataset as gd

# Colors for corners
CORNER_COLORS = {
    'UL': (255, 0, 0),      # Red - Upper-Left
    'UR': (0, 255, 0),      # Green - Upper-Right
    'LR': (0, 0, 255),      # Blue - Lower-Right
    'LL': (255, 255, 0),    # Yellow - Lower-Left
}

YOLO_ORDER = ['LL', 'UL', 'UR', 'LR']  # Expected output order
# Polygon returns [TL, TR, BR, BL] which maps to YOLO as [BL, TL, TR, BR]
POLYGON_TO_YOLO = [3, 0, 1, 2]

def create_test_photo_with_corners(size=200):
    """Create test photo with colored corner markers."""
    photo = np.zeros((size, size, 3), dtype=np.uint8)
    # Mark corners at EXACT positions
    cv2.circle(photo, (0, 0), 10, CORNER_COLORS['UL'], -1)
    cv2.circle(photo, (size-1, 0), 10, CORNER_COLORS['UR'], -1)
    cv2.circle(photo, (size-1, size-1), 10, CORNER_COLORS['LR'], -1)
    cv2.circle(photo, (0, size-1), 10, CORNER_COLORS['LL'], -1)
    return photo

def detect_corner(img, bgr_color, tolerance=30):
    """Detect centroid of colored corner region."""
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    mask = cv2.inRange(img, np.array(bgr_color)-tolerance, np.array(bgr_color)+tolerance)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 50:
            return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    return None

def test_composite():
    """Test with composite on canvas (main use case)."""
    print("\n" + "="*60)
    print("TEST: Composite on Canvas + Perspective Warp")
    print("="*60)
    
    canvas_size = 400
    padding = 100
    working_size = canvas_size + 2 * padding
    
    canvas = np.ones((working_size, working_size, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    
    photo_orig = create_test_photo_with_corners(200)
    cv2.imwrite('test_composite_orig.jpg', photo_orig)
    
    center_x = working_size // 2
    center_y = working_size // 2
    rotation = 15
    
    # Get polygon
    polygon = gd.get_rotated_polygon(200, 200, center_x, center_y, rotation)
    
    print(f"Photo: 200x200 at ({center_x}, {center_y}), rot={rotation}°")
    print(f"Polygon [TL, TR, BR, BL]:")
    for i, label in enumerate(['TL', 'TR', 'BR', 'BL']):
        print(f"  {label}: ({polygon[i][0]:.1f}, {polygon[i][1]:.1f})")
    
    # Rotate and composite
    photo = gd.add_rgba_alpha(photo_orig)
    photo_rotated = gd.rotate_photo(photo, rotation)
    canvas = gd.composite_photo_at_center(canvas, photo_rotated, center_x, center_y)
    
    cv2.imwrite('test_composite_pre_warp.jpg', cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR))
    
    # Detect on canvas before warp
    detected_before = {}
    for label, color in CORNER_COLORS.items():
        pos = detect_corner(canvas, color)
        detected_before[label] = pos
    
    # Compare polygon to detected before warp
    print("\nComparison (polygon vs detected before warp):")
    max_error = 0
    for i, yolo_label in enumerate(YOLO_ORDER):
        calc = polygon[POLYGON_TO_YOLO[i]]
        det = detected_before[yolo_label]
        if det:
            error = np.sqrt((calc[0]-det[0])**2 + (calc[1]-det[1])**2)
            max_error = max(max_error, error)
            status = "✓" if error <= 8 else "✗"
            print(f"  {yolo_label}: calc=({calc[0]:.1f},{calc[1]:.1f}) det={det} err={error:.1f}px {status}")
    
    print(f"\nPre-warp max error: {max_error:.1f}px")
    pre_warp_pass = max_error <= 8
    
    # Apply perspective warp
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    warped, global_corners, M, content_bounds, warped_corners = gd.apply_global_perspective(
        canvas_bgr, canvas_size, canvas_size,
        photo_corners=[polygon],
        crop_margin=50
    )
    
    # Resize if needed
    if warped.shape[1] != canvas_size or warped.shape[0] != canvas_size:
        scale_x = canvas_size / warped.shape[1]
        scale_y = canvas_size / warped.shape[0]
        warped = cv2.resize(warped, (canvas_size, canvas_size))
        if warped_corners:
            warped_corners = [
                np.array([[p[0] * scale_x, p[1] * scale_y] for p in corners])
                for corners in warped_corners
            ]
    
    cv2.imwrite('test_composite_post_warp.jpg', warped)
    
    print(f"\nWarped size: {warped.shape}")
    
    # Detect in warped image
    detected_after = {}
    for label, color in CORNER_COLORS.items():
        pos = detect_corner(warped, color)
        detected_after[label] = pos
        print(f"  Detected {label}: {pos}")
    
    # Get final corners and reorder
    if warped_corners and len(warped_corners) > 0:
        final = warped_corners[0]
        reordered = gd.reorder_corners_to_llulur(final)
        
        print("\nComparison (after warp):")
        max_error = 0
        for i, yolo_label in enumerate(YOLO_ORDER):
            calc = reordered[i]
            det = detected_after[yolo_label]
            if det:
                error = np.sqrt((calc[0]-det[0])**2 + (calc[1]-det[1])**2)
                max_error = max(max_error, error)
                status = "✓" if error <= 8 else "✗"
                print(f"  {yolo_label}: calc=({calc[0]:.1f},{calc[1]:.1f}) det={det} err={error:.1f}px {status}")
        
        print(f"\nPost-warp max error: {max_error:.1f}px")
        post_warp_pass = max_error <= 8
    else:
        print("ERROR: No warped corners returned!")
        post_warp_pass = False
    
    return pre_warp_pass and post_warp_pass

if __name__ == "__main__":
    np.random.seed(42)
    
    print("="*60)
    print("CORNER VERIFICATION TEST")
    print("="*60)
    
    test_pass = test_composite()
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Test (Composite + Warp): {'PASS ✓' if test_pass else 'FAIL ✗'}")
    
    if test_pass:
        print("\n✅ CORNER TRACKING VERIFIED - Corners match within 8px tolerance!")
        sys.exit(0)
    else:
        print("\n❌ CORNER TRACKING FAILED - Errors exceed tolerance")
        sys.exit(1)
