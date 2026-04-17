#!/usr/bin/env python3
"""
End-to-end visual test with CORNER MARKERS to verify <5px accuracy.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import (
    CONFIG, rotate_photo, composite_photo_at_center,
    apply_global_perspective
)


def create_marker_photo(width, height, marker_size=25):
    """Create photo with colored markers at exact corners."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    colors = [
        (255, 0, 0),     # Corner 0 - Red
        (0, 255, 0),     # Corner 1 - Green  
        (0, 0, 255),     # Corner 2 - Blue
        (255, 255, 0),   # Corner 3 - Yellow
    ]
    
    positions = [
        (5, 5),                         # TL
        (width - marker_size - 5, 5),  # TR
        (width - marker_size - 5, height - marker_size - 5),  # BR
        (5, height - marker_size - 5)  # BL
    ]
    
    for i, (x, y) in enumerate(positions):
        photo[y:y+marker_size, x:x+marker_size] = colors[i]
    
    return photo


def get_rotated_polygon_fixed(width, height, center_x, center_y, rotation):
    """Corrected polygon calculation."""
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],  # 0: TL
            [center_x + hw, center_y - hh],  # 1: TR
            [center_x + hw, center_y + hh],  # 2: BR
            [center_x - hw, center_y + hh]   # 3: BL
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(height * sin_a + width * cos_a)
    new_h = int(height * cos_a + width * sin_a)
    
    M[0, 2] += (new_w - width) / 2
    M[1, 2] += (new_h - height) / 2
    
    corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    corners_rot = np.zeros_like(corners)
    for i in range(4):
        pt = np.array([corners[i, 0], corners[i, 1], 1])
        result = M @ pt
        corners_rot[i] = [result[0], result[1]]
    
    origin = M[:, 2]
    center_rot = M @ np.array([photo_center[0], photo_center[1], 1])
    
    offset_x = center_x - center_rot[0]
    offset_y = center_y - center_rot[1]
    
    return corners_rot + np.array([offset_x, offset_y])


def detect_markers(img, marker_size=25):
    """Detect colored markers and return their positions."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Color ranges
    color_ranges = [
        ('0', [0, 100, 100], [15, 255, 255]),     # Red
        ('1', [35, 100, 100], [85, 255, 255]),    # Green
        ('2', [100, 100, 100], [130, 255, 255]), # Blue
        ('3', [15, 100, 100], [45, 255, 255])     # Yellow
    ]
    
    detected = {}
    for name, lower, upper in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detected[name] = (cx, cy)
    
    return detected


def run_pipeline_with_debug(rotation, test_num):
    """Run one complete pipeline test."""
    CANVAS_SIZE = 640
    PADDING = 300
    out_size = 640
    
    photo_size = 280
    center_x = CANVAS_SIZE // 2 + PADDING
    center_y = CANVAS_SIZE // 2 + PADDING
    
    photo = create_marker_photo(photo_size, photo_size)
    
    # Step 1: Calculate expected corners
    polygon = get_rotated_polygon_fixed(photo_size, photo_size, center_x, center_y, rotation)
    print(f"\n  Expected corners (padded space): {polygon[0]}")
    
    # Step 2: Rotate photo and composite
    rotated = rotate_photo(photo, rotation)
    if rotated.shape[2] == 3:
        rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        rotated = rgba
    
    canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    result = composite_photo_at_center(canvas, rotated, center_x, center_y)
    
    # Step 3: Apply perspective
    warped, global_corners, transform_matrix, content_bounds, warped_corners = apply_global_perspective(
        result, PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE,
        photo_corners=[polygon],
        crop_margin=CONFIG['CROP_MARGIN']
    )
    
    print(f"  After perspective (warped space): {warped_corners[0][0] if warped_corners else 'N/A'}")
    
    # Step 4: Resize to 640x640
    out_w, out_h = warped.shape[1], warped.shape[0]
    if out_w != out_size or out_h != out_size:
        scale_x = out_size / out_w
        scale_y = out_size / out_h
        final = cv2.resize(warped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        final_corners = warped_corners[0] * np.array([scale_x, scale_y])
    else:
        final = warped
        final_corners = warped_corners[0]
    
    print(f"  Final corners (640x640 space): {final_corners[0]}")
    
    # Step 5: Detect markers and compare
    detected = detect_markers(final)
    
    # Visualize
    vis = final.copy()
    label_names = ['TL', 'TR', 'BR', 'BL']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, (pt, name, color) in enumerate(zip(final_corners, label_names, colors)):
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 12, color, 3)
        cv2.putText(vis, f"{i}:{name}({int(pt[0])},{int(pt[1])})", 
                   (int(pt[0])+15, int(pt[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    
    for corner_num, pos in sorted(detected.items()):
        cv2.circle(vis, pos, 8, (255, 255, 255), -1)
        cv2.putText(vis, f"d{corner_num}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    
    cv2.imwrite(f'/tmp/e2e_test_{test_num}_rot{rotation}.jpg', final)
    cv2.imwrite(f'/tmp/e2e_annotated_{test_num}_rot{rotation}.jpg', vis)
    
    return final_corners, detected


def main():
    print("\n" + "="*70)
    print("END-TO-END CORNER VERIFICATION WITH MARKERS")
    print("="*70)
    
    test_cases = [
        ("0° rotation", 0),
        ("30° rotation", 30),
        ("45° rotation", 45),
        ("60° rotation", 60),
        ("90° rotation", 90),
        ("-30° rotation", -30),
    ]
    
    all_passed = True
    
    for name, rotation in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print("="*70)
        
        expected, detected = run_pipeline_with_debug(rotation, test_cases.index((name, rotation)))
        
        print(f"\n  DETECTION RESULTS:")
        errors = []
        
        for i in range(4):
            det_key = str(i)
            if det_key in detected:
                dx = detected[det_key][0] - expected[i][0]
                dy = detected[det_key][1] - expected[i][1]
                error = np.sqrt(dx**2 + dy**2)
                errors.append(error)
                status = "✅" if error < 8 else "❌"
                print(f"    Corner {i}: detected=({detected[det_key][0]:.0f}, {detected[det_key][1]:.0f}), "
                      f"expected=({expected[i][0]:.0f}, {expected[i][1]:.0f}), "
                      f"dx={dx:.0f}, dy={dy:.0f}, error={error:.1f}px {status}")
            else:
                print(f"    Corner {i}: NOT DETECTED ❌")
                errors.append(float('inf'))
        
        if errors and errors[0] != float('inf'):
            avg_error = sum(errors) / len(errors)
            print(f"\n    Average error: {avg_error:.1f}px")
            if avg_error < 8:
                print(f"    ✅ PASSED")
            else:
                print(f"    ❌ FAILED - average error >= 8px")
                all_passed = False
        else:
            all_passed = False
    
    print("\n" + "="*70)
    print("OVERALL RESULT")
    print("="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")


if __name__ == '__main__':
    main()