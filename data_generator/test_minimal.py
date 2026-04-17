#!/usr/bin/env python3
"""
MINIMAL corner tracking test - single corner marker only.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import CONFIG, rotate_photo, composite_photo_at_center, apply_global_perspective


def get_rotated_polygon_v2(width, height, center_x, center_y, rotation):
    """Fixed polygon calculation."""
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],
            [center_x + hw, center_y - hh],
            [center_x + hw, center_y + hh],
            [center_x - hw, center_y + hh]
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(height * sin_a + width * cos_a)
    new_h = int(height * cos_a + width * sin_a)
    
    M[0, 2] += (new_w - width) / 2
    M[1, 2] += (new_h - height) / 2
    
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    
    corners_rot = np.zeros_like(corners)
    for i in range(4):
        pt = np.array([corners[i, 0], corners[i, 1], 1])
        result = M @ pt
        corners_rot[i] = [result[0], result[1]]
    
    center_rot = M @ np.array([photo_center[0], photo_center[1], 1])
    
    return corners_rot + np.array([center_x - center_rot[0], center_y - center_rot[1]])


def create_simple_photo(width, height):
    """Photo with single red marker at top-left corner (0,0) position."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Single bright red marker at (0,0) in PHOTO space
    marker = 25
    photo[0:marker, 0:marker] = [0, 0, 255]  # Red at top-left
    
    return photo


def run_minimal_test():
    """Run with 0° rotation to isolate the issue."""
    CANVAS_SIZE = 640
    PADDING = 300
    OUT_SIZE = 640
    
    photo_width, photo_height = 300, 200
    center_x = CANVAS_SIZE // 2 + PADDING  # 620
    center_y = CANVAS_SIZE // 2 + PADDING  # 620
    rotation = 0
    
    print(f"\n{'='*70}")
    print("MINIMAL TEST: No rotation")
    print(f"{'='*70}")
    print(f"  Photo: {photo_width}x{photo_height}")
    print(f"  Center in padded space: ({center_x}, {center_y})")
    
    photo = create_simple_photo(photo_width, photo_height)
    
    # Step 1: Calculate expected polygon
    polygon = get_rotated_polygon_v2(photo_width, photo_height, center_x, center_y, rotation)
    print(f"\n  Expected polygon (padded space):")
    print(f"    Corner 0 (TL): {polygon[0]}")
    print(f"    Corner 1 (TR): {polygon[1]}")
    print(f"    Corner 2 (BR): {polygon[2]}")
    print(f"    Corner 3 (BL): {polygon[3]}")
    
    # Expected: for 300x200 at (620,620), corners should be:
    # 0: (470, 520), 1: (770, 520), 2: (770, 720), 3: (470, 720)
    print(f"\n  Manual calculation:")
    print(f"    TL should be: ({center_x - 150}, {center_y - 100}) = ({center_x - 150}, {center_y - 100})")
    
    # Step 2: Rotate and composite
    rotated = rotate_photo(photo, rotation)
    if rotated.shape[2] == 3:
        rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        rotated = rgba
    
    canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    result = composite_photo_at_center(canvas, rotated, center_x, center_y)
    
    # Save before perspective
    vis1 = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
    cv2.polylines(vis1, [polygon.astype(np.int32)], True, (0, 255, 0), 2)
    for i, pt in enumerate(polygon):
        cv2.circle(vis1, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    cv2.imwrite('/tmp/step1_before_perspective.jpg', vis1)
    print(f"\n  Saved: step1_before_perspective.jpg")
    
    # Step 3: Apply perspective
    warped, global_corners, transform_matrix, content_bounds, warped_corners = apply_global_perspective(
        result, PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE,
        photo_corners=[polygon],
        crop_margin=CONFIG['CROP_MARGIN']
    )
    
    out_w, out_h = warped.shape[1], warped.shape[0]
    print(f"\n  After perspective: {out_w}x{out_h}")
    print(f"  Warped corners: {warped_corners[0]}")
    
    # Step 4: Resize to 640x640
    if out_w != OUT_SIZE or out_h != OUT_SIZE:
        scale_x = OUT_SIZE / out_w
        scale_y = OUT_SIZE / out_h
        final = cv2.resize(warped, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
        final_corners = warped_corners[0] * np.array([scale_x, scale_y])
    else:
        final = warped
        final_corners = warped_corners[0]
    
    print(f"\n  Final corners (640x640 space):")
    print(f"    {final_corners}")
    
    # Step 5: Detect red marker in final image
    hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n  Marker detection:")
    detected_positions = []
    if contours:
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detected_positions.append((cx, cy))
                    print(f"    Found: ({cx}, {cy})")
    else:
        print(f"    No markers found")
    
    # Step 6: Compare
    print(f"\n  Comparison:")
    if detected_positions:
        detected = detected_positions[0]
        expected = final_corners[0]  # Corner 0 should be at the marker
        error = np.sqrt((detected[0] - expected[0])**2 + (detected[1] - expected[1])**2)
        print(f"    Detected: {detected}")
        print(f"    Expected (corner 0): {expected}")
        print(f"    Error: {error:.1f}px")
    
    # Visualize final
    vis2 = cv2.cvtColor(final, cv2.COLOR_BGRA2BGR)
    for i, (pt, color) in enumerate(zip(final_corners, [(255,0,0), (0,255,0), (0,0,255), (255,255,0)])):
        cv2.circle(vis2, (int(pt[0]), int(pt[1])), 10, color, -1)
        cv2.putText(vis2, str(i), (int(pt[0])+12, int(pt[1])-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if detected_positions:
        cv2.circle(vis2, detected_positions[0], 15, (255, 255, 255), 2)
    
    cv2.imwrite('/tmp/step2_final.jpg', vis2)
    print(f"\n  Saved: step2_final.jpg")


def test_with_rotation():
    """Test with various rotations."""
    print(f"\n{'='*70}")
    print("ROTATION TESTS")
    print("="*70)
    
    CANVAS_SIZE = 640
    PADDING = 300
    OUT_SIZE = 640
    
    photo_width, photo_height = 300, 200
    
    for angle in [0, 30, 45, 60, 90]:
        center_x = CANVAS_SIZE // 2 + PADDING
        center_y = CANVAS_SIZE // 2 + PADDING
        
        photo = create_simple_photo(photo_width, photo_height)
        polygon = get_rotated_polygon_v2(photo_width, photo_height, center_x, center_y, angle)
        
        rotated = rotate_photo(photo, angle)
        if rotated.shape[2] == 3:
            rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = 255
            rotated = rgba
        
        canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
        canvas[:, :, 3] = 0
        result = composite_photo_at_center(canvas, rotated, center_x, center_y)
        
        warped, _, _, _, warped_corners = apply_global_perspective(
            result, PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE,
            photo_corners=[polygon],
            crop_margin=CONFIG['CROP_MARGIN']
        )
        
        out_w, out_h = warped.shape[1], warped.shape[0]
        scale_x = OUT_SIZE / out_w if out_w > 0 else 1
        scale_y = OUT_SIZE / out_h if out_h > 0 else 1
        
        final = cv2.resize(warped, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
        final_corners = warped_corners[0] * np.array([scale_x, scale_y])
        
        # Detect marker
        hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    detected = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
        
        expected = final_corners[0]
        error = np.sqrt((detected[0]-expected[0])**2 + (detected[1]-expected[1])**2) if detected else float('inf')
        status = "✅" if error < 8 else "❌"
        print(f"  {angle:3d}°: detected={detected}, expected=({expected[0]:.0f},{expected[1]:.0f}), error={error:.1f}px {status}")


if __name__ == '__main__':
    run_minimal_test()
    test_with_rotation()