#!/usr/bin/env python3
"""
Test WITHOUT perspective to isolate rotation + resize error.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import rotate_photo, composite_photo_at_center


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
    """Photo with single bright red marker at top-left corner."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 200
    marker = 25
    photo[0:marker, 0:marker] = [0, 0, 255]
    return photo


def test_no_perspective():
    """Test rotation + resize WITHOUT perspective warp."""
    print("\n" + "="*70)
    print("TEST: Rotation + Resize ONLY (NO Perspective)")
    print("="*70)
    
    CANVAS_SIZE = 640
    PADDING = 300
    OUT_SIZE = 640
    
    photo_width, photo_height = 300, 200
    
    print(f"\n  Photo: {photo_width}x{photo_height}")
    print(f"  Canvas: {CANVAS_SIZE}x{CANVAS_SIZE} + {PADDING}px padding")
    print(f"  Output: {OUT_SIZE}x{OUT_SIZE}")
    
    for angle in [0, 30, 45, 60, 90, -30]:
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
        
        # NO PERSPECTIVE - just resize
        result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        out_w, out_h = result_bgr.shape[1], result_bgr.shape[0]
        
        # The corners are in the padded canvas space (0 to CANVAS_SIZE+2*PADDING)
        # Need to subtract PADDING to get into canvas space (0 to CANVAS_SIZE)
        canvas_corners = polygon - np.array([PADDING, PADDING])
        
        # Now resize
        scale = OUT_SIZE / CANVAS_SIZE
        scaled_corners = canvas_corners * scale
        
        final = cv2.resize(result_bgr, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
        
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
        
        expected = scaled_corners[0]
        error = np.sqrt((detected[0]-expected[0])**2 + (detected[1]-expected[1])**2) if detected else float('inf')
        status = "✅" if error < 5 else "❌"
        print(f"  {angle:3d}°: detected={detected}, expected=({expected[0]:.0f},{expected[1]:.0f}), error={error:.1f}px {status}")


def test_with_lanczos_resize():
    """Test with LANCZOS4 resize for better quality."""
    print("\n" + "="*70)
    print("TEST: With LANCZOS4 Resize (Better Quality)")
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
        
        # Simulate perspective by just taking center crop
        # (For testing, we want to see if resize quality matters)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        
        scale = OUT_SIZE / CANVAS_SIZE
        canvas_corners = polygon - np.array([PADDING, PADDING])
        scaled_corners = canvas_corners * scale
        
        final = cv2.resize(result_bgr, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
        
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
        
        expected = scaled_corners[0]
        error = np.sqrt((detected[0]-expected[0])**2 + (detected[1]-expected[1])**2) if detected else float('inf')
        status = "✅" if error < 5 else "❌"
        print(f"  {angle:3d}°: detected={detected}, expected=({expected[0]:.0f},{expected[1]:.0f}), error={error:.1f}px {status}")


if __name__ == '__main__':
    test_no_perspective()
    test_with_lanczos_resize()