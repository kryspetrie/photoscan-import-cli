#!/usr/bin/env python3
"""
Test polygon calculation WITHOUT edge blur to isolate the issue.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import rotate_photo, composite_photo_at_center, CONFIG, blur_alpha_edges


def create_marker_photo(width, height, marker_size=30):
    """Photo with LARGE colored corner markers - large enough to not be affected by edge blur."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Large markers (80x80) - the centroid should be stable even with edge blur
    # TL - Red (top-left corner)
    photo[0:marker_size, 0:marker_size] = [0, 0, 255]
    
    # TR - Green (top-right corner)  
    photo[0:marker_size, width-marker_size:width] = [0, 255, 0]
    
    # BR - Blue (bottom-right corner)
    photo[height-marker_size:height, width-marker_size:width] = [255, 0, 0]
    
    # BL - Yellow (bottom-left corner)
    photo[height-marker_size:height, 0:marker_size] = [0, 255, 255]
    
    return photo


def detect_colored_markers(img):
    """Detect the 4 colored markers."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    markers = {}
    
    # TL - Red
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:
            m = cv2.moments(c)
            markers[0] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    # TR - Green
    mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:
            m = cv2.moments(c)
            markers[1] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    # BR - Blue
    mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:
            m = cv2.moments(c)
            markers[2] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    # BL - Yellow
    mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([45, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 100:
            m = cv2.moments(c)
            markers[3] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    return markers


def get_rotated_polygon_v2(width, height, center_x, center_y, rotation):
    """Polygon calculation."""
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


def main():
    CANVAS_SIZE = 640
    PADDING = 300
    
    photo_width, photo_height = 300, 200
    
    print("\n" + "="*70)
    print("TESTING WITH LARGE MARKERS")
    print("="*70)
    print(f"  Marker size: 30x30 (larger than edge blur effect)")
    
    for angle in [0, 30, 45, 60, 90, -30]:
        center_x = CANVAS_SIZE // 2 + PADDING
        center_y = CANVAS_SIZE // 2 + PADDING
        
        photo = create_marker_photo(photo_width, photo_height, marker_size=30)
        
        # Add alpha and edge blur (as in real pipeline)
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
        photo[:, :, 3] = 255
        photo = blur_alpha_edges(photo)
        
        rotated = rotate_photo(photo, angle)
        
        canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
        canvas[:, :, 3] = 0
        result = composite_photo_at_center(canvas, rotated, center_x, center_y)
        
        result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        detected = detect_colored_markers(result_bgr)
        
        polygon = get_rotated_polygon_v2(photo_width, photo_height, center_x, center_y, angle)
        
        print(f"\n  {angle:3d}° Rotation:")
        errors = []
        for i in range(4):
            if i in detected:
                dx = detected[i][0] - polygon[i][0]
                dy = detected[i][1] - polygon[i][1]
                error = np.sqrt(dx**2 + dy**2)
                errors.append(error)
                status = "✅" if error < 5 else "❌"
                print(f"    Corner {i}: calc=({polygon[i][0]:.1f},{polygon[i][1]:.1f}), "
                      f"det=({detected[i][0]},{detected[i][1]}), "
                      f"dx={dx:.1f}, dy={dy:.1f}, error={error:.1f}px {status}")
            else:
                print(f"    Corner {i}: NOT DETECTED ❌")
        
        if errors:
            avg = sum(errors) / len(errors)
            print(f"    Average error: {avg:.2f}px")


if __name__ == '__main__':
    main()