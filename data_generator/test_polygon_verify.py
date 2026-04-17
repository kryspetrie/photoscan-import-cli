#!/usr/bin/env python3
"""
Verify polygon calculation against ACTUAL photo corners.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import rotate_photo, composite_photo_at_center


def create_marker_photo(width, height):
    """Photo with colored markers at each corner."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    marker = 20
    # TL - Red
    photo[0:marker, 0:marker] = [0, 0, 255]
    # TR - Green
    photo[0:marker, width-marker:width] = [0, 255, 0]
    # BR - Blue
    photo[height-marker:height, width-marker:width] = [255, 0, 0]
    # BL - Yellow
    photo[height-marker:height, 0:marker] = [0, 255, 255]
    
    return photo


def get_rotated_polygon_v2(width, height, center_x, center_y, rotation):
    """Polygon calculation to verify."""
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


def detect_colored_markers(img):
    """Detect the 4 colored markers."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    markers = {}
    
    # TL - Red
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 50:
            m = cv2.moments(c)
            markers[0] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    # TR - Green
    mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 50:
            m = cv2.moments(c)
            markers[1] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    # BR - Blue
    mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 50:
            m = cv2.moments(c)
            markers[2] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    # BL - Yellow
    mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([45, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 50:
            m = cv2.moments(c)
            markers[3] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    return markers


def main():
    CANVAS_SIZE = 640
    PADDING = 300
    
    photo_width, photo_height = 300, 200
    
    print("\n" + "="*70)
    print("POLYGON VERIFICATION: Padded Canvas Space (NO perspective)")
    print("="*70)
    
    for angle in [0, 30, 45, 60, 90, -30]:
        center_x = CANVAS_SIZE // 2 + PADDING
        center_y = CANVAS_SIZE // 2 + PADDING
        
        photo = create_marker_photo(photo_width, photo_height)
        polygon = get_rotated_polygon_v2(photo_width, photo_height, center_x, center_y, angle)
        
        rotated = rotate_photo(photo, angle)
        if rotated.shape[2] == 3:
            rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = 255
            rotated = rgba
        
        canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
        canvas[:, :, 3] = 0
        result = composite_photo_at_center(canvas, rotated, center_x, center_y)
        
        # Detect actual marker positions
        result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        detected = detect_colored_markers(result_bgr)
        
        print(f"\n  {angle:3d}° Rotation:")
        print(f"    Calculated polygon: {polygon[0]}")
        print(f"    Detected markers: {detected}")
        
        # Compare
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
        
        if errors:
            avg = sum(errors) / len(errors)
            print(f"    Average error: {avg:.2f}px")


if __name__ == '__main__':
    main()