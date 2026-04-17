#!/usr/bin/env python3
"""
Test with markers at EXACT photo corners.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import rotate_photo, composite_photo_at_center


def create_corner_photo(width, height):
    """Photo with markers at EXACT corners (0,0), (w,0), (w,h), (0,h)."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    marker_size = 5  # Small to be precise
    
    # TL corner (0,0)
    photo[0:marker_size, 0:marker_size] = [0, 0, 255]  # Red
    
    # TR corner (w,0)
    photo[0:marker_size, width-marker_size:width] = [0, 255, 0]  # Green
    
    # BR corner (w,h)
    photo[height-marker_size:height, width-marker_size:width] = [255, 0, 0]  # Blue
    
    # BL corner (0,h)
    photo[height-marker_size:height, 0:marker_size] = [0, 255, 255]  # Yellow
    
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
        if cv2.contourArea(c) > 10:
            m = cv2.moments(c)
            markers[0] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    # TR - Green
    mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 10:
            m = cv2.moments(c)
            markers[1] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    # BR - Blue
    mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 10:
            m = cv2.moments(c)
            markers[2] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    # BL - Yellow
    mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([45, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 10:
            m = cv2.moments(c)
            markers[3] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    return markers


def main():
    CANVAS_SIZE = 640
    PADDING = 300
    
    photo_width, photo_height = 300, 200
    center_x = CANVAS_SIZE // 2 + PADDING  # 620
    center_y = CANVAS_SIZE // 2 + PADDING  # 620
    
    print("\n" + "="*70)
    print("TEST: Markers at EXACT Photo Corners")
    print("="*70)
    print(f"  Photo: {photo_width}x{photo_height}")
    print(f"  Markers: 5x5 at corners (0,0), (300,0), (300,200), (0,200)")
    print(f"  Canvas center: ({center_x}, {center_y})")
    print(f"  Expected corners: (470,520), (770,520), (770,720), (470,720)")
    
    for angle in [0, 30, 45, 60, 90, -30]:
        photo = create_corner_photo(photo_width, photo_height)
        rotated = rotate_photo(photo, angle)
        rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        
        canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
        canvas[:, :, 3] = 0
        result = composite_photo_at_center(canvas, rgba, center_x, center_y)
        
        result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        detected = detect_colored_markers(result_bgr)
        
        # Expected corners for this rotation
        # Simple: for 0°, corners are center ± hw, ± hh
        hw, hh = photo_width / 2, photo_height / 2
        
        if abs(angle) < 1:
            expected = [
                (center_x - hw, center_y - hh),  # TL
                (center_x + hw, center_y - hh),  # TR
                (center_x + hw, center_y + hh),  # BR
                (center_x - hw, center_y + hh),  # BL
            ]
        else:
            # Use rotation matrix
            photo_center = (photo_width / 2, photo_height / 2)
            M = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
            cos_a = abs(M[0, 0])
            sin_a = abs(M[0, 1])
            new_w = int(photo_height * sin_a + photo_width * cos_a)
            new_h = int(photo_height * cos_a + photo_width * sin_a)
            M[0, 2] += (new_w - photo_width) / 2
            M[1, 2] += (new_h - photo_height) / 2
            
            corners = np.array([[0, 0], [photo_width, 0], [photo_width, photo_height], [0, photo_height]], dtype=np.float32)
            corners_rot = np.zeros_like(corners)
            for i in range(4):
                pt = np.array([corners[i, 0], corners[i, 1], 1])
                result_pt = M @ pt
                corners_rot[i] = [result_pt[0], result_pt[1]]
            
            center_rot = M @ np.array([photo_center[0], photo_center[1], 1])
            expected = []
            for i in range(4):
                expected.append((
                    center_x - center_rot[0] + corners_rot[i, 0],
                    center_y - center_rot[1] + corners_rot[i, 1]
                ))
        
        print(f"\n  {angle:3d}° Rotation:")
        errors = []
        for i in range(4):
            if i in detected:
                dx = detected[i][0] - expected[i][0]
                dy = detected[i][1] - expected[i][1]
                error = np.sqrt(dx**2 + dy**2)
                errors.append(error)
                status = "✅" if error < 5 else "❌"
                print(f"    Corner {i}: expected=({expected[i][0]:.1f},{expected[i][1]:.1f}), "
                      f"detected=({detected[i][0]},{detected[i][1]}), "
                      f"error={error:.1f}px {status}")
            else:
                print(f"    Corner {i}: NOT DETECTED ❌")
        
        if errors:
            avg = sum(errors) / len(errors)
            print(f"    Average error: {avg:.2f}px")


if __name__ == '__main__':
    main()