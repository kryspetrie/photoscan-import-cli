#!/usr/bin/env python3
"""
Full end-to-end verification with EXACT corner markers.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import (
    CONFIG, rotate_photo, composite_photo_at_center, apply_global_perspective
)


def create_corner_photo(width, height):
    """Photo with markers at EXACT corners."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    marker_size = 8
    
    # TL (0,0) - Red
    photo[0:marker_size, 0:marker_size] = [0, 0, 255]
    # TR (w,0) - Green
    photo[0:marker_size, width-marker_size:width] = [0, 255, 0]
    # BR (w,h) - Blue
    photo[height-marker_size:height, width-marker_size:width] = [255, 0, 0]
    # BL (0,h) - Yellow
    photo[height-marker_size:height, 0:marker_size] = [0, 255, 255]
    
    return photo


def get_rotated_polygon(width, height, center_x, center_y, rotation):
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


def detect_markers(img):
    """Detect corner markers."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    markers = {}
    
    ranges = [
        (0, [0, 100, 100], [15, 255, 255]),   # TL - Red
        (1, [35, 100, 100], [85, 255, 255]),  # TR - Green
        (2, [100, 100, 100], [130, 255, 255]), # BR - Blue
        (3, [15, 100, 100], [45, 255, 255])    # BL - Yellow
    ]
    
    for idx, lower, upper in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 10:
                m = cv2.moments(c)
                markers[idx] = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    
    return markers


def run_pipeline(angle, out_size=640):
    """Run full pipeline."""
    CANVAS_SIZE = 640
    PADDING = 300
    
    photo_width, photo_height = 300, 200
    center_x = CANVAS_SIZE // 2 + PADDING
    center_y = CANVAS_SIZE // 2 + PADDING
    
    photo = create_corner_photo(photo_width, photo_height)
    polygon = get_rotated_polygon(photo_width, photo_height, center_x, center_y, angle)
    
    rotated = rotate_photo(photo, angle)
    rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 255
    
    canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    result = composite_photo_at_center(canvas, rgba, center_x, center_y)
    
    # Apply perspective
    warped, global_corners, transform_matrix, content_bounds, warped_corners = apply_global_perspective(
        result, PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE,
        photo_corners=[polygon],
        crop_margin=CONFIG['CROP_MARGIN']
    )
    
    # Resize to output
    warped_h, warped_w = warped.shape[:2]
    scale_x = out_size / warped_w
    scale_y = out_size / warped_h
    
    final = cv2.resize(warped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    final_corners = warped_corners[0] * np.array([scale_x, scale_y])
    
    # Detect markers
    detected = detect_markers(final)
    
    return final_corners, detected, warped_corners[0]


def main():
    print("\n" + "="*70)
    print("FULL PIPELINE VERIFICATION: Rotation + Perspective + Resize")
    print("="*70)
    print("  5x5 pixel markers at photo corners (0,0), (w,0), (w,h), (0,h)")
    
    all_errors = []
    
    for angle in [0, 30, 45, 60, 90, -30]:
        final_corners, detected, warped = run_pipeline(angle)
        
        print(f"\n  {angle:3d}° Rotation:")
        errors = []
        for i in range(4):
            if i in detected:
                dx = detected[i][0] - final_corners[i][0]
                dy = detected[i][1] - final_corners[i][1]
                error = np.sqrt(dx**2 + dy**2)
                errors.append(error)
                status = "✅" if error < 5 else "⚠️" if error < 8 else "❌"
                print(f"    Corner {i}: expected=({final_corners[i][0]:.1f},{final_corners[i][1]:.1f}), "
                      f"detected=({detected[i][0]},{detected[i][1]}), "
                      f"error={error:.1f}px {status}")
            else:
                print(f"    Corner {i}: NOT DETECTED ❌")
        
        if errors:
            avg = sum(errors) / len(errors)
            max_e = max(errors)
            all_errors.extend(errors)
            print(f"    Average error: {avg:.2f}px (max: {max_e:.2f}px)")
    
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    if all_errors:
        overall_avg = sum(all_errors) / len(all_errors)
        overall_max = max(all_errors)
        print(f"  Overall average error: {overall_avg:.2f}px")
        print(f"  Overall max error: {overall_max:.2f}px")
        
        under_5 = sum(1 for e in all_errors if e < 5)
        under_8 = sum(1 for e in all_errors if e < 8)
        total = len(all_errors)
        print(f"  Corners within 5px: {under_5}/{total} ({100*under_5/total:.0f}%)")
        print(f"  Corners within 8px: {under_8}/{total} ({100*under_8/total:.0f}%)")
        
        if under_5 == total:
            print("\n  ✅ ALL CORNERS WITHIN 5px ACCURACY!")
        elif under_8 == total:
            print("\n  ⚠️  ALL CORNERS WITHIN 8px (acceptable for training)")


if __name__ == '__main__':
    main()