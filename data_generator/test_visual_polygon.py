#!/usr/bin/env python3
"""
Visual verification test - draws polygon on image and checks alignment.
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


def get_rotated_polygon_fixed(width, height, center_x, center_y, rotation):
    """Corrected polygon calculation."""
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
    
    center_rot = M @ np.array([photo_center[0], photo_center[1], 1])
    
    return corners_rot + np.array([center_x - center_rot[0], center_y - center_rot[1]])


def create_test_photo(width, height):
    """Create a simple photo with clear edges."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 220
    
    # Draw a border and corner markers
    cv2.rectangle(photo, (0, 0), (width-1, height-1), (50, 50, 50), 3)
    
    # Corner markers (bright colors at corners)
    marker_sz = 30
    # TL - Cyan
    photo[5:5+marker_sz, 5:5+marker_sz] = [255, 255, 0]
    # TR - Magenta
    photo[5:5+marker_sz, width-marker_sz-5:width-5] = [255, 0, 255]
    # BR - White
    photo[height-marker_sz-5:height-5, width-marker_sz-5:width-5] = [255, 255, 255]
    # BL - Green
    photo[height-marker_sz-5:height-5, 5:5+marker_sz] = [0, 255, 255]
    
    # Add lines pointing to corners
    cv2.line(photo, (50, 15), (width-50, 15), (100, 100, 100), 1)
    cv2.line(photo, (15, 50), (15, height-50), (100, 100, 100), 1)
    
    return photo


def run_pipeline(rotation):
    """Run full pipeline and return final image with polygon drawn."""
    CANVAS_SIZE = 640
    PADDING = 300
    out_size = 640
    
    photo_width, photo_height = 280, 200  # Rectangular for clarity
    center_x = CANVAS_SIZE // 2 + PADDING
    center_y = CANVAS_SIZE // 2 + PADDING
    
    photo = create_test_photo(photo_width, photo_height)
    
    # Get expected polygon
    polygon = get_rotated_polygon_fixed(photo_width, photo_height, center_x, center_y, rotation)
    
    # Rotate and composite
    rotated = rotate_photo(photo, rotation)
    if rotated.shape[2] == 3:
        rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        rotated = rgba
    
    canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    result = composite_photo_at_center(canvas, rotated, center_x, center_y)
    
    # Apply perspective
    warped, global_corners, transform_matrix, content_bounds, warped_corners = apply_global_perspective(
        result, PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE,
        photo_corners=[polygon],
        crop_margin=CONFIG['CROP_MARGIN']
    )
    
    # Resize to 640x640
    out_w, out_h = warped.shape[1], warped.shape[0]
    if out_w != out_size or out_h != out_size:
        scale_x = out_size / out_w
        scale_y = out_size / out_h
        final = cv2.resize(warped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        final_corners = warped_corners[0] * np.array([scale_x, scale_y])
    else:
        final = warped
        final_corners = warped_corners[0]
    
    return final, final_corners


def measure_polygon_accuracy(img, corners):
    """
    Measure how well the polygon matches visible photo edges.
    Uses edge detection to find actual photo boundary and compare to polygon.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find the centroid of edges in small regions near polygon corners
    corner_errors = []
    offsets = [(0, 0), (50, 0), (50, 50), (0, 50)]  # Regions to search around each corner
    
    for i, (offset_x, offset_y) in enumerate(offsets):
        cx, cy = int(corners[i][0]), int(corners[i][1])
        
        # Define search region
        x1 = max(0, cx + offset_x - 40)
        y1 = max(0, cy + offset_y - 40)
        x2 = min(img.shape[1], cx + offset_x + 40)
        y2 = min(img.shape[0], cy + offset_y + 40)
        
        if x2 > x1 and y2 > y1:
            region = edges[y1:y2, x1:x2]
            
            # Find centroid of edges in region
            moments = cv2.moments(region)
            if moments['m00'] > 100:
                edge_cx = moments['m10'] / moments['m00'] + x1
                edge_cy = moments['m01'] / moments['m00'] + y1
                
                error = np.sqrt((edge_cx - corners[i][0])**2 + (edge_cy - corners[i][1])**2)
                corner_errors.append(error)
            else:
                corner_errors.append(None)
    
    return corner_errors


def main():
    print("\n" + "="*70)
    print("VISUAL POLYGON VERIFICATION")
    print("="*70)
    
    test_angles = [0, 15, 30, 45, 60, 90, -30]
    all_results = []
    
    for angle in test_angles:
        print(f"\n{'='*70}")
        print(f"TEST: {angle}° rotation")
        print("="*70)
        
        final, corners = run_pipeline(angle)
        
        # Create annotated image
        vis = final.copy()
        
        # Draw polygon
        pts = corners.astype(np.int32)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        labels = ['0', '1', '2', '3']
        
        # Draw polygon edges
        cv2.polylines(vis, [pts], True, (0, 255, 255), 3)
        
        # Draw corner markers
        for i, (pt, color, label) in enumerate(zip(corners, colors, labels)):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 15, color, -1)
            cv2.putText(vis, label, (int(pt[0])+18, int(pt[1])-18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 3)
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 15, (255, 255, 255), 2)
        
        # Also search for edges near corners
        edge_errors = measure_polygon_accuracy(final, corners)
        
        # Save
        cv2.imwrite(f'/tmp/polygon_test_{angle}.jpg', vis)
        print(f"\n  Polygon corners:")
        for i, pt in enumerate(corners):
            err_str = f"edge_err={edge_errors[i]:.1f}" if edge_errors[i] else "edge_err=N/A"
            print(f"    {i}: ({pt[0]:.0f}, {pt[1]:.0f}) - {err_str}")
        
        print(f"\n  Saved: polygon_test_{angle}.jpg")
        
        all_results.append((angle, corners, edge_errors))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for angle, corners, edge_errors in all_results:
        valid_errors = [e for e in edge_errors if e is not None]
        if valid_errors:
            avg = sum(valid_errors) / len(valid_errors)
            max_e = max(valid_errors)
            print(f"  {angle:4d}°: avg_edge_error={avg:.1f}px, max={max_e:.1f}px")
        else:
            print(f"  {angle:4d}°: no valid edge measurements")


if __name__ == '__main__':
    main()