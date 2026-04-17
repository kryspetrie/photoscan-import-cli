#!/usr/bin/env python3
"""
Debug script to visualize corner positions through the pipeline.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import (
    CONFIG, get_rotated_polygon, rotate_photo, composite_photo_at_center,
    apply_global_perspective
)


def create_marker_photo(width, height, marker_size=20):
    """Create a photo with colored markers at corners."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    colors = [
        (255, 0, 0),     # TL - Red
        (0, 255, 0),     # TR - Green  
        (0, 0, 255),     # BR - Blue
        (255, 255, 0),   # BL - Yellow
    ]
    
    for i, color in enumerate(colors):
        if i == 0:  # TL
            x, y = 5, 5
        elif i == 1:  # TR
            x, y = width - marker_size - 5, 5
        elif i == 2:  # BR
            x, y = width - marker_size - 5, height - marker_size - 5
        else:  # BL
            x, y = 5, height - marker_size - 5
        
        photo[y:y+marker_size, x:x+marker_size] = color
    
    return photo


def visualize_at_stage(stage_name, img, corners=None, title=""):
    """Save visualization of current stage."""
    if img is None:
        return
    vis = img.copy()
    if len(vis.shape) == 3 and vis.shape[2] == 4:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR)
    
    if corners is not None:
        pts = corners.astype(np.int32)
        labels = ['TL', 'TR', 'BR', 'BL']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, (pt, label, color) in enumerate(zip(corners, labels, colors)):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 10, color, -1)
            cv2.putText(vis, f"{label}", (int(pt[0])+12, int(pt[1])-12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.polylines(vis, [pts], True, color, 2)
    
    cv2.imwrite(f'/tmp/debug_{stage_name.replace(" ", "_")}.jpg', vis)
    print(f"  Saved: debug_{stage_name.replace(' ', '_')}.jpg (shape: {img.shape})")


def main():
    CANVAS_SIZE = 640
    PADDING = 300
    
    photo_size = 300
    center = CANVAS_SIZE // 2
    
    print(f"\n{'='*70}")
    print("DEBUG: Corner Tracking Through Pipeline")
    print(f"{'='*70}")
    print(f"  CANVAS_SIZE: {CANVAS_SIZE}")
    print(f"  PADDING: {PADDING}")
    print(f"  Photo size: {photo_size}x{photo_size}")
    print(f"  Center: ({center}, {center})")
    print(f"  CROP_MARGIN: {CONFIG['CROP_MARGIN']}")
    
    photo = create_marker_photo(photo_size, photo_size, marker_size=25)
    
    # Stage 1: Initial photo
    print(f"\n{'='*70}")
    print("STAGE 1: Initial Photo")
    print(f"{'='*70}")
    photo_h, photo_w = photo.shape[:2]
    initial_corners = np.array([
        [0, 0],
        [photo_w, 0],
        [photo_w, photo_h],
        [0, photo_h]
    ], dtype=np.float32)
    visualize_at_stage("1 original photo", photo, initial_corners)
    print(f"  Corners (photo space): {initial_corners}")
    
    # Stage 2: After rotation
    print(f"\n{'='*70}")
    print("STAGE 2: After Rotation (45°)")
    print(f"{'='*70}")
    rotation = 45
    rotated_photo = rotate_photo(photo, rotation)
    if rotated_photo.shape[2] == 3:
        rgba = cv2.cvtColor(rotated_photo, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        rotated_photo = rgba
    visualize_at_stage("2 rotated photo", rotated_photo)
    
    # Stage 3: Canvas with photo placed
    print(f"\n{'='*70}")
    print("STAGE 3: Canvas with Photo Placed")
    print(f"{'='*70}")
    
    canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0  # Transparent
    visualize_at_stage("3a empty canvas", canvas)
    
    # Get corner positions in PADDED space
    corner_cx = center + PADDING
    corner_cy = center + PADDING
    polygon = get_rotated_polygon(photo_w, photo_h, corner_cx, corner_cy, rotation)
    print(f"  Polygon (padded space):\n    {polygon}")
    
    result = composite_photo_at_center(canvas, rotated_photo, corner_cx, corner_cy)
    visualize_at_stage("3b canvas_with_photo", result, polygon)
    
    # Stage 4: After perspective warp
    print(f"\n{'='*70}")
    print("STAGE 4: After Perspective Warp")
    print(f"{'='*70}")
    
    photo_corners_list = [polygon]
    warped, global_corners, transform_matrix, content_bounds, warped_photo_corners = apply_global_perspective(
        result, PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE,
        photo_corners=photo_corners_list,
        crop_margin=CONFIG['CROP_MARGIN']
    )
    
    print(f"  Global corners (warped crop space):\n    {global_corners}")
    print(f"  Photo corners from perspective transform:\n    {warped_photo_corners[0]}")
    print(f"  Content bounds: {content_bounds}")
    
    visualize_at_stage("4 warped", warped, warped_photo_corners[0] if warped_photo_corners else None)
    
    # Stage 5: After resize to 640x640
    print(f"\n{'='*70}")
    print("STAGE 5: After Resize to 640x640")
    print(f"{'='*70}")
    
    out_w, out_h = warped.shape[1], warped.shape[0]
    print(f"  Pre-resize shape: {warped.shape}")
    
    if out_w != CANVAS_SIZE or out_h != CANVAS_SIZE:
        scale_x = CANVAS_SIZE / out_w
        scale_y = CANVAS_SIZE / out_h
        print(f"  Scale factors: ({scale_x:.4f}, {scale_y:.4f})")
        
        final_img = cv2.resize(warped, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_LINEAR)
        
        final_corners = warped_photo_corners[0] * np.array([scale_x, scale_y])
    else:
        final_img = warped
        final_corners = warped_photo_corners[0]
    
    print(f"  Final corners (should be in 0-640 space):\n    {final_corners}")
    visualize_at_stage("5 final", final_img, final_corners)
    
    # Detect actual markers in final image
    print(f"\n{'='*70}")
    print("DETECTION: Actual Marker Positions in Final Image")
    print(f"{'='*70}")
    
    hsv = cv2.cvtColor(final_img, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        'TL': ([0, 100, 100], [10, 255, 255]),      # Red
        'TR': ([40, 100, 100], [80, 255, 255]),     # Green
        'BR': ([100, 100, 100], [130, 255, 255]),   # Blue
        'BL': ([20, 100, 100], [40, 255, 255]),     # Yellow
    }
    
    detected = {}
    for name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detected[name] = (cx, cy)
    
    print(f"\n  Detected marker positions:")
    for name, pos in sorted(detected.items()):
        print(f"    {name}: {pos}")
    
    # Compare
    print(f"\n  Comparison (detected vs expected):")
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, label in enumerate(labels):
        if label in detected:
            dx = detected[label][0] - final_corners[i][0]
            dy = detected[label][1] - final_corners[i][1]
            error = np.sqrt(dx**2 + dy**2)
            print(f"    {label}: detected=({detected[label][0]:.0f}, {detected[label][1]:.0f}), "
                  f"expected=({final_corners[i][0]:.0f}, {final_corners[i][1]:.0f}), "
                  f"error={error:.1f}px")
    
    # Save final annotated image
    vis = final_img.copy()
    pts = final_corners.astype(np.int32)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, (pt, label, color) in enumerate(zip(final_corners, labels, colors)):
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, color, -1)
        cv2.putText(vis, f"{label}", (int(pt[0])+10, int(pt[1])-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.polylines(vis, [pts], True, color, 2)
        if label in detected:
            cv2.circle(vis, (detected[label][0], detected[label][1]), 12, (255, 255, 255), 2)
    
    cv2.imwrite('/tmp/debug_final_annotated.jpg', vis)
    print(f"\n  Saved: /tmp/debug_final_annotated.jpg")


if __name__ == '__main__':
    main()