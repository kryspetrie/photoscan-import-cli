#!/usr/bin/env python3
"""
Coordinate Transform Debug Script
=================================

Tests the coordinate transformation pipeline step-by-step to identify bugs.

Pipeline:
1. Place photo at (cx, cy) with corners at (cx±w/2, cy±h/2)
2. Apply perspective transform to corners
3. Crop to content bounds
4. Resize to 640x640
5. Compare final corners vs. visual position
"""

import sys
import os
from pathlib import Path
import math
import random

import numpy as np
import cv2
from PIL import Image

# Configuration
CANVAS_SIZE = 640
CONFIG = {
    'PERSPECTIVE_STRENGTH_MIN': 0.05,
    'PERSPECTIVE_STRENGTH_MAX': 0.15,  # Start conservative
    'CROP_MARGIN': 40,
}

def get_photo_corners(width, height, center_x, center_y, rotation=0):
    """Get 4 corners of a (possibly rotated) rectangle."""
    hw, hh = width / 2, height / 2
    corners = np.array([
        [-hw, -hh],  # TL: top-left relative
        [ hw, -hh],  # TR: top-right relative
        [ hw,  hh],  # BR: bottom-right relative
        [-hw,  hh],  # BL: bottom-left relative
    ], dtype=np.float32)
    
    if abs(rotation) > 0.5:
        angle_rad = math.radians(rotation)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotated = np.array([
            [
                corners[i, 0] * cos_a - corners[i, 1] * sin_a + center_x,
                corners[i, 0] * sin_a + corners[i, 1] * cos_a + center_y
            ]
            for i in range(4)
        ], dtype=np.float32)
        return rotated
    
    # No rotation - just translate
    corners[:, 0] += center_x
    corners[:, 1] += center_y
    return corners


def apply_perspective_transform_corners(corners, src_size, dst_corners):
    """Apply perspective transform to corners."""
    src = np.array([
        [0, 0],
        [src_size - 1, 0],
        [src_size - 1, src_size - 1],
        [0, src_size - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src, dst_corners)
    
    # Transform each corner
    warped = []
    for corner in corners:
        x, y = corner
        # Apply transform manually
        xn = (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / (M[2, 0] * x + M[2, 1] * y + M[2, 2])
        yn = (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / (M[2, 0] * x + M[2, 1] * y + M[2, 2])
        warped.append([xn, yn])
    
    return np.array(warped, dtype=np.float32)


def create_test_image(photo_path, test_num, photo_w, photo_h, center_x, center_y, rotation, perspective_strength, direction):
    """Create a test image and track coordinates through transformation."""
    
    print(f"\n{'='*60}")
    print(f"TEST {test_num}: {photo_w}x{photo_h} at ({center_x}, {center_y}), rot={rotation}°")
    print(f"Perspective: {perspective_strength*100:.0f}%, dir={direction}")
    print('='*60)
    
    # Load photo
    photo = cv2.imread(str(photo_path))
    if photo is None:
        print("ERROR: Could not load photo")
        return None, None, None
    
    # Resize photo
    photo = cv2.resize(photo, (photo_w, photo_h))
    
    # Create canvas
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 180
    
    # Place photo at center
    x1 = int(center_x - photo_w / 2)
    y1 = int(center_y - photo_h / 2)
    canvas[y1:y1+photo_h, x1:x1+photo_w] = photo
    
    print(f"\n[Step 1] Original photo placed:")
    print(f"  Canvas: {CANVAS_SIZE}x{CANVAS_SIZE}")
    print(f"  Photo corners (canvas coords):")
    
    # Original corners in canvas coords
    original_corners = get_photo_corners(photo_w, photo_h, center_x, center_y, rotation)
    corner_names = ['TL', 'TR', 'BR', 'BL']
    for i, (name, corner) in enumerate(zip(corner_names, original_corners)):
        print(f"    {name}: ({corner[0]:.1f}, {corner[1]:.1f})")
    
    # Create background for perspective
    bg = canvas.copy()
    
    # Perspective transform
    src = np.array([
        [0, 0],
        [CANVAS_SIZE - 1, 0],
        [CANVAS_SIZE - 1, CANVAS_SIZE - 1],
        [0, CANVAS_SIZE - 1]
    ], dtype=np.float32)
    
    max_offset = CANVAS_SIZE * perspective_strength
    
    # Direction-based offsets
    if direction == 0:  # Left closer
        dst = np.array([
            [max_offset, 0],
            [CANVAS_SIZE - 1 - max_offset * 0.5, 0],
            [CANVAS_SIZE - 1 - max_offset * 0.5, CANVAS_SIZE - 1],
            [max_offset, CANVAS_SIZE - 1]
        ], dtype=np.float32)
    elif direction == 1:  # Right closer
        dst = np.array([
            [max_offset * 0.5, 0],
            [CANVAS_SIZE - 1 - max_offset, 0],
            [CANVAS_SIZE - 1 - max_offset, CANVAS_SIZE - 1],
            [max_offset * 0.5, CANVAS_SIZE - 1]
        ], dtype=np.float32)
    elif direction == 2:  # Top closer
        dst = np.array([
            [0, max_offset],
            [CANVAS_SIZE - 1, max_offset],
            [CANVAS_SIZE - 1, CANVAS_SIZE - 1 - max_offset * 0.5],
            [0, CANVAS_SIZE - 1 - max_offset * 0.5]
        ], dtype=np.float32)
    else:  # Bottom closer (direction == 3)
        dst = np.array([
            [0, max_offset * 0.5],
            [CANVAS_SIZE - 1, max_offset * 0.5],
            [CANVAS_SIZE - 1, CANVAS_SIZE - 1 - max_offset],
            [0, CANVAS_SIZE - 1 - max_offset]
        ], dtype=np.float32)
    
    print(f"\n[Step 2] Perspective transform:")
    print(f"  Source corners: {src.tolist()}")
    print(f"  Dest corners: {dst.tolist()}")
    
    # Calculate bounding box of warped output
    min_x = min(c[0] for c in dst)
    max_x = max(c[0] for c in dst)
    min_y = min(c[1] for c in dst)
    max_y = max(c[1] for c in dst)
    out_w = int(max_x - min_x) + 1
    out_h = int(max_y - min_y) + 1
    
    print(f"  Output bounds: ({min_x:.1f}, {min_y:.1f}) to ({max_x:.1f}, {max_y:.1f})")
    print(f"  Output size: {out_w}x{out_h}")
    
    # Apply perspective
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(bg, M, (out_w, out_h))
    
    print(f"\n[Step 3] Transform corners through perspective:")
    warped_corners = apply_perspective_transform_corners(original_corners, CANVAS_SIZE, dst)
    for i, (name, corner) in enumerate(zip(corner_names, warped_corners)):
        print(f"    {name}: ({corner[0]:.1f}, {corner[1]:.1f})")
    
    # Apply crop offset
    print(f"\n[Step 4] Apply crop offset (-{min_x:.1f}, -{min_y:.1f}):")
    cropped_corners = warped_corners.copy()
    cropped_corners[:, 0] -= min_x
    cropped_corners[:, 1] -= min_y
    for i, (name, corner) in enumerate(zip(corner_names, cropped_corners)):
        print(f"    {name}: ({corner[0]:.1f}, {corner[1]:.1f})")
    
    # Resize to 640x640
    print(f"\n[Step 5] Resize to {CANVAS_SIZE}x{CANVAS_SIZE}:")
    scale_x = CANVAS_SIZE / out_w if out_w > 0 else 1
    scale_y = CANVAS_SIZE / out_h if out_h > 0 else 1
    print(f"  Scale factors: ({scale_x:.4f}, {scale_y:.4f})")
    
    final_corners = cropped_corners.copy()
    final_corners[:, 0] *= scale_x
    final_corners[:, 1] *= scale_y
    for i, (name, corner) in enumerate(zip(corner_names, final_corners)):
        print(f"    {name}: ({corner[0]:.1f}, {corner[1]:.1f})")
    
    # Resize warped image
    warped_resized = cv2.resize(warped, (CANVAS_SIZE, CANVAS_SIZE))
    
    # Calculate bounding box
    x_min = min(c[0] for c in final_corners)
    x_max = max(c[0] for c in final_corners)
    y_min = min(c[1] for c in final_corners)
    y_max = max(c[1] for c in final_corners)
    bbox_x = (x_min + x_max) / 2 / CANVAS_SIZE
    bbox_y = (y_min + y_max) / 2 / CANVAS_SIZE
    bbox_w = (x_max - x_min) / CANVAS_SIZE
    bbox_h = (y_max - y_min) / CANVAS_SIZE
    
    print(f"\n[Step 6] Bounding box (normalized):")
    print(f"  Center: ({bbox_x:.4f}, {bbox_y:.4f})")
    print(f"  Size: {bbox_w:.4f} x {bbox_h:.4f}")
    
    # Draw debug visualization
    debug_img = warped_resized.copy()
    
    # Draw photo outline using corners
    pts = final_corners.astype(np.int32)
    cv2.polylines(debug_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Draw corners
    colors = [(255, 0, 0), (0, 255, 255), (0, 255, 0), (0, 128, 255)]  # B, Y, G, O
    labels = ['LL', 'UL', 'UR', 'LR']
    
    # Reorder to LL, UL, UR, LR based on y then x
    ordered_indices = np.argsort(final_corners[:, 1])
    
    for idx, i in enumerate(ordered_indices):
        x, y = int(final_corners[i, 0]), int(final_corners[i, 1])
        cv2.circle(debug_img, (x, y), 8, colors[idx], -1)
        cv2.putText(debug_img, labels[idx], (x + 10, y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw bounding box
    bbox_pixels = [
        int((bbox_x - bbox_w/2) * CANVAS_SIZE),
        int((bbox_y - bbox_h/2) * CANVAS_SIZE),
        int((bbox_x + bbox_w/2) * CANVAS_SIZE),
        int((bbox_y + bbox_h/2) * CANVAS_SIZE)
    ]
    cv2.rectangle(debug_img, 
                  (bbox_pixels[0], bbox_pixels[1]), 
                  (bbox_pixels[2], bbox_pixels[3]), 
                  (255, 0, 255), 2)
    
    # Add text info
    info = f"Test {test_num}: {photo_w}x{photo_h} rot={rotation}° persp={perspective_strength*100:.0f}%"
    cv2.putText(debug_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Corners: LL={final_corners[0]}, UL={final_corners[1]}, UR={final_corners[2]}, LR={final_corners[3]}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return debug_img, {
        'original_corners': original_corners,
        'warped_corners': warped_corners,
        'cropped_corners': cropped_corners,
        'final_corners': final_corners,
        'bbox': (bbox_x, bbox_y, bbox_w, bbox_h),
        'scale': (scale_x, scale_y),
        'out_w': out_w,
        'out_h': out_h,
    }


def main():
    """Run diagnostic tests."""
    print("=" * 60)
    print("COORDINATE TRANSFORM DEBUG")
    print("=" * 60)
    
    # Find source image
    source_dir = Path("./images")
    sources = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.jpeg'))
    if not sources:
        print("ERROR: No source images found")
        return
    photo_path = sources[0]
    print(f"Using source: {photo_path}")
    
    # Create output directory
    output_dir = Path("../data/debug_transform")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_cases = [
        # Test 1: Center, no rotation
        {'test': 1, 'w': 300, 'h': 225, 'cx': 320, 'cy': 320, 'rot': 0, 
         'persp': 0.10, 'dir': 0},
        # Test 2: Off-center, no rotation
        {'test': 2, 'w': 250, 'h': 188, 'cx': 250, 'cy': 400, 'rot': 0,
         'persp': 0.15, 'dir': 1},
        # Test 3: With rotation
        {'test': 3, 'w': 280, 'h': 210, 'cx': 400, 'cy': 300, 'rot': 15,
         'persp': 0.12, 'dir': 2},
        # Test 4: Strong perspective
        {'test': 4, 'w': 320, 'h': 240, 'cx': 320, 'cy': 320, 'rot': 0,
         'persp': 0.20, 'dir': 3},
        # Test 5: Corner placement
        {'test': 5, 'w': 200, 'h': 150, 'cx': 200, 'cy': 200, 'rot': -10,
         'persp': 0.15, 'dir': 0},
    ]
    
    results = []
    for tc in test_cases:
        img, data = create_test_image(
            photo_path,
            tc['test'], tc['w'], tc['h'], tc['cx'], tc['cy'],
            tc['rot'], tc['persp'], tc['dir']
        )
        if img is not None:
            out_path = output_dir / f"test_{tc['test']:02d}.png"
            cv2.imwrite(str(out_path), img)
            print(f"\n  Saved: {out_path}")
            results.append(data)
    
    # Create comparison image
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY OF COORDINATE TRANSFORMS")
        print('='*60)
        
        for i, r in enumerate(results):
            print(f"\nTest {i+1}:")
            print(f"  Original corners: {r['original_corners'].tolist()}")
            print(f"  Final corners: {r['final_corners'].tolist()}")
            print(f"  Scale applied: ({r['scale'][0]:.4f}, {r['scale'][1]:.4f})")
            print(f"  BBox: center=({r['bbox'][0]:.4f}, {r['bbox'][1]:.4f}), size=({r['bbox'][2]:.4f}, {r['bbox'][3]:.4f})")
    
    print(f"\n{'='*60}")
    print("Debug images saved to: {output_dir}")
    print("="*60)
    
    # Create side-by-side comparison
    if len(results) >= 4:
        # Stack 2x2
        top = np.hstack([cv2.imread(str(output_dir / f"test_{i:02d}.png")) for i in [1, 2]])
        bottom = np.hstack([cv2.imread(str(output_dir / f"test_{i:02d}.png")) for i in [3, 4]])
        combined = np.vstack([top, bottom])
        cv2.imwrite(str(output_dir / "comparison_2x2.png"), combined)
        print(f"Comparison image: {output_dir / 'comparison_2x2.png'}")


if __name__ == "__main__":
    main()
