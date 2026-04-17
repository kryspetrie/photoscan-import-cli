#!/usr/bin/env python3
"""
DETAILED step-by-step trace of the pipeline.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import rotate_photo, composite_photo_at_center, apply_global_perspective


def create_simple_photo(width, height):
    photo = np.ones((height, width, 3), dtype=np.uint8) * 200
    marker = 25
    photo[0:marker, 0:marker] = [0, 0, 255]
    return photo


def detailed_trace():
    """Trace through the exact pipeline."""
    CANVAS_SIZE = 640
    PADDING = 300
    OUT_SIZE = 640
    
    photo_width, photo_height = 300, 200
    center_x = CANVAS_SIZE // 2 + PADDING  # 620
    center_y = CANVAS_SIZE // 2 + PADDING  # 620
    rotation = 0
    
    print("\n" + "="*70)
    print("DETAILED TRACE")
    print("="*70)
    print(f"\nInitial parameters:")
    print(f"  Canvas: {CANVAS_SIZE}x{CANVAS_SIZE}")
    print(f"  Padding: {PADDING}px each side")
    print(f"  Total canvas: {CANVAS_SIZE + 2*PADDING}x{CANVAS_SIZE + 2*PADDING}")
    print(f"  Photo: {photo_width}x{photo_height}")
    print(f"  Center in padded space: ({center_x}, {center_y})")
    print(f"  Rotation: {rotation}°")
    
    print(f"\n" + "-"*70)
    print("STEP 1: Create canvas and photo")
    print("-"*70)
    
    photo = create_simple_photo(photo_width, photo_height)
    print(f"  Photo created: {photo_width}x{photo_height}")
    print(f"  Marker at top-left of photo: (0:25, 0:25)")
    
    print(f"\n" + "-"*70)
    print("STEP 2: Rotate photo")
    print("-"*70)
    
    rotated = rotate_photo(photo, rotation)
    print(f"  Rotated photo shape: {rotated.shape}")
    print(f"  (For 0° rotation, should be same as original)")
    
    if rotated.shape[2] == 3:
        rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        rotated = rgba
    print(f"  With alpha: {rotated.shape}")
    
    print(f"\n" + "-"*70)
    print("STEP 3: Create canvas and composite")
    print("-"*70)
    
    canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    print(f"  Empty canvas: {canvas.shape}")
    print(f"  Canvas background: gray (180)")
    print(f"  Canvas alpha: 0 (transparent)")
    
    print(f"\n  Before composite:")
    print(f"    Canvas center (for placement): ({center_x}, {center_y})")
    print(f"    Rotated photo center: ({rotated.shape[1]//2}, {rotated.shape[0]//2})")
    print(f"    Photo TL will be at: ({center_x - rotated.shape[1]//2}, {center_y - rotated.shape[0]//2})")
    
    result = composite_photo_at_center(canvas, rotated, center_x, center_y)
    print(f"\n  After composite: {result.shape}")
    
    # Where is the red pixel?
    hsv = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) > 0:
        marker_x = np.median(x_coords)
        marker_y = np.median(y_coords)
        print(f"\n  Marker position after composite: ({marker_x:.0f}, {marker_y:.0f})")
        print(f"  Expected position (padded space): (470, 520)")
        print(f"  Offset from expected: ({marker_x - 470:.0f}, {marker_y - 520:.0f})")
    
    print(f"\n" + "-"*70)
    print("STEP 4: Apply perspective warp")
    print("-"*70)
    
    # For now, skip perspective - just measure the composite
    print("  SKIPPING perspective for this test")
    
    print(f"\n" + "-"*70)
    print("STEP 5: Resize to output size")
    print("-"*70)
    
    # Simulate the crop that perspective would do
    # The canvas is 1240x1240 (640 + 2*300)
    # After crop of 60px margin on each side, it becomes 1120x1120
    # Then resize to 640x640 = scale of 640/1120 = 0.571
    
    print(f"  Current image: {result.shape[1]}x{result.shape[0]}")
    print(f"  Expected crop: ({60}px margin on {CANVAS_SIZE + 2*PADDING}px image)")
    
    # The issue is: we need to understand where in the 640x640 output the marker appears
    # if the canvas gets cropped and resized
    
    # Let's say perspective crops to some size, then resize to 640x640
    # Scale factor depends on the actual crop
    
    # For simplicity, let's just resize the 1240x1240 image directly to 640x640
    # (ignoring the perspective crop for now)
    
    result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
    scale = OUT_SIZE / result.shape[1]  # 640 / 1240 = 0.516
    print(f"  Direct resize scale: {scale:.3f}")
    
    final = cv2.resize(result_bgr, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # Find marker in final
    hsv_final = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
    mask_final = cv2.inRange(hsv_final, np.array([0, 100, 100]), np.array([15, 255, 255]))
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 10:
            M = cv2.moments(largest)
            if M['m00'] > 0:
                marker_final = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
                print(f"\n  Marker in final 640x640 image: {marker_final}")
    
    print(f"\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)
    print(f"  Canvas center: ({center_x}, {center_y})")
    print(f"  Photo center (rotated): ({rotated.shape[1]//2}, {rotated.shape[0]//2})")
    print(f"  Photo TL in canvas: ({center_x - rotated.shape[1]//2}, {center_y - rotated.shape[0]//2})")
    
    # For 0° rotation, photo is 300x200
    # Photo TL should be at (620-150, 620-100) = (470, 520)
    # Marker is at (0:25, 0:25) relative to photo, so at (470+0:25, 520+0:25) in canvas


if __name__ == '__main__':
    detailed_trace()