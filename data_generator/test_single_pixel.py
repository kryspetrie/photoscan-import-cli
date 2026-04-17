#!/usr/bin/env python3
"""
Debug test: verify exact pixel positions.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import rotate_photo, composite_photo_at_center


def create_simple_photo(width, height):
    """Photo with single white pixel at center."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Single white pixel at exact center
    cy, cx = height // 2, width // 2
    photo[cy-2:cy+3, cx-2:cx+3] = [255, 255, 255]
    
    return photo


def main():
    CANVAS_SIZE = 640
    PADDING = 300
    
    photo_width, photo_height = 300, 200
    center_x = CANVAS_SIZE // 2 + PADDING  # 620
    center_y = CANVAS_SIZE // 2 + PADDING  # 620
    
    print("\n" + "="*70)
    print("DEBUG: Single pixel position verification")
    print("="*70)
    print(f"  Photo: {photo_width}x{photo_height}")
    print(f"  Pixel at: ({photo_width//2}, {photo_height//2}) = (150, 100)")
    print(f"  Canvas center: ({center_x}, {center_y})")
    print(f"  Expected pixel in canvas: ({center_x}, {center_y}) = (620, 620)")
    
    photo = create_simple_photo(photo_width, photo_height)
    print(f"\n  Photo center pixel value: {photo[100, 150]}")
    
    # Test rotate_photo
    rotated = rotate_photo(photo, 0)
    print(f"\n  After rotate_photo(0°):")
    print(f"    Shape: {rotated.shape}")
    print(f"    Pixel at center: ", end="")
    if rotated.shape[0] > 100 and rotated.shape[1] > 150:
        print(rotated[100, 150])
    else:
        print("OUT OF RANGE")
    
    # Composite
    rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 255
    
    canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    
    result = composite_photo_at_center(canvas, rgba, center_x, center_y)
    
    print(f"\n  After composite:")
    print(f"    Canvas center pixel: {result[620, 620]}")
    print(f"    Pixel at (470, 520): {result[520, 470]}")
    
    # Find the white pixel
    result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n  White pixel detection:")
    if contours:
        for cnt in contours:
            if cv2.contourArea(cnt) > 0:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    print(f"    Found at: ({cx}, {cy})")
                    print(f"    Expected: ({center_x}, {center_y}) = (620, 620)")
                    print(f"    Error: ({cx - center_x}, {cy - center_y})")
    else:
        print("    No white pixel found!")


if __name__ == '__main__':
    main()