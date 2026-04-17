#!/usr/bin/env python3
"""Final debug - check exact transformation."""
import numpy as np
import cv2


def rotate_photo(photo, angle):
    """Rotate a photo by specified degrees."""
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo
    
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(
        photo, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(128, 128, 128, 0)
    )
    
    return rotated


def verify_canvas_offset():
    """Verify the canvas offset is handled correctly."""
    pw, ph = 300, 200
    angle = -45
    photo_center = (pw / 2, ph / 2)
    
    print("="*70)
    print("VERIFYING CANVAS OFFSET")
    print("="*70)
    
    # Build rotation matrix
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    print(f"\nOriginal size: {pw}x{ph}")
    print(f"Rotated size: {new_w}x{new_h}")
    
    # Canvas offset
    canvas_offset_x = (new_w - pw) / 2
    canvas_offset_y = (new_h - ph) / 2
    print(f"Canvas offset: ({canvas_offset_x}, {canvas_offset_y})")
    
    # Build full matrix (M_raw + canvas offset)
    M_full = M_raw.copy()
    M_full[0, 2] += canvas_offset_x
    M_full[1, 2] += canvas_offset_y
    print(f"\nM_raw:\n{M_raw}")
    print(f"\nM_full (with canvas offset):\n{M_full}")
    
    # Test: where does (150, 0) end up with M_raw and M_full?
    pt = np.array([150, 0, 1])
    raw_result = M_raw @ pt
    full_result = M_full @ pt
    print(f"\nPoint (150, 0):")
    print(f"  M_raw @ (150, 0) = ({raw_result[0]:.1f}, {raw_result[1]:.1f})")
    print(f"  M_full @ (150, 0) = ({full_result[0]:.1f}, {full_result[1]:.1f})")
    
    # Now test: where does (150, 0) end up with our polygon formula?
    # polygon formula: corner_canvas = canvas_offset + M_raw @ corner
    canvas_offset_tuple = (150 - pw/2, 150 - ph/2)  # cx - pw/2, cy - ph/2
    polygon_pos = (canvas_offset_tuple[0] + raw_result[0], canvas_offset_tuple[1] + raw_result[1])
    print(f"\nOur polygon formula:")
    print(f"  canvas_offset = ({canvas_offset_tuple[0]}, {canvas_offset_tuple[1]})")
    print(f"  polygon_pos = ({polygon_pos[0]:.1f}, {polygon_pos[1]:.1f})")
    
    # Test with actual photo
    print(f"\n" + "-"*50)
    print("ACTUAL TEST")
    print("-"*50)
    
    # Create photo with marker at (150, 0) - top center
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    # Single pixel marker at exact position
    photo[0, 150] = [0, 0, 255, 255]  # RED at (150, 0)
    
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    
    print(f"\nMarker at (150, 0) in source")
    print(f"Rotated image size: {rot_w}x{rot_h}")
    
    # Find the marker
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            px = int(M['m10'] / M['m00'])
            py = int(M['m01'] / M['m00'])
            print(f"Detected marker at ({px}, {py})")
            print(f"M_full predicted: ({full_result[0]:.1f}, {full_result[1]:.1f})")
            print(f"Difference: ({px - full_result[0]:.1f}, {py - full_result[1]:.1f})")
    
    # Now test composite
    print(f"\n" + "-"*50)
    print("COMPOSITE TEST")
    print("-"*50)
    
    cx, cy = 620, 620  # Canvas center
    
    # Canvas top-left = (cx - new_w/2, cy - new_h/2)
    canvas_top_left_x = cx - new_w / 2
    canvas_top_left_y = cy - new_h / 2
    print(f"Canvas center: ({cx}, {cy})")
    print(f"Rotated image size: {new_w}x{new_h}")
    print(f"Canvas top-left: ({canvas_top_left_x}, {canvas_top_left_y})")
    
    # Composite
    canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    result = canvas.copy()
    
    for y in range(rot_h):
        for x in range(rot_w):
            result[int(canvas_top_left_y) + y, int(canvas_top_left_x) + x] = rotated[y, x]
    
    # Find marker in composite
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            px = int(M['m10'] / M['m00'])
            py = int(M['m01'] / M['m00'])
            print(f"Detected in composite at ({px}, {py})")
            
            # Expected position: canvas_top_left + M_full @ (150, 0)
            expected = (canvas_top_left_x + full_result[0], canvas_top_left_y + full_result[1])
            print(f"Expected position: ({expected[0]:.1f}, {expected[1]:.1f})")
            print(f"Difference: ({px - expected[0]:.1f}, {py - expected[1]:.1f})")
    
    # Test polygon formula
    # polygon = canvas_offset + M_raw @ corner (where canvas_offset = (cx - pw/2, cy - ph/2))
    polygon_expected = (canvas_offset_tuple[0] + raw_result[0], canvas_offset_tuple[1] + raw_result[1])
    print(f"\nOur polygon formula prediction:")
    print(f"  canvas_offset = {canvas_offset_tuple}")
    print(f"  M_raw @ (150, 0) = {raw_result[:2]}")
    print(f"  polygon = {polygon_expected}")


if __name__ == '__main__':
    verify_canvas_offset()
