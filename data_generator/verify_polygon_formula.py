#!/usr/bin/env python3
"""Verify the full transformation chain."""
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


def test_polygon_formula():
    """Test the polygon formula against actual composite."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = -45
    
    print("="*70)
    print(f"TESTING POLYGON FORMULA AT {angle}°")
    print("="*70)
    
    # Build matrices
    photo_center = (pw / 2, ph / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    # Canvas offset (for polygon formula)
    canvas_offset = (cx - pw/2, cy - ph/2)
    
    print(f"\nPhoto center: {photo_center}")
    print(f"Rotated dimensions: {new_w}x{new_h}")
    print(f"Canvas top-left (for composite): ({cx - new_w/2}, {cy - new_h/2})")
    
    # Create photo with marker at (150, 0)
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    photo[0:5, 148:153] = [0, 0, 255, 255]  # RED at (150, 0)
    
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    
    print(f"\nRotated image: {rot_w}x{rot_h}")
    
    # Find marker in rotated image
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_in_rotated = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            marker_in_rotated = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            print(f"Marker in rotated image: {marker_in_rotated}")
    
    # Composite
    canvas_top_left_x = cx - new_w / 2
    canvas_top_left_y = cy - new_h / 2
    
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
    marker_in_composite = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            marker_in_composite = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            print(f"Marker in composite: {marker_in_composite}")
    
    # Calculate polygon position
    pt = np.array([150, 0, 1])
    rotated_pos = M_raw @ pt
    polygon_pos = (canvas_offset[0] + rotated_pos[0], canvas_offset[1] + rotated_pos[1])
    print(f"\nPolygon calculation:")
    print(f"  canvas_offset = {canvas_offset}")
    print(f"  M_raw @ (150, 0) = {rotated_pos[:2]}")
    print(f"  polygon_pos = {polygon_pos}")
    
    # Verify: composite position should be canvas_top_left + marker_in_rotated
    expected_composite = (canvas_top_left_x + marker_in_rotated[0], canvas_top_left_y + marker_in_rotated[1])
    print(f"\nVerification:")
    print(f"  canvas_top_left + marker_in_rotated = {expected_composite}")
    print(f"  marker_in_composite = {marker_in_composite}")
    
    if marker_in_composite:
        diff = (marker_in_composite[0] - expected_composite[0], marker_in_composite[1] - expected_composite[1])
        print(f"  Difference: {diff}")
        
        # The polygon should match marker_in_composite
        print(f"\nFinal comparison:")
        print(f"  polygon_pos = {polygon_pos}")
        print(f"  marker_in_composite = {marker_in_composite}")
        print(f"  Difference: ({marker_in_composite[0] - polygon_pos[0]:.2f}, {marker_in_composite[1] - polygon_pos[1]:.2f})")


def test_with_all_corners():
    """Test with markers at all 4 corners."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = -45
    
    print("\n" + "="*70)
    print(f"TEST WITH ALL CORNER MARKERS AT {angle}°")
    print("="*70)
    
    # Build matrices
    photo_center = (pw / 2, ph / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    # Canvas offset
    canvas_offset = (cx - pw/2, cy - ph/2)
    
    # Canvas top-left for composite
    canvas_top_left_x = cx - new_w / 2
    canvas_top_left_y = cy - new_h / 2
    
    # Create photo with corner markers
    photo = np.ones((ph, pw, 4), dtype=np.uint8) * 180
    photo[:, :, 3] = 255
    
    ms = 15
    # TL (0, 0)
    photo[0:ms, 0:ms] = [0, 0, 255, 255]
    # TR (300, 0)
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]
    # BR (300, 200)
    photo[ph-ms:ph, pw-ms:pw] = [255, 0, 0, 255]
    # BL (0, 200)
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]
    
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    
    # Composite
    canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    result = canvas.copy()
    
    for y in range(rot_h):
        for x in range(rot_w):
            result[int(canvas_top_left_y) + y, int(canvas_top_left_x) + x] = rotated[y, x]
    
    # Detect corners
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    color_ranges = [
        ([0, 100, 100], [15, 255, 255]),
        ([35, 100, 100], [85, 255, 255]),
        ([100, 100, 100], [130, 255, 255]),
        ([15, 100, 100], [45, 255, 255])
    ]
    
    detected = []
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                detected.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))
    
    # Calculate expected positions
    corners_photo = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)
    polygon = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated_pos = M_raw @ pt
        polygon[i, 0] = canvas_offset[0] + rotated_pos[0]
        polygon[i, 1] = canvas_offset[1] + rotated_pos[1]
    
    print(f"\nExpected polygon positions:")
    for i in range(4):
        print(f"  {i}: ({polygon[i, 0]:.1f}, {polygon[i, 1]:.1f})")
    
    print(f"\nDetected positions:")
    for i, d in enumerate(detected):
        print(f"  {i}: {d}")
    
    # Match by position
    matches = []
    used = set()
    for i in range(4):
        best_j, best_dist = None, float('inf')
        for j in range(len(detected)):
            if j in used:
                continue
            dist = np.sqrt((polygon[i, 0] - detected[j][0])**2 + (polygon[i, 1] - detected[j][1])**2)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            matches.append((i, best_j, best_dist))
            used.add(best_j)
    
    print(f"\nErrors:")
    total = 0
    for exp_i, det_i, dist in matches:
        print(f"  Corner {exp_i}: detected={detected[det_i]}, expected=({polygon[exp_i, 0]:.1f}, {polygon[exp_i, 1]:.1f}), error={dist:.2f}px")
        total += dist
    print(f"  TOTAL: {total:.2f}px, AVG: {total/4:.2f}px")


if __name__ == '__main__':
    test_polygon_formula()
    test_with_all_corners()
