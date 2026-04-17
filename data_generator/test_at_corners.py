#!/usr/bin/env python3
"""Test with markers AT THE ACTUAL CORNERS."""
import numpy as np
import cv2


def rotate_photo(photo, angle):
    """Rotate a photo by specified degrees."""
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo, None
    
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
    
    return rotated, M, (new_w, new_h)


def get_polygon_simple(pw, ph, center_x, center_y, rotation):
    """Calculate where corners end up using M_inv approach.
    
    The key insight: warpAffine uses the inverse transform internally.
    For each pixel Q in the destination, it samples from M_inv @ Q in source.
    
    So for a corner at source position C, it appears at destination position D
    where D satisfies: C = M_inv @ D, or equivalently D = M @ C.
    
    To compute D: D = M @ C
    """
    if abs(rotation) < 1:
        return np.array([
            [center_x - pw/2, center_y - ph/2],
            [center_x + pw/2, center_y - ph/2],
            [center_x + pw/2, center_y + ph/2],
            [center_x - pw/2, center_y + ph/2]
        ], dtype=np.float32)
    
    # Build forward transform matrix (same as rotate_photo)
    photo_center = (pw / 2, ph / 2)
    M_forward = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M_forward[0, 0])
    sin_a = abs(M_forward[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    M_forward[0, 2] += (new_w - pw) / 2
    M_forward[1, 2] += (new_h - ph) / 2
    
    # Corners in PHOTO SPACE
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    # Transform corners: D = M_forward @ C
    corners_rotated = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        result = M_forward @ pt
        corners_rotated[i] = [result[0], result[1]]
    
    # Canvas top-left = (center_x - new_w/2, center_y - new_h/2)
    canvas_top_left_x = center_x - new_w / 2
    canvas_top_left_y = center_y - new_h / 2
    
    # Final position: canvas_top_left + (rotated_position - [new_w/2, new_h/2])
    # Wait, that's not right. Let me think again.
    
    # Actually, M_forward transforms from photo space to rotated image space.
    # The rotated image is placed on canvas with top-left at (canvas_top_left).
    # So corner position on canvas = canvas_top_left + corner position in rotated image
    corners = np.zeros_like(corners_rotated)
    corners[:, 0] = canvas_top_left_x + corners_rotated[:, 0]
    corners[:, 1] = canvas_top_left_y + corners_rotated[:, 1]
    
    return corners


def test_at_corners():
    """Test with markers placed AT THE ACTUAL CORNERS."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = -60
    
    print("="*70)
    print("TEST WITH MARKERS AT ACTUAL CORNERS")
    print("="*70)
    
    # Create photo with markers AT THE CORNERS
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    ms = 10  # Small markers at corners
    # TL corner (0,0)
    photo[0:ms, 0:ms] = [255, 0, 0, 255]           # Red
    # TR corner (pw, 0)
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]       # Green
    # BR corner (pw, ph)
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]   # Blue
    # BL corner (0, ph)
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]     # Yellow
    
    # Rotate
    rotated, M, (rot_w, rot_h) = rotate_photo(photo, angle)
    
    print(f"\nPhoto: {pw}x{ph}")
    print(f"Rotated: {rot_w}x{rot_h}")
    
    # Detect corners in rotated image
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red", 0),
        ([35, 100, 100], [85, 255, 255], "Green", 1),
        ([100, 100, 100], [130, 255, 255], "Blue", 2),
        ([15, 100, 100], [45, 255, 255], "Yellow", 3)
    ]
    
    print(f"\nDetected in rotated image:")
    detected = {}
    for (lower, upper, name, idx) in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M_m = cv2.moments(largest)
            if M_m['m00'] > 0:
                px = int(M_m['m10'] / M_m['m00'])
                py = int(M_m['m01'] / M_m['m00'])
                detected[idx] = (px, py)
                print(f"  {name} (corner {idx}): ({px}, {py})")
    
    # Calculate expected corner positions
    print(f"\nExpected corner positions (M @ corner):")
    for i in range(4):
        pt = np.array([i * pw if i in [1, 2] else 0, ph if i in [2, 3] else 0, 1])
        result = M @ pt
        print(f"  Corner {i}: ({result[0]:.1f}, {result[1]:.1f})")
    
    # Calculate polygon
    polygon = get_polygon_simple(pw, ph, cx, cy, angle)
    print(f"\nPolygon calculation (after placement at {cx},{cy}):")
    for i in range(4):
        print(f"  Corner {i}: ({polygon[i][0]:.1f}, {polygon[i][1]:.1f})")


def test_against_actual_composite():
    """Test polygon calculation against actual composite."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = -60
    
    print("\n" + "="*70)
    print("TEST AGAINST ACTUAL COMPOSITE")
    print("="*70)
    
    # Create photo with markers AT THE CORNERS
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    ms = 10
    photo[0:ms, 0:ms] = [255, 0, 0, 255]
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]
    
    # Rotate
    rotated, M, (rot_w, rot_h) = rotate_photo(photo, angle)
    
    # Composite onto canvas
    canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    
    top_left_x = int(cx - rot_w / 2)
    top_left_y = int(cy - rot_h / 2)
    
    result = canvas.copy()
    for y in range(rot_h):
        for x in range(rot_w):
            src_px = rotated[y, x]
            dst_x = top_left_x + x
            dst_y = top_left_y + y
            if 0 <= dst_x < 1240 and 0 <= dst_y < 1240:
                result[dst_y, dst_x] = src_px
    
    # Detect in composite
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red"),
        ([35, 100, 100], [85, 255, 255], "Green"),
        ([100, 100, 100], [130, 255, 255], "Blue"),
        ([15, 100, 100], [45, 255, 255], "Yellow")
    ]
    
    print(f"\nDetected in composite:")
    actual_composite = {}
    for i, (lower, upper, name) in enumerate(ranges):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M_m = cv2.moments(largest)
            if M_m['m00'] > 0:
                px = int(M_m['m10'] / M_m['m00'])
                py = int(M_m['m01'] / M_m['m00'])
                actual_composite[i] = (px, py)
                print(f"  {i} ({name}): ({px}, {py})")
    
    # Calculate expected
    polygon = get_polygon_simple(pw, ph, cx, cy, angle)
    print(f"\nExpected (polygon calculation):")
    for i in range(4):
        print(f"  {i}: ({polygon[i][0]:.1f}, {polygon[i][1]:.1f})")
    
    # Compare
    print(f"\nComparison (by color index):")
    total_error = 0
    for i in range(4):
        if i in actual_composite:
            dx = actual_composite[i][0] - polygon[i][0]
            dy = actual_composite[i][1] - polygon[i][1]
            dist = np.sqrt(dx**2 + dy**2)
            total_error += dist
            print(f"  {i}: actual={actual_composite[i]}, expected=({polygon[i][0]:.1f}, {polygon[i][1]:.1f}), error={dist:.1f}px")
    
    print(f"\nTOTAL ERROR: {total_error:.1f}px, AVG: {total_error/4:.1f}px")
    
    if total_error/4 < 5:
        print("✅ PASS: Average error < 5px")
    else:
        print("❌ FAIL: Average error >= 5px")


if __name__ == '__main__':
    test_at_corners()
    test_against_actual_composite()
