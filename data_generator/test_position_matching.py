#!/usr/bin/env python3
"""Test with position-based matching instead of color-based."""
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


def match_by_position(expected, detected):
    """Match detected corners to expected by closest position."""
    matches = []
    used = set()
    
    for i, exp in enumerate(expected):
        best_j = None
        best_dist = float('inf')
        for j, det in enumerate(detected):
            if j in used:
                continue
            dist = np.sqrt((exp[0] - det[0])**2 + (exp[1] - det[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            matches.append((i, best_j, best_dist))
            used.add(best_j)
    
    return matches


def test_polygon_formula():
    """Test polygon formula with position-based matching."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = -60
    
    print("="*70)
    print(f"TEST POLYGON FORMULA AT {angle}°")
    print("="*70)
    
    # Build rotation matrix
    photo_center = (pw / 2, ph / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    canvas_top_left_x = cx - new_w / 2
    canvas_top_left_y = cy - new_h / 2
    
    # Corners in PHOTO SPACE: TL(0), TR(1), BR(2), BL(3)
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    # Calculate expected polygon positions
    polygon = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        polygon[i, 0] = canvas_top_left_x + rotated[0]
        polygon[i, 1] = canvas_top_left_y + rotated[1]
    
    print(f"\nExpected polygon (corners in canvas space):")
    for i, c in enumerate(polygon):
        print(f"  {i}: ({c[0]:.1f}, {c[1]:.1f})")
    
    # Create photo and composite
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    ms = 10
    photo[0:ms, 0:ms] = [255, 0, 0, 255]
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]
    
    rotated = rotate_photo(photo, angle)
    
    canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    
    result = canvas.copy()
    rot_h, rot_w = rotated.shape[:2]
    for y in range(rot_h):
        for x in range(rot_w):
            src_px = rotated[y, x]
            dst_x = int(canvas_top_left_x) + x
            dst_y = int(canvas_top_left_y) + y
            if 0 <= dst_x < 1240 and 0 <= dst_y < 1240:
                result[dst_y, dst_x] = src_px
    
    # Detect corners by color
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    color_names = ["Red", "Green", "Blue", "Yellow"]
    color_ranges = [
        ([0, 100, 100], [15, 255, 255]),
        ([35, 100, 100], [85, 255, 255]),
        ([100, 100, 100], [130, 255, 255]),
        ([15, 100, 100], [45, 255, 255])
    ]
    
    detected = []
    print(f"\nDetected corners (by color):")
    for i, ((lower, upper), name) in enumerate(zip(color_ranges, color_names)):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                detected.append((px, py))
                print(f"  {i} ({name}): ({px}, {py})")
    
    if len(detected) != 4:
        print(f"  WARNING: Only detected {len(detected)} corners!")
        return
    
    # Match detected to expected by position
    print(f"\nMatching by closest position:")
    matches = match_by_position(polygon, detected)
    
    total_error = 0
    for exp_i, det_i, dist in matches:
        det = detected[det_i]
        exp = polygon[exp_i]
        print(f"  Polygon[{exp_i}] -> Detected[{det_i}]: dist={dist:.1f}px (expected=({exp[0]:.1f},{exp[1]:.1f}), actual={det})")
        total_error += dist
    
    print(f"\nTOTAL ERROR: {total_error:.1f}px, AVG: {total_error/4:.1f}px")
    
    if total_error/4 < 5:
        print("✅ PASS: Average error < 5px")
    else:
        print("❌ FAIL: Average error >= 5px")


def test_multiple_angles():
    """Test polygon formula across multiple angles."""
    print("\n" + "="*70)
    print("TEST ACROSS MULTIPLE ANGLES")
    print("="*70)
    
    pw, ph = 300, 200
    cx, cy = 620, 620
    
    results = []
    
    for angle in range(-60, 91, 15):
        # Build rotation matrix
        photo_center = (pw / 2, ph / 2)
        M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
        
        cos_a = abs(M_raw[0, 0])
        sin_a = abs(M_raw[0, 1])
        new_w = int(ph * sin_a + pw * cos_a)
        new_h = int(ph * cos_a + pw * sin_a)
        
        canvas_top_left_x = cx - new_w / 2
        canvas_top_left_y = cy - new_h / 2
        
        # Corners in PHOTO SPACE
        corners_photo = np.array([
            [0, 0], [pw, 0], [pw, ph], [0, ph]
        ], dtype=np.float32)
        
        # Calculate expected polygon positions
        polygon = np.zeros_like(corners_photo)
        for i in range(4):
            pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
            rotated = M_raw @ pt
            polygon[i, 0] = canvas_top_left_x + rotated[0]
            polygon[i, 1] = canvas_top_left_y + rotated[1]
        
        # Create photo and composite
        photo = np.zeros((ph, pw, 4), dtype=np.uint8)
        photo[:, :, :3] = 180
        photo[:, :, 3] = 255
        
        ms = 10
        photo[0:ms, 0:ms] = [255, 0, 0, 255]
        photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]
        photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]
        photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]
        
        rotated = rotate_photo(photo, angle)
        
        canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
        canvas[:, :, 3] = 0
        
        result = canvas.copy()
        rot_h, rot_w = rotated.shape[:2]
        for y in range(rot_h):
            for x in range(rot_w):
                src_px = rotated[y, x]
                dst_x = int(canvas_top_left_x) + x
                dst_y = int(canvas_top_left_y) + y
                if 0 <= dst_x < 1240 and 0 <= dst_y < 1240:
                    result[dst_y, dst_x] = src_px
        
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
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    px = int(M['m10'] / M['m00'])
                    py = int(M['m01'] / M['m00'])
                    detected.append((px, py))
        
        if len(detected) != 4:
            results.append((angle, None, None))
            continue
        
        # Match by position
        matches = match_by_position(polygon, detected)
        total_error = sum(m[2] for m in matches)
        avg_error = total_error / 4
        max_error = max(m[2] for m in matches)
        
        results.append((angle, avg_error, max_error))
        
        status = "✅" if avg_error < 5 else "⚠️" if avg_error < 10 else "❌"
        print(f"  {angle:>4}°: avg={avg_error:.2f}px, max={max_error:.2f}px {status}")
    
    # Summary
    print("\n" + "-"*50)
    valid = [(a, e, m) for a, e, m in results if e is not None]
    if valid:
        avg_all = sum(e for _, e, _ in valid) / len(valid)
        max_all = max(m for _, _, m in valid)
        print(f"Overall: avg={avg_all:.2f}px, max={max_all:.2f}px")
        
        if avg_all < 5:
            print("✅ PASS: All tests < 5px average")
        else:
            print("❌ FAIL: Some tests >= 5px average")


if __name__ == '__main__':
    test_polygon_formula()
    test_multiple_angles()
