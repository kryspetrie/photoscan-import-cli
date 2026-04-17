#!/usr/bin/env python3
"""Direct test at 0° to verify basic polygon calculation."""
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


def test_0_degrees():
    """Test polygon at 0° rotation - the simplest case."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = 0
    
    print("="*70)
    print("TEST AT 0° ROTATION")
    print("="*70)
    
    # Create photo with markers at corners
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    ms = 10
    photo[0:ms, 0:ms] = [255, 0, 0, 255]           # Red at TL
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]       # Green at TR
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]   # Blue at BR
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]     # Yellow at BL
    
    # Rotate (should return same photo)
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    print(f"\nRotated size: {rot_w}x{rot_h}")
    print(f"(Should be same as original: {pw}x{ph})")
    
    # Composite
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
    
    # Detect corners
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red"),
        ([35, 100, 100], [85, 255, 255], "Green"),
        ([100, 100, 100], [130, 255, 255], "Blue"),
        ([15, 100, 100], [45, 255, 255], "Yellow")
    ]
    
    print(f"\nDetected corners:")
    detected = {}
    for i, (lower, upper, name) in enumerate(ranges):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                detected[i] = (px, py)
                print(f"  {i} ({name}): ({px}, {py})")
    
    # Calculate expected corners
    # At 0°, corners should be at:
    # TL: (cx - pw/2, cy - ph/2) = (620-150, 620-100) = (470, 520)
    # TR: (cx + pw/2, cy - ph/2) = (620+150, 620-100) = (770, 520)
    # BR: (cx + pw/2, cy + ph/2) = (620+150, 620+100) = (770, 720)
    # BL: (cx - pw/2, cy + ph/2) = (620-150, 620+100) = (470, 720)
    expected = {
        0: (int(cx - pw/2), int(cy - ph/2)),   # TL
        1: (int(cx + pw/2), int(cy - ph/2)),   # TR
        2: (int(cx + pw/2), int(cy + ph/2)),   # BR
        3: (int(cx - pw/2), int(cy + ph/2))    # BL
    }
    
    print(f"\nExpected corners:")
    for i, pos in expected.items():
        print(f"  {i}: {pos}")
    
    # Compare
    print(f"\nComparison:")
    total_error = 0
    for i in range(4):
        if i in detected:
            dx = detected[i][0] - expected[i][0]
            dy = detected[i][1] - expected[i][1]
            dist = np.sqrt(dx**2 + dy**2)
            total_error += dist
            print(f"  {i}: actual={detected[i]}, expected={expected[i]}, diff=({dx}, {dy}), dist={dist:.1f}px")
    
    print(f"\nTOTAL ERROR: {total_error:.1f}px, AVG: {total_error/4:.1f}px")
    
    if total_error/4 < 5:
        print("✅ PASS at 0°")
    else:
        print("❌ FAIL at 0°")


def test_polygon_formula():
    """Test the polygon calculation against the actual composite."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = -60
    
    print("\n" + "="*70)
    print(f"TEST POLYGON FORMULA AT {angle}°")
    print("="*70)
    
    # The CORRECT formula should be:
    # 1. Rotate corner around photo center: rotated = M_raw @ corner
    # 2. Place on canvas at (cx, cy) as center: canvas_pos = (cx - new_w/2, cy - new_h/2) + rotated
    
    photo_center = (pw / 2, ph / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    canvas_top_left_x = cx - new_w / 2
    canvas_top_left_y = cy - new_h / 2
    
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    polygon = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        polygon[i, 0] = canvas_top_left_x + rotated[0]
        polygon[i, 1] = canvas_top_left_y + rotated[1]
    
    print(f"\nCalculated polygon corners:")
    for i in range(4):
        print(f"  {i}: ({polygon[i][0]:.1f}, {polygon[i][1]:.1f})")
    
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
    
    # Detect
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red"),
        ([35, 100, 100], [85, 255, 255], "Green"),
        ([100, 100, 100], [130, 255, 255], "Blue"),
        ([15, 100, 100], [45, 255, 255], "Yellow")
    ]
    
    print(f"\nDetected corners:")
    detected = {}
    for i, (lower, upper, name) in enumerate(ranges):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                detected[i] = (px, py)
                print(f"  {i} ({name}): ({px}, {py})")
    
    # Compare by color
    print(f"\nComparison by color index:")
    total_error = 0
    for i in range(4):
        if i in detected:
            dx = detected[i][0] - polygon[i][0]
            dy = detected[i][1] - polygon[i][1]
            dist = np.sqrt(dx**2 + dy**2)
            total_error += dist
            print(f"  {i}: actual={detected[i]}, expected=({polygon[i][0]:.1f}, {polygon[i][1]:.1f}), error={dist:.1f}px")
    
    print(f"\nTOTAL ERROR: {total_error:.1f}px, AVG: {total_error/4:.1f}px")
    
    if total_error/4 < 5:
        print("✅ PASS")
    else:
        print("❌ FAIL")


if __name__ == '__main__':
    test_0_degrees()
    test_polygon_formula()
