#!/usr/bin/env python3
"""Direct comparison of polygon calculation vs actual marker positions."""
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


def get_polygon(pw, ph, center_x, center_y, rotation):
    """Calculate polygon using the correct formula."""
    if abs(rotation) < 1:
        return np.array([
            [center_x - pw/2, center_y - ph/2],
            [center_x + pw/2, center_y - ph/2],
            [center_x + pw/2, center_y + ph/2],
            [center_x - pw/2, center_y + ph/2]
        ], dtype=np.float32)
    
    photo_center = (pw / 2, ph / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    canvas_offset = (center_x - pw/2, center_y - ph/2)
    
    corners_photo = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)
    polygon = np.zeros_like(corners_photo)
    
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        polygon[i, 0] = rotated[0] + canvas_offset[0]
        polygon[i, 1] = rotated[1] + canvas_offset[1]
    
    return polygon


def test_0_degrees():
    """Test at 0° where we know the exact positions."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    
    print("="*70)
    print("TEST AT 0° - VERIFYING POLYGON vs ACTUAL COMPOSITE")
    print("="*70)
    
    # Create photo with markers at EXACT corners
    photo = np.ones((ph, pw, 4), dtype=np.uint8) * 180
    photo[:, :, 3] = 255
    
    # Markers at the EXACT corners (1 pixel wide)
    photo[0, 0] = [0, 0, 255, 255]     # TL corner - RED
    photo[0, pw-1] = [0, 255, 0, 255]  # TR corner - GREEN
    photo[ph-1, pw-1] = [255, 0, 0, 255]  # BR corner - BLUE
    photo[ph-1, 0] = [0, 255, 255, 255]  # BL corner - YELLOW
    
    # Rotate (should return same photo)
    rotated = rotate_photo(photo, 0)
    
    # Composite
    rot_h, rot_w = rotated.shape[:2]
    canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    
    top_left_x = int(cx - rot_w / 2)
    top_left_y = int(cy - rot_h / 2)
    
    print(f"\nRotated size: {rot_w}x{rot_h}")
    print(f"Canvas top-left: ({top_left_x}, {top_left_y})")
    print(f"Expected corner positions: TL({top_left_x}, {top_left_y}), TR({top_left_x+pw}, {top_left_y}), BR({top_left_x+pw}, {top_left_y+ph}), BL({top_left_x}, {top_left_y+ph})")
    
    result = canvas.copy()
    for y in range(rot_h):
        for x in range(rot_w):
            result[top_left_y + y, top_left_x + x] = rotated[y, x]
    
    # Calculate polygon
    polygon = get_polygon(pw, ph, cx, cy, 0)
    print(f"\nCalculated polygon:")
    for i, c in enumerate(polygon):
        print(f"  {i}: ({c[0]:.1f}, {c[1]:.1f})")
    
    # Find the actual corner pixels in composite
    print(f"\nActual corner pixel values in composite:")
    tl = result[top_left_y, top_left_x]
    tr = result[top_left_y, top_left_x + pw - 1]
    br = result[top_left_y + ph - 1, top_left_x + pw - 1]
    bl = result[top_left_y + ph - 1, top_left_x]
    print(f"  TL ({top_left_x}, {top_left_y}): {tl}")
    print(f"  TR ({top_left_x+pw-1}, {top_left_y}): {tr}")
    print(f"  BR ({top_left_x+pw-1}, {top_left_y+ph-1}): {br}")
    print(f"  BL ({top_left_x}, {top_left_y+ph-1}): {bl}")
    
    # Compare
    print(f"\nComparison:")
    expected = [
        (top_left_x, top_left_y),  # TL
        (top_left_x + pw - 1, top_left_y),  # TR
        (top_left_x + pw - 1, top_left_y + ph - 1),  # BR
        (top_left_x, top_left_y + ph - 1)  # BL
    ]
    for i in range(4):
        poly = polygon[i]
        exp = expected[i]
        print(f"  {i}: polygon=({poly[0]:.1f}, {poly[1]:.1f}), expected=({exp[0]}, {exp[1]})")


def test_with_large_markers():
    """Test with markers that are easier to detect."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = 0
    
    print("\n" + "="*70)
    print("TEST WITH LARGE MARKERS AT 0°")
    print("="*70)
    
    # Create photo with large markers at corners
    photo = np.ones((ph, pw, 4), dtype=np.uint8) * 180
    photo[:, :, 3] = 255
    
    ms = 20
    photo[0:ms, 0:ms] = [0, 0, 255, 255]           # TL - RED
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]       # TR - GREEN
    photo[ph-ms:ph, pw-ms:pw] = [255, 0, 0, 255]   # BR - BLUE
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]      # BL - YELLOW
    
    rotated = rotate_photo(photo, angle)
    
    # Composite
    rot_h, rot_w = rotated.shape[:2]
    canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    
    top_left_x = cx - rot_w // 2
    top_left_y = cy - rot_h // 2
    
    result = canvas.copy()
    for y in range(rot_h):
        for x in range(rot_w):
            result[top_left_y + y, top_left_x + x] = rotated[y, x]
    
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
                detected.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))
    
    print(f"\nDetected corners:")
    for i, d in enumerate(detected):
        print(f"  {i}: {d}")
    
    # Calculate polygon
    polygon = get_polygon(pw, ph, cx, cy, angle)
    print(f"\nCalculated polygon:")
    for i, c in enumerate(polygon):
        print(f"  {i}: ({c[0]:.1f}, {c[1]:.1f})")
    
    # Match by position
    matches = []
    used = set()
    for i, exp in enumerate(polygon):
        best_j, best_dist = None, float('inf')
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
    
    print(f"\nErrors (by position matching):")
    total = 0
    for exp_i, det_i, dist in matches:
        print(f"  {exp_i}: detected={detected[det_i]}, expected=({polygon[exp_i][0]:.1f}, {polygon[exp_i][1]:.1f}), error={dist:.2f}px")
        total += dist
    print(f"  TOTAL: {total:.2f}px, AVG: {total/4:.2f}px")


if __name__ == '__main__':
    test_0_degrees()
    test_with_large_markers()
