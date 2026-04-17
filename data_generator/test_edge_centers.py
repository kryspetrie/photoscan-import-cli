#!/usr/bin/env python3
"""Direct test where we KNOW where markers should appear."""
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


def get_rotation_matrix(pw, ph, rotation):
    """Get rotation matrix (raw, without canvas offset)."""
    photo_center = (pw / 2, ph / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    return M_raw


def test_with_centered_markers():
    """Test with markers at the CENTER of each edge, not at corners.
    
    This tests whether the polygon calculation is correct for edge midpoints,
    which should help us understand if corners are also correct.
    """
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = -45
    
    print("="*70)
    print(f"TEST WITH CENTERED EDGE MARKERS AT {angle}°")
    print("="*70)
    
    # Create photo with markers at the CENTER of each edge
    photo = np.ones((ph, pw, 4), dtype=np.uint8) * 180
    photo[:, :, 3] = 255
    
    ms = 20  # marker size
    
    # TOP edge center: at (pw/2, 0) = (150, 0)
    photo[0:ms, pw//2-ms//2:pw//2+ms//2] = [0, 0, 255, 255]  # RED
    
    # BOTTOM edge center: at (pw/2, ph) = (150, 200)
    photo[ph-ms:ph, pw//2-ms//2:pw//2+ms//2] = [0, 255, 0, 255]  # GREEN
    
    # LEFT edge center: at (0, ph/2) = (0, 100)
    photo[ph//2-ms//2:ph//2+ms//2, 0:ms] = [255, 0, 0, 255]  # BLUE
    
    # RIGHT edge center: at (pw, ph/2) = (300, 100)
    photo[ph//2-ms//2:ph//2+ms//2, pw-ms:pw] = [0, 255, 255, 255]  # YELLOW
    
    # Rotate
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    print(f"\nRotated size: {rot_w}x{rot_h}")
    
    # Build rotation matrix
    M_raw = get_rotation_matrix(pw, ph, angle)
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    # Canvas top-left
    canvas_top_left_x = cx - new_w / 2
    canvas_top_left_y = cy - new_h / 2
    
    print(f"Canvas top-left: ({canvas_top_left_x}, {canvas_top_left_y})")
    
    # Composite
    canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    result = canvas.copy()
    
    for y in range(rot_h):
        for x in range(rot_w):
            result[int(canvas_top_left_y) + y, int(canvas_top_left_x) + x] = rotated[y, x]
    
    # Expected positions: transform edge centers through rotation matrix
    # Edge centers in photo space: (150, 0), (150, 200), (0, 100), (300, 100)
    edge_centers = [
        (pw/2, 0, "TOP (RED)"),
        (pw/2, ph, "BOTTOM (GREEN)"),
        (0, ph/2, "LEFT (BLUE)"),
        (pw, ph/2, "RIGHT (YELLOW)")
    ]
    
    print(f"\nExpected edge center positions (M_raw @ center + canvas_offset):")
    for ex, ey, name in edge_centers:
        pt = np.array([ex, ey, 1])
        rotated_pos = M_raw @ pt
        canvas_x = canvas_top_left_x + rotated_pos[0]
        canvas_y = canvas_top_left_y + rotated_pos[1]
        print(f"  {name} ({ex}, {ey}) -> ({canvas_x:.1f}, {canvas_y:.1f})")
    
    # Detect markers
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    color_ranges = [
        ([0, 100, 100], [15, 255, 255], "TOP (RED)"),
        ([35, 100, 100], [85, 255, 255], "BOTTOM (GREEN)"),
        ([100, 100, 100], [130, 255, 255], "LEFT (BLUE)"),
        ([15, 100, 100], [45, 255, 255], "RIGHT (YELLOW)")
    ]
    
    detected = []
    print(f"\nDetected edge centers:")
    for lower, upper, name in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                detected.append((px, py))
                print(f"  {name}: ({px}, {py})")
    
    # Calculate expected positions
    expected = []
    for ex, ey, name in edge_centers:
        pt = np.array([ex, ey, 1])
        rotated_pos = M_raw @ pt
        canvas_x = canvas_top_left_x + rotated_pos[0]
        canvas_y = canvas_top_left_y + rotated_pos[1]
        expected.append((canvas_x, canvas_y))
    
    # Match by closest
    matches = []
    used = set()
    for i, exp in enumerate(expected):
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
    
    print(f"\nErrors:")
    total = 0
    for exp_i, det_i, dist in matches:
        print(f"  Edge {exp_i}: detected={detected[det_i]}, expected=({expected[exp_i][0]:.1f}, {expected[exp_i][1]:.1f}), error={dist:.2f}px")
        total += dist
    print(f"  TOTAL: {total:.2f}px, AVG: {total/4:.2f}px")
    
    if total/4 < 5:
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL")


if __name__ == '__main__':
    test_with_centered_markers()
