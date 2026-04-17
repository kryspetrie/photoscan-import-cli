#!/usr/bin/env python3
"""CORRECT polygon formula based on transformation chain analysis."""
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


def CORRECT_get_rotated_polygon(pw, ph, center_x, center_y, rotation):
    """CORRECT polygon calculation.
    
    Transformation chain:
    1. Source pixel S at (x, y)
    2. After M_raw @ S: position relative to photo center
    3. M = M_raw with canvas offset added
    4. M @ S = M_raw @ S + [(new_w-pw)/2, (new_h-ph)/2]
    5. Rotated image pixel = M @ S
    6. Canvas position = canvas_top_left + rotated_image_pixel
       where canvas_top_left = (center_x - new_w/2, center_y - new_h/2)
    7. Final: corner_canvas = (center_x - new_w/2, center_y - new_h/2) + M @ S
    
    Simplifying:
    corner_canvas = (center_x - new_w/2, center_y - new_h/2) + M_raw @ S + [(new_w-pw)/2, (new_h-ph)/2]
    corner_canvas = M_raw @ S + (center_x - pw/2, center_y - ph/2)
    
    Where M_raw is the rotation matrix WITHOUT canvas offset!
    """
    if abs(rotation) < 1:
        return np.array([
            [center_x - pw/2, center_y - ph/2],
            [center_x + pw/2, center_y - ph/2],
            [center_x + pw/2, center_y + ph/2],
            [center_x - pw/2, center_y + ph/2]
        ], dtype=np.float32)
    
    photo_center = (pw / 2, ph / 2)
    
    # M_raw: rotation matrix WITHOUT canvas offset
    M_raw = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    # Canvas offset for placement
    canvas_offset = (center_x - pw/2, center_y - ph/2)
    
    # Corners in PHOTO SPACE: TL(0), TR(1), BR(2), BL(3)
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    # Calculate corner positions on canvas
    polygon = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt  # M_raw gives rotation around photo center
        polygon[i, 0] = rotated[0] + canvas_offset[0]
        polygon[i, 1] = rotated[1] + canvas_offset[1]
    
    return polygon


def OLD_get_rotated_polygon(pw, ph, center_x, center_y, rotation):
    """OLD (WRONG) polygon calculation for comparison."""
    if abs(rotation) < 1:
        return np.array([
            [center_x - pw/2, center_y - ph/2],
            [center_x + pw/2, center_y - ph/2],
            [center_x + pw/2, center_y + ph/2],
            [center_x - pw/2, center_y + ph/2]
        ], dtype=np.float32)
    
    photo_center = (pw / 2, ph / 2)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    M[0, 2] += (new_w - pw) / 2
    M[1, 2] += (new_h - ph) / 2
    
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    corners_rot = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        result = M @ pt
        corners_rot[i] = [result[0], result[1]]
    
    rotated_center = M @ np.array([photo_center[0], photo_center[1], 1])
    
    corners = np.zeros_like(corners_rot)
    corners[:, 0] = center_x - rotated_center[0] + corners_rot[:, 0]
    corners[:, 1] = center_y - rotated_center[1] + corners_rot[:, 1]
    
    return corners


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


def create_photo_with_correct_colors(w, h):
    """Create photo with markers at corners."""
    photo = np.ones((h, w, 4), dtype=np.uint8) * 180
    photo[:, :, 3] = 255
    
    ms = 15
    photo[0:ms, 0:ms] = [0, 0, 255, 255]           # TL: RED
    photo[0:ms, w-ms:w] = [0, 255, 0, 255]        # TR: GREEN
    photo[h-ms:h, w-ms:w] = [255, 0, 0, 255]      # BR: BLUE
    photo[h-ms:h, 0:ms] = [0, 255, 255, 255]      # BL: YELLOW
    
    return photo


def test_all_angles():
    """Test both formulas across all angles."""
    print("="*70)
    print("COMPARING OLD vs CORRECT POLYGON FORMULAS")
    print("="*70)
    
    pw, ph = 300, 200
    cx, cy = 620, 620
    color_ranges = [
        ([0, 100, 100], [15, 255, 255]),
        ([35, 100, 100], [85, 255, 255]),
        ([100, 100, 100], [130, 255, 255]),
        ([15, 100, 100], [45, 255, 255])
    ]
    
    print(f"\n{'Angle':>6} | {'OLD':>10} | {'CORRECT':>10}")
    print("-" * 35)
    
    old_results = []
    correct_results = []
    
    for angle in range(-60, 91, 15):
        # Calculate polygons
        old_polygon = OLD_get_rotated_polygon(pw, ph, cx, cy, angle)
        correct_polygon = CORRECT_get_rotated_polygon(pw, ph, cx, cy, angle)
        
        # Create and composite photo
        photo = create_photo_with_correct_colors(pw, ph)
        rotated = rotate_photo(photo, angle)
        
        # Get rotated dimensions
        rot_h, rot_w = rotated.shape[:2]
        
        # Build composite
        photo_center = (pw / 2, ph / 2)
        M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
        cos_a = abs(M_raw[0, 0])
        sin_a = abs(M_raw[0, 1])
        new_w = int(ph * sin_a + pw * cos_a)
        new_h = int(ph * cos_a + pw * sin_a)
        canvas_top_left_x = cx - new_w / 2
        canvas_top_left_y = cy - new_h / 2
        
        canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
        canvas[:, :, 3] = 0
        result = canvas.copy()
        for y in range(rot_h):
            for x in range(rot_w):
                src_px = rotated[y, x]
                dst_x = int(canvas_top_left_x) + x
                dst_y = int(canvas_top_left_y) + y
                if 0 <= dst_x < 1240 and 0 <= dst_y < 1240:
                    result[dst_y, dst_x] = src_px
        
        # Detect corners
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        detected = []
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 50:
                    M = cv2.moments(largest)
                    if M['m00'] > 0:
                        detected.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))
        
        if len(detected) < 4:
            continue
        
        # Calculate errors for both formulas
        old_matches = match_by_position(old_polygon, detected)
        correct_matches = match_by_position(correct_polygon, detected)
        
        old_avg = sum(m[2] for m in old_matches) / 4
        correct_avg = sum(m[2] for m in correct_matches) / 4
        
        old_results.append(old_avg)
        correct_results.append(correct_avg)
        
        old_status = "✅" if old_avg < 5 else "⚠️" if old_avg < 10 else "❌"
        correct_status = "✅" if correct_avg < 5 else "⚠️" if correct_avg < 10 else "❌"
        
        print(f"  {angle:>4}° | {old_avg:6.2f}px {old_status} | {correct_avg:6.2f}px {correct_status}")
    
    # Summary
    if old_results and correct_results:
        print("\n" + "-" * 35)
        print(f"Average: OLD={sum(old_results)/len(old_results):.2f}px, CORRECT={sum(correct_results)/len(correct_results):.2f}px")
        
        if sum(correct_results)/len(correct_results) < sum(old_results)/len(old_results):
            print("\n✅ CORRECT FORMULA IS BETTER!")
        else:
            print("\n❌ OLD FORMULA IS BETTER (or same)")


if __name__ == '__main__':
    test_all_angles()
