#!/usr/bin/env python3
"""Final test: Single pixel markers across all angles."""
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


def get_polygon(pw, ph, cx, cy, angle):
    """Calculate polygon corners."""
    if abs(angle) < 1:
        return np.array([
            [cx - pw/2, cy - ph/2],
            [cx + pw/2, cy - ph/2],
            [cx + pw/2, cy + ph/2],
            [cx - pw/2, cy + ph/2]
        ], dtype=np.float32)
    
    photo_center = (pw / 2, ph / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    
    canvas_offset = (cx - pw/2, cy - ph/2)
    
    corners_photo = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)
    polygon = np.zeros_like(corners_photo)
    
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        polygon[i, 0] = canvas_offset[0] + rotated[0]
        polygon[i, 1] = canvas_offset[1] + rotated[1]
    
    return polygon


def test_all_angles():
    """Test all angles with single pixel markers."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    
    print("="*70)
    print("FINAL ACCURACY TEST: Single Pixel Markers")
    print("="*70)
    
    results = []
    
    for angle in range(-60, 91, 15):
        # Create photo with single pixel markers at exact corners
        photo = np.ones((ph, pw, 4), dtype=np.uint8) * 180
        photo[:, :, 3] = 255
        
        photo[0, 0] = [0, 0, 255, 255]           # TL - RED
        photo[0, pw-1] = [0, 255, 0, 255]        # TR - GREEN
        photo[ph-1, pw-1] = [255, 0, 0, 255]     # BR - BLUE
        photo[ph-1, 0] = [0, 255, 255, 255]      # BL - YELLOW
        
        rotated = rotate_photo(photo, angle)
        rot_h, rot_w = rotated.shape[:2]
        
        # Get dimensions
        photo_center = (pw / 2, ph / 2)
        M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
        cos_a = abs(M_raw[0, 0])
        sin_a = abs(M_raw[0, 1])
        new_w = int(ph * sin_a + pw * cos_a)
        new_h = int(ph * cos_a + pw * sin_a)
        
        canvas_top_left_x = cx - new_w / 2
        canvas_top_left_y = cy - new_h / 2
        
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
        
        detected = {}
        for (lower, upper), idx in zip(color_ranges, range(4)):
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            points = np.where(mask > 0)
            if len(points[0]) > 0:
                detected[idx] = (int(points[1][0]), int(points[0][0]))
        
        # Calculate polygon
        polygon = get_polygon(pw, ph, cx, cy, angle)
        
        # Calculate errors
        if len(detected) == 4:
            errors = []
            for i in range(4):
                dist = np.sqrt((polygon[i, 0] - detected[i][0])**2 + (polygon[i, 1] - detected[i][1])**2)
                errors.append(dist)
            avg_err = sum(errors) / 4
            max_err = max(errors)
        else:
            avg_err = None
            max_err = None
        
        results.append((angle, avg_err, max_err, len(detected)))
        
        if avg_err is not None:
            status = "✅" if max_err < 5 else "⚠️" if max_err < 8 else "❌"
            print(f"  {angle:>4}°: avg={avg_err:.2f}px, max={max_err:.2f}px, detected={len(detected)}/4 {status}")
        else:
            print(f"  {angle:>4}°: avg=N/A, detected={len(detected)}/4 ⚠️")
    
    # Summary
    valid = [(a, e, m) for a, e, m, d in results if e is not None]
    if valid:
        all_errors = [e for _, e, _ in valid]
        avg_all = sum(all_errors) / len(all_errors)
        max_all = max(m for _, _, m in valid)
        print(f"\n{'='*60}")
        print(f"SUMMARY: avg={avg_all:.2f}px, max={max_all:.2f}px")
        if max_all < 5:
            print("✅ ALL TESTS PASS - < 5px accuracy!")
        else:
            print("❌ SOME TESTS FAIL - need improvement")


if __name__ == '__main__':
    test_all_angles()
