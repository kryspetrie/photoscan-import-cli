#!/usr/bin/env python3
"""Deep debug of rotation matrix and polygon calculation."""
import numpy as np
import cv2


def rotate_photo(photo, angle):
    """Rotate and return details."""
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


def debug_rotation_matrix():
    """Debug the exact rotation matrix behavior."""
    pw, ph = 300, 200
    angle = -60
    photo_center = (pw / 2, ph / 2)
    
    print("="*70)
    print(f"DEBUGGING ROTATION MATRIX: {angle}°")
    print("="*70)
    
    # Build M_raw (without canvas offset)
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    print(f"\n1. M_raw (rotation around photo center):")
    print(f"   {M_raw}")
    
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    print(f"\n2. Rotated dimensions: {new_w}x{new_h}")
    
    canvas_offset_x = (new_w - pw) / 2
    canvas_offset_y = (new_h - ph) / 2
    print(f"   Canvas offset: ({canvas_offset_x}, {canvas_offset_y})")
    
    # Build M_adjusted (with canvas offset like in rotate_photo)
    M_adjusted = M_raw.copy()
    M_adjusted[0, 2] += canvas_offset_x
    M_adjusted[1, 2] += canvas_offset_y
    print(f"\n3. M_adjusted (M_raw + canvas_offset):")
    print(f"   {M_adjusted}")
    
    # Verify with actual rotation
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, 3] = 255
    
    ms = 20
    photo[0:ms, 0:ms] = [255, 0, 0, 255]           # TL - Red
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]       # TR - Green
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]   # BR - Blue
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]     # BL - Yellow
    
    rotated, actual_M, (rot_w, rot_h) = rotate_photo(photo, angle)
    print(f"\n4. Actual rotation matrix from rotate_photo:")
    print(f"   {actual_M}")
    
    print(f"\n5. Apply M_adjusted to corners:")
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    # Expected corners from M_adjusted
    for i, c in enumerate(corners_photo):
        pt = np.array([c[0], c[1], 1])
        result = M_adjusted @ pt
        print(f"   Corner {i} ({c[0]:.0f},{c[1]:.0f}) -> ({result[0]:.1f}, {result[1]:.1f})")
    
    # Detect actual corners in rotated image
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red"),
        ([35, 100, 100], [85, 255, 255], "Green"),
        ([100, 100, 100], [130, 255, 255], "Blue"),
        ([15, 100, 100], [45, 255, 255], "Yellow")
    ]
    
    print(f"\n6. Actual corner positions in rotated image:")
    actual_rot = {}
    for i, (lower, upper, name) in enumerate(ranges):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                actual_rot[i] = (px, py)
                print(f"   {name} (corner {i}): ({px}, {py})")
    
    # Compare M_adjusted output to actual
    print(f"\n7. Comparing M_adjusted to actual:")
    for i in range(4):
        if i in actual_rot:
            pt = np.array([corners_photo[i][0], corners_photo[i][1], 1])
            expected = M_adjusted @ pt
            actual = actual_rot[i]
            dx = actual[0] - expected[0]
            dy = actual[1] - expected[1]
            print(f"   Corner {i}: M_adjusted=({expected[0]:.1f},{expected[1]:.1f}), actual=({actual[0]},{actual[1]}), diff=({dx:.1f},{dy:.1f})")


def debug_polygon_calculation():
    """Debug the polygon calculation formula."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = -60
    
    print("\n" + "="*70)
    print("DEBUGGING POLYGON CALCULATION")
    print("="*70)
    
    photo_center = (pw / 2, ph / 2)
    
    # Build rotation matrix
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    print(f"\nPhoto center: {photo_center}")
    print(f"Rotated dimensions: {new_w}x{new_h}")
    
    # When compositing, top_left = (cx - new_w/2, cy - new_h/2)
    top_left_x = cx - new_w / 2
    top_left_y = cy - new_h / 2
    print(f"Canvas top-left of rotated photo: ({top_left_x}, {top_left_y})")
    
    # The corner position in rotated image space = M_raw @ corner
    # The corner position on canvas = top_left + M_raw @ corner - photo_center
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    print(f"\nFor each corner P in photo space, canvas position = top_left + M_raw @ (P - photo_center)")
    print(f"Then the rotated photo's center in canvas space = top_left + M_raw @ (0,0)")
    
    for i, P in enumerate(corners_photo):
        # Rotate P around photo_center
        pt = np.array([P[0], P[1], 1])
        rotated = M_raw @ pt
        
        # This gives the CORRECT position in the rotated image (before canvas offset)
        # But wait - M_raw doesn't include any offset, so this IS the correct rotated position
        print(f"\nCorner {i}: P=({P[0]},{P[1]})")
        print(f"  M_raw @ P = ({rotated[0]:.2f}, {rotated[1]:.2f})")
        print(f"  Canvas position = ({top_left_x + rotated[0]:.2f}, {top_left_y + rotated[1]:.2f})")
    
    # Actually, M_raw @ P already gives us the rotated position around the photo center
    # Let me verify this by checking where the photo center ends up
    center_pt = np.array([photo_center[0], photo_center[1], 1])
    rotated_center = M_raw @ center_pt
    print(f"\nPhoto center after rotation: ({rotated_center[0]:.2f}, {rotated_center[1]:.2f})")
    print(f"  This should be close to (0, 0) for pure rotation around center")


if __name__ == '__main__':
    debug_rotation_matrix()
    debug_polygon_calculation()
