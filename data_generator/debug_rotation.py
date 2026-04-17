#!/usr/bin/env python3
"""Debug the corner calculation step by step."""
import numpy as np
import cv2


def get_rotated_polygon_debug(pw, ph, center_x, center_y, rotation):
    """Debug version with print statements."""
    print(f"\n  Input: pw={pw}, ph={ph}, center=({center_x}, {center_y}), rotation={rotation}°")
    
    if abs(rotation) < 1:
        corners = np.array([
            [center_x - pw/2, center_y - ph/2],
            [center_x + pw/2, center_y - ph/2],
            [center_x + pw/2, center_y + ph/2],
            [center_x - pw/2, center_y + ph/2]
        ], dtype=np.float32)
        return corners
    
    photo_center = (pw / 2, ph / 2)
    print(f"  Photo center (in photo space): {photo_center}")
    
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    print(f"  Raw rotation matrix M:")
    print(f"    {M}")
    
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    print(f"  Rotated dimensions: {new_w}x{new_h}")
    
    # Add the translation for canvas expansion
    M[0, 2] += (new_w - pw) / 2
    M[1, 2] += (new_h - ph) / 2
    print(f"  Adjusted matrix M (after adding canvas offset):")
    print(f"    {M}")
    
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    print(f"  Corners in PHOTO SPACE: {corners_photo.tolist()}")
    
    corners_rot = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        result = M @ pt
        corners_rot[i] = [result[0], result[1]]
    print(f"  Corners after rotation (in rotated image space): {corners_rot.tolist()}")
    
    rotated_center = M @ np.array([photo_center[0], photo_center[1], 1])
    print(f"  Photo center in rotated image space: {rotated_center}")
    
    corners = np.zeros_like(corners_rot)
    corners[:, 0] = center_x - rotated_center[0] + corners_rot[:, 0]
    corners[:, 1] = center_y - rotated_center[1] + corners_rot[:, 1]
    print(f"  Final corners in CANVAS SPACE: {corners.tolist()}")
    
    return corners


def verify_rotation_is_correct():
    """Verify that the rotation math is correct by comparing to actual image transformation."""
    pw, ph = 300, 200
    cx, cy = 620, 620  # Center in padded canvas space
    angle = -60
    
    print("="*70)
    print(f"VERIFYING ROTATION: {angle}°")
    print("="*70)
    
    # Create a simple photo with markers
    photo = np.zeros((ph, pw, 3), dtype=np.uint8)
    # Add markers at each corner
    photo[0:10, 0:10] = [255, 0, 0]      # TL - Red
    photo[0:10, pw-10:pw] = [0, 255, 0]  # TR - Green
    photo[ph-10:ph, pw-10:pw] = [0, 0, 255]  # BR - Blue
    photo[ph-10:ph, 0:10] = [255, 255, 0]  # BL - Yellow
    
    # Rotate using OpenCV
    rotated = rotate_photo(photo, angle)
    print(f"\n  Original photo size: {pw}x{ph}")
    print(f"  Rotated photo size: {rotated.shape[1]}x{rotated.shape[0]}")
    
    # Find actual corner positions in rotated image
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    
    ranges = [
        (0, [0, 100, 100], [15, 255, 255], "TL (Red)"),
        (1, [35, 100, 100], [85, 255, 255], "TR (Green)"),
        (2, [100, 100, 100], [130, 255, 255], "BR (Blue)"),
        (3, [15, 100, 100], [45, 255, 255], "BL (Yellow)")
    ]
    
    actual_rotated_corners = {}
    for idx, lower, upper, name in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                actual_rotated_corners[idx] = (px, py)
                print(f"  {name} in rotated image: ({px}, {py})")
    
    # Calculate expected corner positions using our function
    expected_corners = get_rotated_polygon_debug(pw, ph, cx, cy, angle)
    
    print("\n  COMPARISON:")
    print("  Index | Actual (rotated space) | Expected (canvas space offset)")
    print("  ------+------------------------+-----------------------------")
    for i in range(4):
        if i in actual_rotated_corners:
            print(f"    {i}   |  {actual_rotated_corners[i]}             |  ({expected_corners[i][0]:.1f}, {expected_corners[i][1]:.1f})")
    
    # The key question: are the expected corners relative to (cx, cy)?
    # If we subtract (cx, cy) from expected, we should get corners relative to origin
    print("\n  Expected corners relative to canvas origin (620, 620):")
    for i in range(4):
        rel_x = expected_corners[i][0] - cx
        rel_y = expected_corners[i][1] - cy
        print(f"    {i}: ({rel_x:.1f}, {rel_y:.1f})")


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


if __name__ == '__main__':
    verify_rotation_is_correct()
