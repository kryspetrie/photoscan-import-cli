#!/usr/bin/env python3
"""Debug the marker rotation - check if markers rotate correctly."""
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


def debug_rotation():
    """Debug why markers don't rotate to expected positions."""
    pw, ph = 300, 200
    angle = -60
    
    print("="*70)
    print(f"DEBUGGING ROTATION OF {pw}x{ph} BY {angle}°")
    print("="*70)
    
    # Create photo with markers IN THE CENTER of each corner region
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    # Markers at the geometric CENTER of each corner (not at the edge)
    # This ensures they don't get clipped by rotation
    mr = 5  # marker radius
    cx, cy = pw//2, ph//2  # photo center
    
    # TL marker - at (10, 10)
    photo[8:13, 8:13] = [255, 0, 0, 255]           # Red at (10, 10)
    # TR marker - at (pw-10, 10)
    photo[8:13, pw-13:pw-8] = [0, 255, 0, 255]     # Green at (290, 10)
    # BR marker - at (pw-10, ph-10)
    photo[ph-13:ph-8, pw-13:pw-8] = [0, 0, 255, 255]  # Blue at (290, 190)
    # BL marker - at (10, ph-10)
    photo[ph-13:ph-8, 8:13] = [0, 255, 255, 255]    # Yellow at (10, 190)
    
    print(f"\nOriginal markers:")
    print(f"  Red (TL): (8-13, 8-13) -> center ~(10, 10)")
    print(f"  Green (TR): (8-13, {pw-13}-{pw-8}) -> center ~(290, 10)")
    print(f"  Blue (BR): ({ph-13}-{ph-8}, {pw-13}-{pw-8}) -> center ~(290, 190)")
    print(f"  Yellow (BL): ({ph-13}-{ph-8}, 8-13) -> center ~(10, 190)")
    
    # Rotate
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    print(f"\nRotated size: {rot_w}x{rot_h}")
    
    # Build the rotation matrix
    photo_center = (pw / 2, ph / 2)
    M = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    M[0, 2] += (new_w - pw) / 2
    M[1, 2] += (new_h - ph) / 2
    
    print(f"\nRotation matrix M:\n{M}")
    
    # Calculate where each marker SHOULD go
    marker_centers = [
        (10, 10, "Red/TL"),
        (290, 10, "Green/TR"),
        (290, 190, "Blue/BR"),
        (10, 190, "Yellow/BL")
    ]
    
    print(f"\nExpected positions (M @ marker_center):")
    for mx, my, name in marker_centers:
        pt = np.array([mx, my, 1])
        result = M @ pt
        print(f"  {name} ({mx},{my}) -> ({result[0]:.1f}, {result[1]:.1f})")
    
    # Detect actual markers
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red"),
        ([35, 100, 100], [85, 255, 255], "Green"),
        ([100, 100, 100], [130, 255, 255], "Blue"),
        ([15, 100, 100], [45, 255, 255], "Yellow")
    ]
    
    print(f"\nDetected positions in rotated image:")
    detected = {}
    for (lower, upper, name) in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M_m = cv2.moments(largest)
            if M_m['m00'] > 0:
                px = int(M_m['m10'] / M_m['m00'])
                py = int(M_m['m01'] / M_m['m00'])
                detected[name] = (px, py)
                print(f"  {name}: ({px}, {py})")
    
    # Compare
    print(f"\nComparison:")
    name_to_idx = {"Red": 0, "Green": 1, "Blue": 2, "Yellow": 3}
    for mx, my, name in marker_centers:
        pt = np.array([mx, my, 1])
        expected = M @ pt
        if name in detected:
            actual = detected[name]
            dx = actual[0] - expected[0]
            dy = actual[1] - expected[1]
            dist = np.sqrt(dx**2 + dy**2)
            print(f"  {name}: expected=({expected[0]:.1f}, {expected[1]:.1f}), actual={actual}, diff=({dx:.1f}, {dy:.1f}), dist={dist:.1f}px")
    
    # The key question: are the detected markers in the correct quadrants?
    print(f"\nQuadrant check (center of rotated image is at {rot_w/2}, {rot_h/2}):")
    for name, pos in detected.items():
        qx = "LEFT" if pos[0] < rot_w/2 else "RIGHT"
        qy = "TOP" if pos[1] < rot_h/2 else "BOTTOM"
        print(f"  {name}: ({pos[0]}, {pos[1]}) - {qx}, {qy}")


if __name__ == '__main__':
    debug_rotation()
