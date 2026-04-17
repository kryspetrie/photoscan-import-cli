#!/usr/bin/env python3
"""Debug the rotation math vs actual composite."""
import numpy as np
import cv2


def rotate_photo_debug(photo, angle):
    """Rotate and show the actual corner positions."""
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


def debug_rotation_vs_composite():
    """Compare rotation math to actual result."""
    pw, ph = 300, 200
    cx, cy = 620, 620  # Center in padded canvas space
    angle = -60
    
    print("="*70)
    print("STEP-BY-STEP DEBUG")
    print("="*70)
    
    # Step 1: Create a simple photo
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    ms = 20
    photo[0:ms, 0:ms] = [255, 0, 0, 255]           # TL - Red
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]       # TR - Green
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]   # BR - Blue
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]     # BL - Yellow
    
    print(f"\nStep 1: Original photo {pw}x{ph}")
    print(f"  Markers at:")
    print(f"    TL: (0, 0) - Red")
    print(f"    TR: ({pw}, 0) - Green")
    print(f"    BR: ({pw}, {ph}) - Blue")
    print(f"    BL: (0, {ph}) - Yellow")
    
    # Step 2: Rotate photo
    rotated, rot_M, (new_w, new_h) = rotate_photo_debug(photo, angle)
    print(f"\nStep 2: Rotate by {angle}°")
    print(f"  Rotated photo size: {new_w}x{new_h}")
    print(f"  Rotation matrix M:\n{rot_M}")
    
    # Step 3: Find actual corner positions in rotated image
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red"),
        ([35, 100, 100], [85, 255, 255], "Green"),
        ([100, 100, 100], [130, 255, 255], "Blue"),
        ([15, 100, 100], [45, 255, 255], "Yellow")
    ]
    
    print(f"\nStep 3: Detect marker positions in rotated image")
    rotated_corners = {}
    for i, (lower, upper, name) in enumerate(ranges):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                rotated_corners[i] = (px, py)
                print(f"    {name}: ({px}, {py})")
    
    # Step 4: Verify our rotation math against actual result
    print(f"\nStep 4: Verify rotation matrix math")
    print(f"  Photo center: ({pw/2}, {ph/2})")
    
    # The actual rotation matrix used by OpenCV
    center = (pw / 2, ph / 2)
    M_verify = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(M_verify[0, 0])
    sin_a = abs(M_verify[0, 1])
    rw = int(ph * sin_a + pw * cos_a)
    rh = int(ph * cos_a + pw * sin_a)
    M_verify[0, 2] += (rw - pw) / 2
    M_verify[1, 2] += (rh - ph) / 2
    
    print(f"  Our matrix M:\n{M_verify}")
    
    # Apply M to corners
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    print(f"\n  Applying M to corners:")
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        result = M_verify @ pt
        if i in rotated_corners:
            actual = rotated_corners[i]
            print(f"    Corner {i}: calc=({result[0]:.1f}, {result[1]:.1f}), actual=({actual[0]}, {actual[1]})")
        else:
            print(f"    Corner {i}: calc=({result[0]:.1f}, {result[1]:.1f})")
    
    # Step 5: Composite onto canvas
    print(f"\nStep 5: Composite onto canvas at ({cx}, {cy})")
    
    canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    
    # Manual composite to see exact placement
    top_left_x = int(cx - new_w / 2)
    top_left_y = int(cy - new_h / 2)
    print(f"  Placing rotated photo with top-left at ({top_left_x}, {top_left_y})")
    
    result = canvas.copy()
    for y in range(new_h):
        for x in range(new_w):
            src_px = rotated[y, x]
            dst_x = top_left_x + x
            dst_y = top_left_y + y
            if 0 <= dst_x < 1240 and 0 <= dst_y < 1240:
                result[dst_y, dst_x] = src_px
    
    # Detect corners in composite
    hsv2 = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    canvas_corners = {}
    for i, (lower, upper, name) in enumerate(ranges):
        mask = cv2.inRange(hsv2, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                canvas_corners[i] = (px, py)
                print(f"    {name}: ({px}, {py})")
    
    # The canvas position should be: rotated_position + top_left
    print(f"\n  Expected positions (rotated_pos + top_left):")
    for i in range(4):
        if i in rotated_corners:
            rot_pos = rotated_corners[i]
            expected = (rot_pos[0] + top_left_x, rot_pos[1] + top_left_y)
            actual = canvas_corners.get(i, (0, 0))
            print(f"    {i}: ({expected[0]:.1f}, {expected[1]:.1f}) vs ({actual[0]}, {actual[1]})")


if __name__ == '__main__':
    debug_rotation_vs_composite()
