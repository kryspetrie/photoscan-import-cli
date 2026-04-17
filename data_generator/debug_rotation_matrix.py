#!/usr/bin/env python3
"""Debug rotation matrix - verify it matches what OpenCV does."""
import numpy as np
import cv2


def verify_rotation_matrix():
    """Verify the rotation matrix matches actual transformation."""
    pw, ph = 300, 200
    angle = -45
    photo_center = (pw / 2, ph / 2)
    
    print("="*70)
    print(f"VERIFYING ROTATION MATRIX FOR {angle}°")
    print("="*70)
    
    # Build matrix using cv2
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    
    print(f"\nPhoto center: {photo_center}")
    print(f"Rotation matrix M_raw:\n{M_raw}")
    
    # For a point at the photo center, it should stay at the photo center
    pt_center = np.array([photo_center[0], photo_center[1], 1])
    rotated_center = M_raw @ pt_center
    print(f"\nM_raw @ center = {rotated_center}")
    print(f"Expected: (150.0, 100.0)")
    
    # Test points around the center
    test_points = [
        (150, 0, "TOP"),
        (150, 200, "BOTTOM"),
        (0, 100, "LEFT"),
        (300, 100, "RIGHT"),
        (0, 0, "TL"),
        (300, 0, "TR"),
        (0, 200, "BL"),
        (300, 200, "BR")
    ]
    
    print(f"\nM_raw transformation for test points:")
    for x, y, name in test_points:
        pt = np.array([x, y, 1])
        result = M_raw @ pt
        print(f"  {name} ({x}, {y}) -> ({result[0]:.1f}, {result[1]:.1f})")
    
    # Now verify by actually rotating an image with a marker at one test point
    print(f"\n" + "-"*50)
    print("ACTUAL ROTATION VERIFICATION")
    print("-"*50)
    
    # Create image with marker at TOP (150, 0)
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    photo[0:5, 148:153] = [0, 0, 255, 255]  # RED marker at TOP center
    
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    print(f"\nOriginal marker at (150, 0)")
    print(f"Rotated image size: {rot_w}x{rot_h}")
    
    # Find the marker in the rotated image
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([15, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            px = int(M['m10'] / M['m00'])
            py = int(M['m01'] / M['m00'])
            print(f"Detected marker at ({px}, {py})")
            print(f"M_raw predicted: ({M_raw[0,2]:.1f}, {M_raw[1,2]:.1f})")
    
    # Save rotated image for visual inspection
    cv2.imwrite('/tmp/rotated_marker.png', rotated)
    print(f"\nSaved to /tmp/rotated_marker.png")
    
    # Also save a visualization showing where the marker should be
    vis = rotated.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            px = int(M['m10'] / M['m00'])
            py = int(M['m01'] / M['m00'])
            cv2.circle(vis, (px, py), 10, (0, 255, 255), 2)
            cv2.putText(vis, f"({px}, {py})", (px+15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw where M_raw predicts the center should be
    cv2.circle(vis, (int(M_raw[0,2]), int(M_raw[1,2])), 10, (255, 255, 0), 2)
    cv2.putText(vis, f"M_raw center ({int(M_raw[0,2])}, {int(M_raw[1,2])})", (int(M_raw[0,2])+15, int(M_raw[1,2])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    cv2.imwrite('/tmp/rotated_vis.png', vis)
    print(f"Saved visualization to /tmp/rotated_vis.png")


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
    verify_rotation_matrix()
