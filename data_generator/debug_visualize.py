#!/usr/bin/env python3
"""Visualize the rotated image to understand the transformation."""
import numpy as np
import cv2
import sys


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


def visualize_rotation():
    """Visualize the rotation and understand the coordinate mapping."""
    pw, ph = 300, 200
    angle = -60
    
    print("="*70)
    print(f"VISUALIZING ROTATION: {angle}°")
    print("="*70)
    
    # Create photo with markers
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    ms = 10
    photo[0:ms, 0:ms] = [255, 0, 0, 255]           # Red at TL
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]       # Green at TR
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]  # Blue at BR
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]     # Yellow at BL
    
    # Rotate
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    
    print(f"\nOriginal: {pw}x{ph}")
    print(f"Rotated: {rot_w}x{rot_h}")
    
    # Build rotation matrix
    photo_center = (pw / 2, ph / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    print(f"\nM_raw (without canvas offset):\n{M_raw}")
    print(f"  R @ (0,0) = ({M_raw[0,2]:.1f}, {M_raw[1,2]:.1f})")
    print(f"  R @ (300,0) = ({300*M_raw[0,0] + M_raw[0,2]:.1f}, {300*M_raw[0,1] + M_raw[1,2]:.1f})")
    print(f"  R @ (300,200) = ({300*M_raw[0,0] + 200*M_raw[0,1] + M_raw[0,2]:.1f}, {300*M_raw[0,1] + 200*M_raw[1,1] + M_raw[1,2]:.1f})")
    print(f"  R @ (0,200) = ({200*M_raw[0,1] + M_raw[0,2]:.1f}, {200*M_raw[1,1] + M_raw[1,2]:.1f})")
    
    # Full matrix with canvas offset
    M = M_raw.copy()
    M[0, 2] += (new_w - pw) / 2
    M[1, 2] += (new_h - ph) / 2
    
    print(f"\nM (with canvas offset):\n{M}")
    print(f"  M @ (0,0) = ({M[0,2]:.1f}, {M[1,2]:.1f})")
    print(f"  M @ (300,0) = ({300*M[0,0] + M[0,2]:.1f}, {300*M[0,1] + M[1,2]:.1f})")
    print(f"  M @ (300,200) = ({300*M[0,0] + 200*M[0,1] + M[0,2]:.1f}, {300*M[0,1] + 200*M[1,1] + M[1,2]:.1f})")
    print(f"  M @ (0,200) = ({200*M[0,1] + M[0,2]:.1f}, {200*M[1,1] + M[1,2]:.1f})")
    
    # Detect corners in rotated image
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red/TL"),
        ([35, 100, 100], [85, 255, 255], "Green/TR"),
        ([100, 100, 100], [130, 255, 255], "Blue/BR"),
        ([15, 100, 100], [45, 255, 255], "Yellow/BL")
    ]
    
    print(f"\nDetected corners in rotated image:")
    for (lower, upper, name) in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M_c = cv2.moments(largest)
            if M_c['m00'] > 0:
                px = int(M_c['m10'] / M_c['m00'])
                py = int(M_c['m01'] / M_c['m00'])
                print(f"  {name}: ({px}, {py})")
    
    # Now let's understand what each corner should map to
    print(f"\nCoordinate mapping analysis:")
    print(f"  Corner (0,0) should go to approximately (173, -0) in rotated image")
    print(f"  Corner (300,0) should go to approximately (323, 260) in rotated image")
    print(f"  Corner (300,200) should go to approximately (150, 359) in rotated image")
    print(f"  Corner (0,200) should go to approximately (0, 100) in rotated image")
    
    # Save the rotated image for visual inspection
    cv2.imwrite('/tmp/rotated_debug.png', rotated)
    print(f"\nSaved rotated image to /tmp/rotated_debug.png")
    
    # Print pixel values at expected corners to verify
    print(f"\nPixel values at key positions in rotated image:")
    key_positions = [(173, 0), (323, 260), (150, 359), (0, 100)]
    for x, y in key_positions:
        if 0 <= x < rot_w and 0 <= y < rot_h:
            pixel = rotated[y, x]
            print(f"  ({x}, {y}): BGR={pixel[:3]}, alpha={pixel[3]}")


if __name__ == '__main__':
    visualize_rotation()
