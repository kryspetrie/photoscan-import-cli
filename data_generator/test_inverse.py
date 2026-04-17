#!/usr/bin/env python3
"""Test the inverse transformation theory."""
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


def test_inverse_transform():
    """Test if inverse transform gives correct corner positions."""
    pw, ph = 300, 200
    angle = -60
    
    print("="*70)
    print("TESTING INVERSE TRANSFORM THEORY")
    print("="*70)
    
    # Create photo with markers
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    ms = 20
    photo[0:ms, 0:ms] = [255, 0, 0, 255]           # TL - Red
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]       # TR - Green
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]   # BR - Blue
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]     # BL - Yellow
    
    rotated, M, (rot_w, rot_h) = rotate_photo(photo, angle)
    
    print(f"\nOriginal photo: {pw}x{ph}")
    print(f"Rotated photo: {rot_w}x{rot_h}")
    print(f"\nForward transform matrix M:\n{M}")
    
    # Compute inverse of M
    M_inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2, :]
    print(f"\nInverse matrix M_inv:\n{M_inv}")
    
    # The theory: warpAffine does dst = src at position M_inv @ dst_pixel
    # So to find where source corner ends up, we need the inverse!
    # Actually wait, let me verify this by checking a known point
    
    # The photo center (150, 100) should go to the center of the rotated image
    # which is approximately (rot_w/2, rot_h/2) = (161.5, 179.5)
    photo_center = np.array([pw/2, ph/2, 1])
    center_dst = M @ photo_center
    print(f"\nPhoto center ({pw/2}, {ph/2}) -> M @ center = ({center_dst[0]:.1f}, {center_dst[1]:.1f})")
    print(f"Expected center of rotated image: ({rot_w/2}, {rot_h/2})")
    
    # Now let's find where corners ended up using M_inv
    # If corner P in source, and dst at position Q, then Q = M @ P
    # So Q = M_inv @ (some dst pixel)
    # For each detected corner Q, we need P = M_inv @ Q
    
    # Actually, let me test with a simple point
    # Point (0,0) in source
    pt = np.array([0, 0, 1])
    dst = M @ pt
    print(f"\nSource (0,0) -> M @ (0,0) = ({dst[0]:.1f}, {dst[1]:.1f})")
    
    # Verify with M_inv
    recovered = M_inv @ np.array([dst[0], dst[1], 1])
    print(f"M_inv @ ({dst[0]:.1f}, {dst[1]:.1f}) = ({recovered[0]:.1f}, {recovered[1]:.1f})")
    
    # Detect actual corners in rotated image
    hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red", 0),
        ([35, 100, 100], [85, 255, 255], "Green", 1),
        ([100, 100, 100], [130, 255, 255], "Blue", 2),
        ([15, 100, 100], [45, 255, 255], "Yellow", 3)
    ]
    
    print(f"\n" + "-"*50)
    print("Detected corners in rotated image:")
    actual_corners = {}
    corners_source = [(0, 0), (pw, 0), (pw, ph), (0, ph)]
    
    for (lower, upper, name, idx) in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M_moments = cv2.moments(largest)
            if M_moments['m00'] > 0:
                px = int(M_moments['m10'] / M_moments['m00'])
                py = int(M_moments['m01'] / M_moments['m00'])
                actual_corners[idx] = (px, py)
                
                # What source pixel maps to this?
                dst_pt = np.array([px, py, 1])
                src_pt = M_inv @ dst_pt
                print(f"  {name} (corner {idx}): detected at ({px}, {py})")
                print(f"    M_inv @ ({px}, {py}) = source ({src_pt[0]:.1f}, {src_pt[1]:.1f})")
                print(f"    Expected source: ({corners_source[idx][0]}, {corners_source[idx][1]})")


if __name__ == '__main__':
    test_inverse_transform()
