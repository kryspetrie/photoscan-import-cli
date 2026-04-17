#!/usr/bin/env python3
"""Check if markers are in correct positions in source image."""
import numpy as np
import cv2


def check_source_markers():
    """Verify markers are at correct positions in source image."""
    pw, ph = 300, 200
    
    print("="*70)
    print("CHECKING SOURCE MARKER POSITIONS")
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
    
    # Save source image
    cv2.imwrite('/tmp/source_markers.png', photo)
    print(f"\nSaved source image to /tmp/source_markers.png")
    
    # Detect markers in source
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red/TL"),
        ([35, 100, 100], [85, 255, 255], "Green/TR"),
        ([100, 100, 100], [130, 255, 255], "Blue/BR"),
        ([15, 100, 100], [45, 255, 255], "Yellow/BL")
    ]
    
    print(f"\nDetected markers in source image:")
    detected = {}
    for i, (lower, upper, name) in enumerate(ranges):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
                detected[i] = (px, py)
                print(f"  {i} ({name}): ({px}, {py})")
    
    # Expected positions
    print(f"\nExpected marker positions:")
    print(f"  0 (Red/TL): should be near (5, 5)")
    print(f"  1 (Green/TR): should be near ({pw-5}, 5)")
    print(f"  2 (Blue/BR): should be near ({pw-5}, {ph-5})")
    print(f"  3 (Yellow/BL): should be near (5, {ph-5})")
    
    # Check for errors
    expected = {
        0: (5, 5),
        1: (pw-5, 5),
        2: (pw-5, ph-5),
        3: (5, ph-5)
    }
    
    print(f"\nPosition errors:")
    for i in range(4):
        if i in detected:
            dx = detected[i][0] - expected[i][0]
            dy = detected[i][1] - expected[i][1]
            dist = np.sqrt(dx**2 + dy**2)
            print(f"  {i}: detected={detected[i]}, expected={expected[i]}, error={dist:.1f}px")


if __name__ == '__main__':
    check_source_markers()
