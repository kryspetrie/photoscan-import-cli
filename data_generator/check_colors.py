#!/usr/bin/env python3
"""Check exactly what colors are at each position in the source image."""
import numpy as np
import cv2


def check_colors():
    """Check color values at specific positions."""
    pw, ph = 300, 200
    
    print("="*70)
    print("CHECKING COLOR VALUES")
    print("="*70)
    
    # Create photo with markers
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180  # BGR background = gray
    photo[:, :, 3] = 255
    
    ms = 10
    # TL - Red (R=255, G=0, B=0 in RGB, but OpenCV uses BGR)
    photo[0:ms, 0:ms] = [255, 0, 0, 255]
    # TR - Green (R=0, G=255, B=0 in RGB)
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]
    # BR - Blue (R=0, G=0, B=255 in RGB)
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]
    # BL - Yellow (R=255, G=255, B=0 in RGB)
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]
    
    print(f"\nSource photo colors (BGR format in numpy):")
    print(f"  Background (100, 100, 100) - Gray")
    print(f"  photo[0:ms, 0:ms] = [255, 0, 0] = Red in RGB = Blue in BGR?")
    print(f"  photo[0:ms, pw-ms:pw] = [0, 255, 0] = Green in RGB/BGR")
    print(f"  photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255] = Blue in RGB = Red in BGR?")
    print(f"  photo[ph-ms:ph, 0:ms] = [0, 255, 255] = Yellow in RGB = Magenta in BGR?")
    
    # Check pixel values
    print(f"\nPixel values at corners:")
    print(f"  (5, 5) = {photo[5, 5]}")  # Should be Red [255, 0, 0]
    print(f"  (pw-5, 5) = {photo[5, pw-5]}")  # Should be Green [0, 255, 0]
    print(f"  (pw-5, ph-5) = {photo[ph-5, pw-5]}")  # Should be Blue [0, 0, 255]
    print(f"  (5, ph-5) = {photo[ph-5, 5]}")  # Should be Yellow [0, 255, 255]
    
    # Convert to HSV and check
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    print(f"\nHSV values at corners:")
    print(f"  (5, 5) = {hsv[5, 5]}")
    print(f"  (pw-5, 5) = {hsv[5, pw-5]}")
    print(f"  (pw-5, ph-5) = {hsv[ph-5, pw-5]}")
    print(f"  (5, ph-5) = {hsv[ph-5, 5]}")
    
    # What does cv2.imwrite save as?
    cv2.imwrite('/tmp/markers_bgra.png', photo)
    
    # Read it back as BGR
    loaded = cv2.imread('/tmp/markers_bgra.png')
    print(f"\nAfter save/load (BGR):")
    print(f"  (5, 5) = {loaded[5, 5]}")
    print(f"  (pw-5, 5) = {loaded[5, pw-5]}")
    print(f"  (pw-5, ph-5) = {loaded[ph-5, pw-5]}")
    print(f"  (5, ph-5) = {loaded[ph-5, 5]}")
    
    # Detect colors in the BGR image (after save/load)
    hsv_loaded = cv2.cvtColor(loaded, cv2.COLOR_BGR2HSV)
    print(f"\nHSV after save/load:")
    print(f"  (5, 5) = {hsv_loaded[5, 5]}")
    print(f"  (pw-5, 5) = {hsv_loaded[5, pw-5]}")
    print(f"  (pw-5, ph-5) = {hsv_loaded[ph-5, pw-5]}")
    print(f"  (5, ph-5) = {hsv_loaded[ph-5, 5]}")


if __name__ == '__main__':
    check_colors()
