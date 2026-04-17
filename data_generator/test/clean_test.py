#!/usr/bin/env python3
"""
Clean test with 3x3 pixel markers at exact positions (avoiding centroid ambiguity).
"""

import cv2
import numpy as np

def create_test_photo(width, height):
    """Create test photo with 3x3 markers at EXACT positions."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (128, 128, 128)
    
    # 3x3 markers at exact corners
    marker = 3
    
    # LL - Green - at (1, height-2) - center of 3x3 at (2, height-2)
    img[height-3:height, 0:3] = (0, 255, 0)
    
    # UL - Blue - at (0, 0)
    img[0:3, 0:3] = (255, 0, 0)
    
    # UR - Yellow - at (width-3, 0)
    img[0:3, width-3:width] = (0, 255, 255)
    
    # LR - Magenta - at (width-3, height-3)
    img[height-3:height, width-3:width] = (255, 0, 255)
    
    return img


def find_marker_centroid(canvas, color_name):
    """Find centroid of colored marker."""
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    ranges = {
        'green': ([35, 100, 100], [85, 255, 255]),
        'blue': ([100, 100, 100], [130, 255, 255]),
        'yellow': ([15, 100, 100], [35, 255, 255]),
        'magenta': ([140, 100, 100], [170, 255, 255]),
    }
    lower, upper = [np.array(x) for x in ranges[color_name]]
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return None


def test_rotation(photo_w, photo_h, rotation, center_x, center_y):
    """Test rotation with 3x3 markers."""
    print(f"\n{'='*60}")
    print(f"Test: {photo_w}x{photo_h} @ {rotation}°")
    print(f"{'='*60}")
    
    # Create photo
    photo = create_test_photo(photo_w, photo_h)
    
    # Expected marker positions in photo space (centers of 3x3 blocks)
    # LL: center at (1.5, photo_h - 1.5) ≈ (2, height-2)
    # UL: center at (1.5, 1.5) ≈ (2, 2)
    # UR: center at (photo_w - 1.5, 1.5) ≈ (width-2, 2)
    # LR: center at (photo_w - 1.5, photo_h - 1.5) ≈ (width-2, height-2)
    
    expected_photo = {
        'LL': (1.5, photo_h - 1.5),
        'UL': (1.5, 1.5),
        'UR': (photo_w - 1.5, 1.5),
        'LR': (photo_w - 1.5, photo_h - 1.5),
    }
    
    # Rotate photo
    h, w = photo.shape[:2]
    center = (w / 2, h / 2)
    M_rot = cv2.getRotationMatrix2D(center, rotation, 1.0)
    cos = abs(M_rot[0, 0])
    sin = abs(M_rot[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(w * sin + h * cos)
    M_rot[0, 2] += (new_w - w) / 2
    M_rot[1, 2] += (new_h - h) / 2
    
    rotated = cv2.warpAffine(photo, M_rot, (new_w, new_h), borderValue=(0, 0, 0))
    
    # Composite
    canvas = np.zeros((1240, 1240, 3), dtype=np.uint8)
    canvas[:] = (50, 50, 50)
    
    tl_x = int(center_x - new_w / 2)
    tl_y = int(center_y - new_h / 2)
    
    end_x = min(tl_x + new_w, 1240)
    end_y = min(tl_y + new_h, 1240)
    start_x = max(0, tl_x)
    start_y = max(0, tl_y)
    copy_w = end_x - start_x
    copy_h = end_y - start_y
    
    if copy_w > 0 and copy_h > 0:
        src_start_x = start_x - tl_x
        src_start_y = start_y - tl_y
        canvas[start_y:end_y, start_x:end_x] = rotated[src_start_y:src_start_y+copy_h, src_start_x:src_start_x+copy_w]
    
    # Find markers
    actual = {
        'LL': find_marker_centroid(canvas, 'green'),
        'UL': find_marker_centroid(canvas, 'blue'),
        'UR': find_marker_centroid(canvas, 'yellow'),
        'LR': find_marker_centroid(canvas, 'magenta'),
    }
    
    # Calculate expected marker positions using same formula
    M = cv2.getRotationMatrix2D((photo_w/2, photo_h/2), rotation, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(photo_h * sin + photo_w * cos)
    new_h = int(photo_w * sin + photo_h * cos)
    M[0, 2] += (new_w - photo_w) / 2
    M[1, 2] += (new_h - photo_h) / 2
    
    top_left_x = center_x - new_w / 2
    top_left_y = center_y - new_h / 2
    
    expected = {}
    for name, (px, py) in expected_photo.items():
        pt = np.array([px, py, 1])
        rotated = M @ pt
        expected[name] = (top_left_x + rotated[0], top_left_y + rotated[1])
    
    print("Expected marker positions (calculated):")
    for name, pos in expected.items():
        print(f"  {name}: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    print("\nActual marker positions (detected):")
    for name, pos in actual.items():
        if pos:
            print(f"  {name}: {pos}")
        else:
            print(f"  {name}: NOT FOUND")
    
    print("\nError:")
    all_errors = []
    for name in ['LL', 'UL', 'UR', 'LR']:
        if actual[name]:
            exp = expected[name]
            err_x = actual[name][0] - exp[0]
            err_y = actual[name][1] - exp[1]
            err_dist = np.sqrt(err_x**2 + err_y**2)
            all_errors.append(err_dist)
            status = "✓" if err_dist < 5 else "⚠"
            print(f"  {name}: Det={actual[name]} Exp=({exp[0]:.2f},{exp[1]:.2f}) Err=({err_x:.2f},{err_y:.2f}) {err_dist:.2f}px {status}")
    
    if all_errors:
        print(f"\nMax error: {max(all_errors):.2f}px, Avg: {np.mean(all_errors):.2f}px")
    
    # Save crop
    crop = canvas[center_y-250:center_y+250, center_x-250:center_x+250]
    cv2.imwrite(f"clean_test_w{photo_w}_h{photo_h}_r{rotation}.jpg", crop)
    
    return all_errors


def main():
    print("="*60)
    print("CLEAN ROTATION TEST - 3x3 markers at exact positions")
    print("="*60)
    
    test_cases = [
        (200, 300, 0),
        (200, 300, 15),
        (200, 300, 20),
        (200, 300, -20),
        (300, 200, 0),
        (300, 200, 15),
        (300, 200, 25),
        (400, 300, 0),
        (400, 300, 20),
    ]
    
    all_errors = []
    for w, h, rot in test_cases:
        errors = test_rotation(w, h, rot, 620, 620)
        all_errors.extend(errors)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    if all_errors:
        print(f"Total: {len(all_errors)} measurements")
        print(f"Max: {max(all_errors):.2f}px, Avg: {np.mean(all_errors):.2f}px")
        print(f"Within 2px: {sum(1 for e in all_errors if e < 2)}/{len(all_errors)}")
        print(f"Within 5px: {sum(1 for e in all_errors if e < 5)}/{len(all_errors)}")


if __name__ == "__main__":
    main()
