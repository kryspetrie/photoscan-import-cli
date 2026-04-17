#!/usr/bin/env python3
"""
End-to-End Corner Verification Test

This test verifies that:
1. A marker placed at a photo corner appears at the expected pixel position in the final image
2. The corner coordinates stored in the label file match the visual positions to <5px accuracy

The test:
1. Creates a synthetic photo with distinctive color markers at each corner
2. Runs the full pipeline (rotate → composite → perspective → crop → resize)
3. Detects the markers in the final 640x640 image
4. Compares detected positions with expected positions
5. Reports pixel accuracy for each corner
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import (
    CONFIG, get_rotated_polygon, rotate_photo, composite_photo_at_center,
    apply_photo_shadow, apply_global_perspective
)


def create_marker_photo(width, height, marker_size=20):
    """Create a photo with distinctive colored markers at the 4 corners."""
    photo = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Corner markers - bright, distinctive colors
    colors = [
        (255, 0, 0),     # TL - Red
        (0, 255, 0),     # TR - Green  
        (0, 0, 255),     # BR - Blue
        (255, 255, 0),   # BL - Yellow
    ]
    
    for i, color in enumerate(colors):
        if i == 0:  # TL
            x, y = 5, 5
        elif i == 1:  # TR
            x, y = width - marker_size - 5, 5
        elif i == 2:  # BR
            x, y = width - marker_size - 5, height - marker_size - 5
        else:  # BL
            x, y = 5, height - marker_size - 5
        
        photo[y:y+marker_size, x:x+marker_size] = color
    
    return photo


def detect_corner_markers(img, marker_size=20):
    """Detect colored markers and return their positions."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    markers = {}
    
    # Color ranges for detection (more permissive)
    color_ranges = {
        'TL': ([0, 100, 100], [10, 255, 255]),      # Red
        'TR': ([40, 100, 100], [80, 255, 255]),     # Green
        'BR': ([100, 100, 100], [130, 255, 255]),   # Blue
        'BL': ([20, 100, 100], [40, 255, 255]),      # Yellow
    }
    
    detected = {}
    
    for name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:  # Minimum area threshold
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detected[name] = (cx, cy)
    
    return detected


def run_pipeline(photo, rotation, center_x, center_y, canvas_size=640, padding=300):
    """Run the full pipeline and return the final image and expected corner positions."""
    
    CANVAS_SIZE = canvas_size
    PADDING = padding
    
    # Create canvas with background
    canvas = np.ones((PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0  # Transparent
    
    # Get rotated photo corners BEFORE placing
    photo_h, photo_w = photo.shape[:2]
    polygon = get_rotated_polygon(photo_w, photo_h, center_x + PADDING, center_y + PADDING, rotation)
    
    # Rotate and add alpha channel
    rotated_photo = rotate_photo(photo, rotation)
    if rotated_photo.shape[2] == 3:
        rgba = cv2.cvtColor(rotated_photo, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255  # Full opacity
        rotated_photo = rgba
    
    result = composite_photo_at_center(canvas, rotated_photo, center_x + PADDING, center_y + PADDING)
    
    # Apply global perspective
    photo_corners_list = [polygon]
    warped, global_corners, transform_matrix, content_bounds, warped_photo_corners = apply_global_perspective(
        result, PADDING*2 + CANVAS_SIZE, PADDING*2 + CANVAS_SIZE,
        photo_corners=photo_corners_list,
        crop_margin=CONFIG['CROP_MARGIN']
    )
    
    out_w, out_h = warped.shape[1], warped.shape[0]
    
    # Resize to 640x640 if needed
    if out_w != CANVAS_SIZE or out_h != CANVAS_SIZE:
        scale_x = CANVAS_SIZE / out_w
        scale_y = CANVAS_SIZE / out_h
        
        warped = cv2.resize(warped, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_LINEAR)
        out_w, out_h = CANVAS_SIZE, CANVAS_SIZE
        
        warped_photo_corners = [
            np.array([[kp[0] * scale_x, kp[1] * scale_y] for kp in corners])
            for corners in warped_photo_corners
        ]
    
    expected_corners = warped_photo_corners[0] if warped_photo_corners else polygon
    
    return warped, expected_corners


def test_single_photo():
    """Test with a single centered photo."""
    print("\n" + "="*70)
    print("TEST: Single Photo - No Rotation")
    print("="*70)
    
    photo_size = 300
    center = 640 // 2
    
    photo = create_marker_photo(photo_size, photo_size, marker_size=25)
    
    final_img, expected = run_pipeline(photo, 0, center, center)
    
    detected = detect_corner_markers(final_img)
    
    print(f"\n  Photo size: {photo_size}x{photo_size}")
    print(f"  Rotation: 0°")
    print(f"  Center: ({center}, {center})")
    print()
    
    errors = []
    labels = ['TL', 'TR', 'BR', 'BL']
    
    for label in labels:
        if label in detected:
            dx = detected[label][0] - expected[labels.index(label)][0]
            dy = detected[label][1] - expected[labels.index(label)][1]
            error = np.sqrt(dx**2 + dy**2)
            errors.append(error)
            status = "✅" if error < 5 else "❌"
            print(f"  {label}: detected=({detected[label][0]:.0f}, {detected[label][1]:.0f}) "
                  f"expected=({expected[labels.index(label)][0]:.0f}, {expected[labels.index(label)][1]:.0f}) "
                  f"error={error:.1f}px {status}")
        else:
            print(f"  {label}: NOT DETECTED ❌")
            errors.append(float('inf'))
    
    avg_error = sum(errors) / len(errors) if errors else float('inf')
    print(f"\n  Average error: {avg_error:.1f}px")
    
    cv2.imwrite('/tmp/test_single_no_rot.jpg', final_img)
    print(f"  Saved to /tmp/test_single_no_rot.jpg")
    
    return avg_error < 5


def test_rotated_photo(angle):
    """Test with a rotated photo."""
    print("\n" + "="*70)
    print(f"TEST: Single Photo - {angle}° Rotation")
    print("="*70)
    
    photo_size = 300
    center = 640 // 2 + 50  # Offset to better test rotation
    
    photo = create_marker_photo(photo_size, photo_size, marker_size=25)
    
    final_img, expected = run_pipeline(photo, angle, center, center)
    
    detected = detect_corner_markers(final_img)
    
    print(f"\n  Photo size: {photo_size}x{photo_size}")
    print(f"  Rotation: {angle}°")
    print(f"  Center: ({center}, {center})")
    print()
    
    errors = []
    labels = ['TL', 'TR', 'BR', 'BL']
    
    for label in labels:
        if label in detected:
            dx = detected[label][0] - expected[labels.index(label)][0]
            dy = detected[label][1] - expected[labels.index(label)][1]
            error = np.sqrt(dx**2 + dy**2)
            errors.append(error)
            status = "✅" if error < 5 else "❌"
            print(f"  {label}: detected=({detected[label][0]:.0f}, {detected[label][1]:.0f}) "
                  f"expected=({expected[labels.index(label)][0]:.0f}, {expected[labels.index(label)][1]:.0f}) "
                  f"error={error:.1f}px {status}")
        else:
            print(f"  {label}: NOT DETECTED ❌")
            errors.append(float('inf'))
    
    avg_error = sum(errors) / len(errors) if errors else float('inf')
    print(f"\n  Average error: {avg_error:.1f}px")
    
    cv2.imwrite(f'/tmp/test_single_{angle}deg.jpg', final_img)
    print(f"  Saved to /tmp/test_single_{angle}deg.jpg")
    
    return avg_error < 5


def test_multiple_angles():
    """Test with multiple rotation angles."""
    print("\n" + "="*70)
    print("TEST: Multiple Rotation Angles")
    print("="*70)
    
    photo_size = 250
    center = 640 // 2
    
    angles = [0, 15, 30, 45, 60, 75, 90, 135, 180, -30, -60]
    all_errors = []
    
    for angle in angles:
        photo = create_marker_photo(photo_size, photo_size, marker_size=20)
        final_img, expected = run_pipeline(photo, angle, center, center)
        detected = detect_corner_markers(final_img)
        
        errors = []
        labels = ['TL', 'TR', 'BR', 'BL']
        for label in labels:
            if label in detected:
                dx = detected[label][0] - expected[labels.index(label)][0]
                dy = detected[label][1] - expected[labels.index(label)][1]
                error = np.sqrt(dx**2 + dy**2)
                errors.append(error)
            else:
                errors.append(float('inf'))
        
        avg_error = sum(errors) / len(errors) if errors else float('inf')
        all_errors.append(avg_error)
        
        status = "✅" if avg_error < 5 else "❌"
        print(f"  {angle:4d}°: avg error = {avg_error:.1f}px {status}")
    
    print()
    print(f"  Overall average error: {sum(all_errors)/len(all_errors):.1f}px")
    print(f"  Max error: {max(all_errors):.1f}px")
    print(f"  Min error: {min(all_errors):.1f}px")


def main():
    print("\n" + "#"*70)
    print("#" + " "*20 + "CORNER ACCURACY VERIFICATION" + " "*18 + "#")
    print("#"*70)
    
    results = []
    
    results.append(("No rotation", test_single_photo()))
    results.append(("-30° rotation", test_rotated_photo(-30)))
    results.append(("30° rotation", test_rotated_photo(30)))
    results.append(("60° rotation", test_rotated_photo(60)))
    results.append(("90° rotation", test_rotated_photo(90)))
    
    test_multiple_angles()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Corner accuracy <5px verified!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Corners need adjustment!")
        return 1


if __name__ == '__main__':
    sys.exit(main())