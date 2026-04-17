#!/usr/bin/env python3
"""
End-to-end test with colored markers - verifies get_rotated_polygon()
matches actual photo corners after rotation and compositing.
"""

import cv2
import numpy as np
import math
from pathlib import Path


def rotate_photo(photo, angle):
    """Rotate photo and return new dimensions."""
    h, w = photo.shape[:2]
    if abs(angle) < 1:
        return photo, (w, h)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(photo, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(128, 128, 128, 255))
    return rotated, (new_w, new_h)


def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """The get_rotated_polygon implementation from generate_dataset.py."""
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],
            [center_x + hw, center_y - hh],
            [center_x + hw, center_y + hh],
            [center_x - hw, center_y + hh]
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(height * sin_a + width * cos_a)
    new_h = int(height * cos_a + width * sin_a)
    
    M[0, 2] += (new_w - width) / 2
    M[1, 2] += (new_h - height) / 2
    
    corners_photo = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    corners_rotated = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        result = M @ pt
        corners_rotated[i] = [result[0], result[1]]
    
    rotated_center = M @ np.array([photo_center[0], photo_center[1], 1])
    
    corners_final = np.zeros_like(corners_rotated)
    corners_final[:, 0] = center_x - rotated_center[0] + corners_rotated[:, 0]
    corners_final[:, 1] = center_y - rotated_center[1] + corners_rotated[:, 1]
    
    return corners_final


def place_markers(photo, size=15):
    """Place colored markers at corners."""
    h, w = photo.shape[:2]
    result = photo.copy()
    
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 0, 0)]
    positions = [(size, size), (w-size-1, size), (w-size-1, h-size-1), (size, h-size-1)]
    
    for pos, color in zip(positions, colors):
        cv2.circle(result, pos, size, color, -1)
        cv2.line(result, (pos[0]-size, pos[1]), (pos[0]+size, pos[1]), (255,255,255), 2)
        cv2.line(result, (pos[0], pos[1]-size), (pos[0], pos[1]+size), (255,255,255), 2)
    
    return result


def detect_markers(photo):
    """Detect colored markers and return positions."""
    img_check = photo[:, :, :3]
    
    ranges = [
        (np.array([0, 200, 0]), np.array([100, 255, 100])),   # Green
        (np.array([0, 0, 200]), np.array([100, 100, 255])),   # Red
        (np.array([200, 0, 200]), np.array([255, 100, 255])),  # Magenta
        (np.array([200, 0, 0]), np.array([255, 100, 100])),    # Blue
    ]
    
    detected = []
    for lower, upper in ranges:
        mask = cv2.inRange(img_check, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 25:
                M_m = cv2.moments(largest)
                if M_m['m00'] > 0:
                    detected.append((M_m['m10']/M_m['m00'], M_m['m01']/M_m['m00']))
                    continue
        detected.append(None)
    
    return detected


def composite(canvas, photo, cx, cy):
    """Composite photo onto canvas at center."""
    ph, pw = photo.shape[:2]
    
    if photo.shape[2] == 3:
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
    
    top_x, top_y = int(cx - pw/2), int(cy - ph/2)
    
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = pw, ph
    dst_x1, dst_y1 = top_x, top_y
    dst_x2, dst_y2 = top_x + pw, top_y + ph
    
    if dst_x1 < 0:
        src_x1 = -dst_x1
        dst_x1 = 0
    if dst_y1 < 0:
        src_y1 = -dst_y1
        dst_y1 = 0
    if dst_x2 > canvas.shape[1]:
        src_x2 = canvas.shape[1] - dst_x1
        dst_x2 = canvas.shape[1]
    if dst_y2 > canvas.shape[0]:
        src_y2 = canvas.shape[0] - dst_y1
        dst_y2 = canvas.shape[0]
    
    copy_w = int(dst_x2 - dst_x1)
    copy_h = int(dst_y2 - dst_y1)
    
    if copy_w <= 0 or copy_h <= 0:
        return canvas
    
    canvas_f = canvas.astype(np.float32) / 255.0
    photo_f = photo[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w].astype(np.float32) / 255.0
    
    alpha = photo_f[:, :, 3:4]
    canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] = (
        photo_f[:, :, :3] * alpha + 
        canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] * (1 - alpha)
    ).astype(np.float32)
    
    return (canvas_f * 255).astype(np.uint8)


def run_test(photo_path, photo_w, photo_h, center_x, center_y, rotation, padding):
    """Run test for a single configuration."""
    # Load photo
    photo = cv2.imread(str(photo_path))
    photo = cv2.resize(photo, (photo_w, photo_h))
    
    # Add BGRA
    photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
    photo[:, :, 3] = 255
    
    # Place markers
    photo = place_markers(photo)
    
    # Rotate
    rotated, rotated_size = rotate_photo(photo, rotation)
    
    # Create canvas
    canvas_size = 640 + 2 * padding
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    canvas[:, :, :3] = 180
    canvas[:, :, 3] = 255
    
    # Composite
    canvas_cx = center_x + padding
    canvas_cy = center_y + padding
    canvas = composite(canvas, rotated, canvas_cx, canvas_cy)
    
    # Calculate expected corners in padded space
    expected = get_rotated_polygon(photo_w, photo_h, center_x, center_y, rotation)
    expected_in_canvas = expected + np.array([padding, padding])
    
    # Detect markers
    detected = detect_markers(canvas)
    
    # Compare
    errors = []
    for i in range(4):
        if detected[i] and i < len(expected_in_canvas):
            error = math.sqrt((detected[i][0] - expected_in_canvas[i][0])**2 + 
                            (detected[i][1] - expected_in_canvas[i][1])**2)
            errors.append(error)
    
    if errors:
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        return max_error, avg_error, detected, expected_in_canvas
    else:
        return 999, 999, detected, expected_in_canvas


def main():
    print("=" * 70)
    print("END-TO-END CORNER TRACKING VERIFICATION")
    print("=" * 70)
    
    # Load test image
    sources = list(Path('./images').glob('*.jpg'))
    if not sources:
        print("ERROR: No source images found")
        return False
    
    test_photo = sources[0]
    
    # Test configurations
    configs = [
        (300, 225, 320, 320, 0, 300),    # No rotation
        (300, 225, 320, 320, 15, 300),   # Small rotation
        (300, 225, 320, 320, 25, 300),   # Large rotation
        (250, 200, 200, 200, 20, 300),   # Different size/position
        (400, 300, 400, 400, 30, 300),   # Large photo
    ]
    
    all_passed = True
    total_max_error = 0
    total_avg_error = 0
    
    print("\nTesting corner tracking accuracy:")
    print("-" * 50)
    
    for i, (pw, ph, cx, cy, rot, pad) in enumerate(configs):
        max_err, avg_err, detected, expected = run_test(test_photo, pw, ph, cx, cy, rot, pad)
        total_max_error += max_err
        total_avg_error += avg_err
        
        status = "✅" if max_err < 15 else "❌"
        print(f"Config {i+1}: rot={rot}°, size={pw}x{ph}, center=({cx},{cy})")
        print(f"  Max error: {max_err:.1f}px, Avg error: {avg_err:.1f}px {status}")
        
        if max_err >= 15:
            all_passed = False
    
    print("-" * 50)
    print(f"\nOverall: Max={total_max_error/len(configs):.1f}px, Avg={total_avg_error/len(configs):.1f}px")
    
    if all_passed:
        print("✅ ALL TESTS PASSED - Corner tracking verified!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Corner tracking needs fixes")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
