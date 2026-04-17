#!/usr/bin/env python3
"""
Corner Verification Script - Color Detection Method
=====================================================

This script creates a test image with DISTINCT COLORED MARKERS at each corner
BEFORE any processing, then runs the full pipeline (rotate, composite, warp)
and detects those colors in the final output to compare with calculated corners.

KEY IDEA: If we know exactly where we placed colors, we can verify if our
corner calculations are correct by detecting those colors in the final image.
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path

# Colors for each corner (BGR format for OpenCV)
# These are very distinct colors that should be easy to detect
CORNER_COLORS = {
    'LL': (255, 0, 0),      # Blue - Lower-Left
    'UL': (0, 255, 0),      # Green - Upper-Left
    'UR': (0, 0, 255),      # Red - Upper-Right
    'LR': (255, 0, 255),    # Magenta - Lower-Right
}

# Color detection ranges (in BGR)
COLOR_RANGES = {
    'LL': {'lower': (0, 0, 200), 'upper': (100, 100, 255)},    # Blue
    'UL': {'lower': (0, 200, 0), 'upper': (100, 255, 100)},    # Green
    'UR': {'lower': (200, 0, 0), 'upper': (255, 100, 100)},    # Red
    'LR': {'lower': (200, 0, 200), 'upper': (255, 100, 255)},  # Magenta
}


def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """Calculate the 4 corner coordinates of a rotated rectangle."""
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],  # TL
            [center_x + hw, center_y - hh],  # TR
            [center_x + hw, center_y + hh],  # BR
            [center_x - hw, center_y + hh]   # BL
        ], dtype=np.float32)
    
    M = cv2.getRotationMatrix2D((center_x, center_y), rotation, 1.0)
    
    hw, hh = width / 2, height / 2
    corners = np.array([
        [center_x - hw, center_y - hh],
        [center_x + hw, center_y - hh],
        [center_x + hw, center_y + hh],
        [center_x - hw, center_y + hh]
    ], dtype=np.float32)
    
    rotated = np.zeros_like(corners)
    for i in range(4):
        pt = np.array([corners[i, 0], corners[i, 1], 1], dtype=np.float32)
        result = M @ pt
        rotated[i] = [result[0], result[1]]
    
    return rotated


def reorder_corners_to_llulur(corners):
    """Reorder 4 corners to LL, UL, UR, LR based on position."""
    corners = np.array(corners)
    
    sorted_by_x = sorted(enumerate(corners), key=lambda i_c: corners[i_c[0]][0])
    
    left_indices = [i for i, c in sorted_by_x[:2]]
    right_indices = [i for i, c in sorted_by_x[2:]]
    
    left_corners = [(corners[i][0], corners[i][1]) for i in left_indices]
    left_corners.sort(key=lambda c: c[1])
    ul = np.array(left_corners[0])
    ll = np.array(left_corners[1])
    
    right_corners = [(corners[i][0], corners[i][1]) for i in right_indices]
    right_corners.sort(key=lambda c: c[1])
    ur = np.array(right_corners[0])
    lr = np.array(right_corners[1])
    
    return np.array([ll, ul, ur, lr], dtype=np.float32)


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


def composite_photo_at_center(canvas, photo, cx, cy):
    """Composite photo onto canvas with center at (cx, cy)."""
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    
    # Ensure photo has alpha channel
    if photo.shape[2] == 3:
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
    
    top_left_x = int(cx - pw / 2)
    top_left_y = int(cy - ph / 2)
    
    src_x1, src_y1 = 0, 0
    src_x2, src_y2 = pw, ph
    dst_x1, dst_y1 = top_left_x, top_left_y
    dst_x2, dst_y2 = top_left_x + pw, top_left_y + ph
    
    if dst_x1 < 0:
        src_x1 = -dst_x1
        dst_x1 = 0
    if dst_y1 < 0:
        src_y1 = -dst_y1
        dst_y1 = 0
    if dst_x2 > cw:
        src_x2 = cw - dst_x1
        dst_x2 = cw
    if dst_y2 > ch:
        src_y2 = ch - dst_y1
        dst_y2 = ch
    
    copy_w = int(dst_x2 - dst_x1)
    copy_h = int(dst_y2 - dst_y1)
    
    if copy_w <= 0 or copy_h <= 0:
        return canvas
    
    src_x1, src_y1 = int(src_x1), int(src_y1)
    
    canvas_f = canvas.astype(np.float32) / 255.0
    photo_f = photo[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w].astype(np.float32) / 255.0
    
    alpha = photo_f[:, :, 3:4]
    
    canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] = (
        photo_f[:, :, :3] * alpha + 
        canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, :3] * (1 - alpha)
    ).astype(np.float32)
    canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, 3] = np.maximum(
        canvas_f[dst_y1:dst_y2, dst_x1:dst_x2, 3],
        photo_f[:, :, 3]
    )
    
    return (canvas_f * 255).astype(np.uint8)


def place_colored_corners_on_photo(photo, corner_size=15):
    """
    Place distinct colored markers at each corner of the photo.
    
    The corners are:
    - TL (index 0) -> UL in YOLO -> Green
    - TR (index 1) -> UR in YOLO -> Red
    - BR (index 2) -> LR in YOLO -> Magenta
    - BL (index 3) -> LL in YOLO -> Blue
    """
    h, w = photo.shape[:2]
    photo_with_markers = photo.copy()
    
    # Corner positions in photo (TL, TR, BR, BL)
    corner_positions = [
        (corner_size, corner_size),                      # TL
        (w - corner_size - 1, corner_size),              # TR
        (w - corner_size - 1, h - corner_size - 1),      # BR
        (corner_size, h - corner_size - 1),              # BL
    ]
    
    # Colors for each corner
    colors = [
        CORNER_COLORS['UL'],  # TL -> UL (Green)
        CORNER_COLORS['UR'],  # TR -> UR (Red)
        CORNER_COLORS['LR'],  # BR -> LR (Magenta)
        CORNER_COLORS['LL'],  # BL -> LL (Blue)
    ]
    
    for pos, color in zip(corner_positions, colors):
        x, y = pos
        # Draw a filled circle at the corner
        cv2.circle(photo_with_markers, (x, y), corner_size, color, -1)
        # Add a crosshair for better detection
        cv2.line(photo_with_markers, (x - corner_size, y), (x + corner_size, y), (255, 255, 255), 2)
        cv2.line(photo_with_markers, (x, y - corner_size), (x, y + corner_size), (255, 255, 255), 2)
    
    return photo_with_markers


def detect_colored_corners(image):
    """
    Detect the colored corner markers in the final image.
    Returns a dict of corner_name -> (x, y) or None if not found.
    """
    detected = {}
    
    # Convert to BGR if needed (in case image is RGBA)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    for corner_name, color_range in COLOR_RANGES.items():
        mask = cv2.inRange(image, color_range['lower'], color_range['upper'])
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (should be the corner marker)
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                detected[corner_name] = (cx, cy)
            else:
                detected[corner_name] = None
        else:
            detected[corner_name] = None
    
    return detected


def calculate_expected_corners_after_rotation(width, height, cx, cy, rotation):
    """
    Calculate where each corner marker should be after rotation.
    
    Returns corners in order: LL, UL, UR, LR (YOLO format)
    
    Mapping from photo corners to YOLO format:
    - Photo BL (index 3) -> LL (index 0)
    - Photo TL (index 0) -> UL (index 1)
    - Photo TR (index 1) -> UR (index 2)
    - Photo BR (index 2) -> LR (index 3)
    """
    rotated_corners = get_rotated_polygon(width, height, cx, cy, rotation)
    
    # Map to YOLO order: LL, UL, UR, LR
    # Polygon: TL=0, TR=1, BR=2, BL=3
    # YOLO:    LL=0, UL=1, UR=2, LR=3
    yolo_mapping = [3, 0, 1, 2]  # BL->LL, TL->UL, TR->UR, BR->LR
    
    yolo_corners = np.zeros((4, 2), dtype=np.float32)
    for i, src_idx in enumerate(yolo_mapping):
        yolo_corners[i] = rotated_corners[src_idx]
    
    return yolo_corners


def apply_global_perspective(canvas, M, output_size):
    """Apply a perspective transformation matrix to the canvas."""
    warped = cv2.warpPerspective(
        canvas, M, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return warped


def create_perspective_transform(canvas_w, canvas_h, strength=0.15):
    """Create a perspective transformation matrix."""
    direction = random.randint(0, 7)
    
    max_offset_x = canvas_w * strength
    max_offset_y = canvas_h * strength
    
    # Initialize offsets
    tl_x = tl_y = tr_x = tr_y = 0.0
    bl_x = bl_y = br_x = br_y = 0.0
    
    if direction == 0:
        tl_x = max_offset_x * 0.8
        tr_x = max_offset_x * 0.5
        bl_x = max_offset_x * 0.8
        br_x = max_offset_x * 0.5
    elif direction == 1:
        tl_x = -max_offset_x * 0.5
        tr_x = -max_offset_x * 0.8
        bl_x = -max_offset_x * 0.5
        br_x = -max_offset_x * 0.8
    elif direction == 2:
        tl_y = max_offset_y * 0.8
        tr_y = max_offset_y * 0.8
    elif direction == 3:
        bl_y = max_offset_y * 0.8
        br_y = max_offset_y * 0.8
    elif direction == 4:
        tl_x = max_offset_x * 0.8
        tl_y = max_offset_y * 0.8
        br_x = -max_offset_x * 0.8
        br_y = -max_offset_y * 0.8
    elif direction == 5:
        tr_x = max_offset_x * 0.8
        tr_y = max_offset_y * 0.8
        bl_x = -max_offset_x * 0.8
        bl_y = -max_offset_y * 0.8
    elif direction == 6:
        bl_x = max_offset_x * 0.8
        bl_y = max_offset_y * 0.8
        tr_x = -max_offset_x * 0.8
        tr_y = -max_offset_y * 0.8
    else:
        br_x = max_offset_x * 0.8
        br_y = max_offset_y * 0.8
        tl_x = -max_offset_x * 0.8
        tl_y = -max_offset_y * 0.8
    
    src_corners = np.array([
        [0, 0],
        [canvas_w - 1, 0],
        [canvas_w - 1, canvas_h - 1],
        [0, canvas_h - 1]
    ], dtype=np.float32)
    
    dst_corners = np.array([
        [tl_x, tl_y],
        [canvas_w - 1 + tr_x, tr_y],
        [canvas_w - 1 + br_x, canvas_h - 1 + br_y],
        [bl_x, canvas_h - 1 + bl_y]
    ], dtype=np.float32)
    
    min_x = min(c[0] for c in dst_corners)
    max_x = max(c[0] for c in dst_corners)
    min_y = min(c[1] for c in dst_corners)
    max_y = max(c[1] for c in dst_corners)
    
    offset_x = -min_x
    offset_y = -min_y
    
    dst_shifted = dst_corners.copy()
    dst_shifted[:, 0] += offset_x
    dst_shifted[:, 1] += offset_y
    
    M = cv2.getPerspectiveTransform(src_corners, dst_shifted)
    
    return M, (int(max_x - min_x) + 1, int(max_y - min_y) + 1)


def transform_corners(corners, M):
    """Transform corners through perspective matrix."""
    ones = np.ones((len(corners), 1))
    corners_h = np.hstack([corners, ones])
    transformed = corners_h @ M.T
    return transformed[:, :2] / transformed[:, 2:3]


def test_single_photo_rotation_composite_warp():
    """
    Test rotation + composite + perspective warp on a single photo.
    
    Steps:
    1. Load a test photo
    2. Place colored markers at corners
    3. Rotate the photo
    4. Composite onto canvas
    5. Apply perspective warp
    6. Detect colored markers in output
    7. Compare with expected positions
    """
    print("=" * 70)
    print("TEST: Single Photo Rotation + Composite + Perspective Warp")
    print("=" * 70)
    
    # Load test image
    test_image_path = Path("./images")
    sources = list(test_image_path.glob('*.jpg')) + list(test_image_path.glob('*.png'))
    
    if not sources:
        print("ERROR: No test images found in ./images")
        return
    
    photo = cv2.imread(str(sources[0]))
    if photo is None:
        print(f"ERROR: Could not load {sources[0]}")
        return
    
    # Step 1: Place colored markers BEFORE rotation
    print("\n[1] Original photo corners:")
    h, w = photo.shape[:2]
    photo_with_markers = place_colored_corners_on_photo(photo.copy(), corner_size=15)
    print(f"    Photo size: {w}x{h}")
    
    # Set up test parameters
    photo_w = 300
    photo_h = int(photo_w * 0.75)
    photo_resized = cv2.resize(photo, (photo_w, photo_h))
    
    cx = 400
    cy = 400
    rotation = random.uniform(-25, 25)
    
    print(f"\n[2] Test parameters:")
    print(f"    Photo size: {photo_w}x{photo_h}")
    print(f"    Center: ({cx}, {cy})")
    print(f"    Rotation: {rotation:.1f}°")
    
    # Create canvas and composite
    canvas_size = 1000  # Large canvas to prevent black edges
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    
    # Place markers BEFORE rotation
    photo_with_markers = place_colored_corners_on_photo(photo_resized.copy(), corner_size=15)
    
    # Rotate the photo (with markers)
    rotated_photo = rotate_photo(photo_with_markers, rotation)
    print(f"    Rotated size: {rotated_photo.shape[1]}x{rotated_photo.shape[0]}")
    
    # Composite onto canvas
    canvas = composite_photo_at_center(canvas, rotated_photo, cx, cy)
    print(f"    Composited onto {canvas_size}x{canvas_size} canvas")
    
    # Calculate where corners SHOULD be after rotation
    expected_yolo_corners = calculate_expected_corners_after_rotation(photo_w, photo_h, cx, cy, rotation)
    print("\n[3] Expected corner positions after rotation (YOLO format: LL, UL, UR, LR):")
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        print(f"    {name}: ({expected_yolo_corners[i][0]:.1f}, {expected_yolo_corners[i][1]:.1f})")
    
    # Apply perspective warp
    perspective_strength = random.uniform(0.10, 0.20)
    M, output_size = create_perspective_transform(canvas_size, canvas_size, perspective_strength)
    print(f"\n[4] Perspective warp (strength={perspective_strength:.2f}):")
    print(f"    Output size: {output_size[0]}x{output_size[1]}")
    
    warped = apply_global_perspective(canvas, M, output_size)
    
    # Transform expected corners through perspective
    warped_expected = transform_corners(expected_yolo_corners, M)
    print("\n[5] Expected corners after perspective transform:")
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        print(f"    {name}: ({warped_expected[i][0]:.1f}, {warped_expected[i][1]:.1f})")
    
    # Detect actual colors in warped image
    print("\n[6] Detecting colored markers in warped image...")
    detected = detect_colored_corners(warped)
    
    print("\n[7] Detection results:")
    for name in ['LL', 'UL', 'UR', 'LR']:
        if detected[name]:
            print(f"    {name}: ({detected[name][0]}, {detected[name][1]})")
        else:
            print(f"    {name}: NOT DETECTED")
    
    # Compare expected vs detected
    print("\n[8] COMPARISON (Expected vs Detected):")
    errors = []
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        if detected[name]:
            exp = warped_expected[i]
            det = detected[name]
            error = np.sqrt((exp[0] - det[0])**2 + (exp[1] - det[1])**2)
            errors.append(error)
            print(f"    {name}: Expected=({exp[0]:.1f},{exp[1]:.1f}) Detected=({det[0]},{det[1]}) Error={error:.1f}px")
        else:
            print(f"    {name}: Expected=({warped_expected[i][0]:.1f},{warped_expected[i][1]:.1f}) Detected=NOT FOUND")
    
    if errors:
        avg_error = np.mean(errors)
        max_error = max(errors)
        print(f"\n📊 RESULTS:")
        print(f"    Average error: {avg_error:.1f}px")
        print(f"    Max error: {max_error:.1f}px")
        if avg_error < 10:
            print(f"    ✅ PASS - Corners aligned within tolerance")
        else:
            print(f"    ❌ FAIL - Corner misalignment detected")
    
    # Save test images
    cv2.imwrite('/tmp/test_canvas.png', canvas[:, :, :3])
    cv2.imwrite('/tmp/test_warped.png', warped[:, :, :3])
    print("\n    Saved test_canvas.png and test_warped.png for inspection")
    
    return errors


def test_multiple_configurations():
    """Test multiple configurations to find patterns in errors."""
    print("\n" + "=" * 70)
    print("TEST: Multiple Configurations")
    print("=" * 70)
    
    all_errors = []
    
    # Test various rotations
    rotations = [-25, -15, 0, 15, 25]
    sizes = [(200, 150), (300, 225), (400, 300)]
    positions = [(400, 400), (500, 300), (300, 500)]
    
    for rotation in rotations:
        for size in sizes:
            for pos in positions:
                errors = test_single_configuration(
                    photo_w=size[0],
                    photo_h=size[1],
                    cx=pos[0],
                    cy=pos[1],
                    rotation=rotation,
                    perspective_strength=0.15
                )
                if errors:
                    all_errors.extend(errors)
    
    if all_errors:
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(f"    Total errors: {len(all_errors)}")
        print(f"    Average error: {np.mean(all_errors):.1f}px")
        print(f"    Max error: {max(all_errors):.1f}px")
        print(f"    Min error: {min(all_errors):.1f}px")


def test_single_configuration(photo_w, photo_h, cx, cy, rotation, perspective_strength):
    """Test a single configuration and return errors."""
    test_image_path = Path("./images")
    sources = list(test_image_path.glob('*.jpg')) + list(test_image_path.glob('*.png'))
    
    if not sources:
        return []
    
    photo = cv2.imread(str(sources[0]))
    if photo is None:
        return []
    
    # Resize photo
    photo_resized = cv2.resize(photo, (photo_w, photo_h))
    
    # Place markers
    photo_with_markers = place_colored_corners_on_photo(photo_resized.copy(), corner_size=15)
    
    # Rotate
    rotated_photo = rotate_photo(photo_with_markers, rotation)
    
    # Create canvas and composite
    canvas_size = 1200
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    canvas = composite_photo_at_center(canvas, rotated_photo, cx, cy)
    
    # Calculate expected corners
    expected_yolo_corners = calculate_expected_corners_after_rotation(photo_w, photo_h, cx, cy, rotation)
    
    # Apply perspective
    M, output_size = create_perspective_transform(canvas_size, canvas_size, perspective_strength)
    warped = apply_global_perspective(canvas, M, output_size)
    
    # Transform expected corners
    warped_expected = transform_corners(expected_yolo_corners, M)
    
    # Detect actual
    detected = detect_colored_corners(warped)
    
    # Calculate errors
    errors = []
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        if detected[name]:
            exp = warped_expected[i]
            det = detected[name]
            error = np.sqrt((exp[0] - det[0])**2 + (exp[1] - det[1])**2)
            errors.append(error)
    
    return errors


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CORNER VERIFICATION - Color Detection Method")
    print("="*70)
    
    errors = test_single_photo_rotation_composite_warp()
    
    print("\n" + "="*70)
    print("Running additional tests...")
    print("="*70)
    
    test_multiple_configurations()