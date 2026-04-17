#!/usr/bin/env python3
"""
Focused Corner Tracking Diagnostic
==================================

This script traces the FULL pipeline step-by-step with colored markers
to identify exactly where corner tracking goes wrong.

Pipeline steps:
1. Photo corners (no transforms)
2. After rotation
3. After compositing onto padded canvas
4. After perspective warp
5. After cropping
6. After rescaling to 640x640

Each step we place colored markers and verify the pipeline is correct.
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path


def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """Calculate rotated polygon corners."""
    if abs(rotation) < 1:
        hw, hh = width / 2, height / 2
        return np.array([
            [center_x - hw, center_y - hh],
            [center_x + hw, center_y - hh],
            [center_x + hw, center_y + hh],
            [center_x - hw, center_y + hh]
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


def place_colored_corners(photo, size=15):
    """Place distinct colored circles at photo corners."""
    h, w = photo.shape[:2]
    result = photo.copy()
    
    colors = [
        (0, 255, 0),     # TL - Green
        (0, 0, 255),     # TR - Red
        (255, 0, 255),   # BR - Magenta
        (255, 0, 0),     # BL - Blue
    ]
    
    positions = [
        (size, size),
        (w - size - 1, size),
        (w - size - 1, h - size - 1),
        (size, h - size - 1),
    ]
    
    for pos, color in zip(positions, colors):
        cv2.circle(result, pos, size, color, -1)
        cv2.line(result, (pos[0] - size, pos[1]), (pos[0] + size, pos[1]), (255, 255, 255), 2)
        cv2.line(result, (pos[0], pos[1] - size), (pos[0], pos[1] + size), (255, 255, 255), 2)
    
    return result


def detect_corner_colors(photo):
    """Detect colored corner markers and return their positions."""
    h, w = photo.shape[:2]
    n_channels = photo.shape[2] if len(photo.shape) > 2 else 1
    
    # Only look at first 3 channels (BGR) - ignore alpha
    img_check = photo[:, :, :3]
    
    ranges = {
        0: (np.array([0, 200, 0]), np.array([100, 255, 100])),    # Green (TL)
        1: (np.array([0, 0, 200]), np.array([100, 100, 255])),    # Red (TR) 
        2: (np.array([200, 0, 200]), np.array([255, 100, 255])),  # Magenta (BR) 
        3: (np.array([200, 0, 0]), np.array([255, 100, 100])),     # Blue (BL)
    }
    
    detected = [None, None, None, None]
    
    for idx, (lower, upper) in ranges.items():
        mask = cv2.inRange(img_check, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 25:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    detected[idx] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    
    return detected


def transform_point(point, matrix):
    """Apply 3x3 transformation matrix to a point."""
    x, y = point
    denom = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if abs(denom) < 1e-10:
        return point
    new_x = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denom
    new_y = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denom
    return (new_x, new_y)


def composite_at_center(canvas, photo, cx, cy):
    """Composite photo onto canvas."""
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    
    if photo.shape[2] == 3:
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)
    
    top_x = int(cx - pw / 2)
    top_y = int(cy - ph / 2)
    
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


def create_perspective_matrix(canvas_w, canvas_h, strength=0.15, direction=0):
    """Create perspective transformation matrix."""
    max_offset_x = canvas_w * strength
    max_offset_y = canvas_h * strength
    
    # direction maps to different perspective distortions
    # Each entry: (tl_x, tr_x, bl_x, br_x, tl_y, br_y)
    # For top-side closer (direction 2): tl_y and tr_y are positive
    offsets = {
        0: (max_offset_x * 0.8, max_offset_x * 0.5, max_offset_x * 0.8, max_offset_x * 0.5, 0, 0),  # left closer
        1: (-max_offset_x * 0.5, -max_offset_x * 0.8, -max_offset_x * 0.5, -max_offset_x * 0.8, 0, 0),  # right closer
        2: (0, 0, 0, 0, max_offset_y * 0.8, -max_offset_y * 0.8),  # top closer
        3: (0, 0, 0, 0, -max_offset_y * 0.8, max_offset_y * 0.8),  # bottom closer
    }
    
    if direction not in offsets:
        direction = 0
    
    tl_x, tr_x, bl_x, br_x, tl_y, br_y = offsets[direction]
    tr_y = tl_y  # Both top corners get same vertical offset
    bl_y = -br_y  # Bottom corners get opposite vertical offset
    
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
    min_y = min(c[1] for c in dst_corners)
    
    dst_shifted = dst_corners.copy()
    dst_shifted[:, 0] -= min_x
    dst_shifted[:, 1] -= min_y
    
    M = cv2.getPerspectiveTransform(src_corners, dst_shifted)
    
    max_x = max(c[0] for c in dst_corners)
    max_y = max(c[1] for c in dst_corners)
    
    return M, (int(max_x - min_x) + 1, int(max_y - min_y) + 1)


def corner_distances(detected, expected):
    """Calculate distances between detected and expected corners."""
    distances = []
    for d, e in zip(detected, expected):
        if d is not None and e is not None:
            dist = math.sqrt((d[0] - e[0])**2 + (d[1] - e[1])**2)
            distances.append(dist)
        else:
            distances.append(None)
    return distances


def run_diagnostic():
    """Run the full diagnostic pipeline."""
    
    print("=" * 70)
    print("CORNER TRACKING DIAGNOSTIC")
    print("=" * 70)
    
    # Load test image
    test_image_path = Path("./images")
    sources = list(test_image_path.glob('*.jpg'))
    
    if not sources:
        print("ERROR: No test images found")
        return
    
    photo = cv2.imread(str(sources[0]))
    
    # Test parameters
    PHOTO_W = 300
    PHOTO_H = 225
    CENTER_X = 400  # In final 640x640 space
    CENTER_Y = 400
    ROTATION = 15  # degrees
    PADDING = 300  # Extra padding on each side
    CANVAS_W = 640 + 2 * PADDING  # 1240
    CANVAS_H = 640 + 2 * PADDING  # 1240
    PERSPECTIVE_STRENGTH = 0.15
    CROP_MARGIN = 60
    FINAL_SIZE = 640
    
    print(f"\nParameters:")
    print(f"  Photo: {PHOTO_W}x{PHOTO_H}")
    print(f"  Center: ({CENTER_X}, {CENTER_Y})")
    print(f"  Rotation: {ROTATION}°")
    print(f"  Padding: {PADDING}px")
    print(f"  Canvas: {CANVAS_W}x{CANVAS_H}")
    print(f"  Perspective strength: {PERSPECTIVE_STRENGTH}")
    print(f"  Crop margin: {CROP_MARGIN}px")
    
    # Resize photo
    photo = cv2.resize(photo, (PHOTO_W, PHOTO_H))
    
    # Create canvas
    canvas = np.zeros((CANVAS_H, CANVAS_W, 4), dtype=np.uint8)
    canvas[:, :, :3] = 180  # Gray background
    canvas[:, :, 3] = 255
    
    # Step 1: Original photo with colored corners
    print(f"\n{'=' * 50}")
    print("STEP 1: Original photo corners")
    print(f"{'=' * 50}")
    
    photo_with_markers = place_colored_corners(photo.copy())
    cv2.imwrite('/tmp/step1_original.png', photo_with_markers)
    detected = detect_corner_colors(photo_with_markers)
    
    # Expected positions in photo space (TL, TR, BR, BL)
    marker_offset = 15
    expected_photo = [
        (marker_offset, marker_offset),                      # Corner 0: TL (Green)
        (PHOTO_W - marker_offset - 1, marker_offset),         # Corner 1: TR (Red)
        (PHOTO_W - marker_offset - 1, PHOTO_H - marker_offset - 1),  # Corner 2: BR (Magenta)
        (marker_offset, PHOTO_H - marker_offset - 1),          # Corner 3: BL (Blue)
    ]
    
    # Print what colors we placed at what positions
    print("Colors placed in original photo:")
    print("  Corner 0: TL (Green) at (15, 15)")
    print("  Corner 1: TR (Red) at (284, 15)")
    print("  Corner 2: BR (Magenta) at (284, 209)")
    print("  Corner 3: BL (Blue) at (15, 209)")
    print()
    print("Original corner detection results:")
    for i, d in enumerate(detected):
        e = expected_photo[i]
        if d:
            dist = math.sqrt((d[0] - e[0])**2 + (d[1] - e[1])**2)
            status = "✅" if dist < 5 else "❌"
            print(f"  Corner {i}: detected={d}, expected={e}, error={dist:.1f}px {status}")
        else:
            print(f"  Corner {i}: NOT DETECTED, expected={e}")
    
    # Step 2: After rotation
    print(f"\n{'=' * 50}")
    print("STEP 2: After rotation")
    print(f"{'=' * 50}")
    
    rotated, rotated_size = rotate_photo(photo_with_markers.copy(), ROTATION)
    cv2.imwrite('/tmp/step2_rotated.png', rotated)
    detected_rotated = detect_corner_colors(rotated)
    
    # Calculate expected rotated positions
    if abs(ROTATION) > 0.5:
        M_rot = cv2.getRotationMatrix2D((PHOTO_W/2, PHOTO_H/2), ROTATION, 1.0)
        cos_a = abs(M_rot[0, 0])
        sin_a = abs(M_rot[0, 1])
        new_w = int(PHOTO_H * sin_a + PHOTO_W * cos_a)
        new_h = int(PHOTO_H * cos_a + PHOTO_W * sin_a)
        M_rot[0, 2] += (new_w - PHOTO_W) / 2
        M_rot[1, 2] += (new_h - PHOTO_H) / 2
    else:
        M_rot = np.eye(3)
        M_rot[:2, :] = cv2.getRotationMatrix2D((PHOTO_W/2, PHOTO_H/2), 0, 1.0)
    
    expected_rotated = []
    for i, (x, y) in enumerate(expected_photo):
        pt = np.array([x, y, 1])
        result = M_rot @ pt
        expected_rotated.append((result[0], result[1]))
    
    print("Rotated corner positions:")
    for i, (d, e) in enumerate(zip(detected_rotated, expected_rotated)):
        dist = math.sqrt((d[0] - e[0])**2 + (d[1] - e[1])**2) if d and e else None
        status = "✅" if (dist is None or dist < 10) else "❌"
        print(f"  Corner {i}: detected={d}, expected=({e[0]:.1f},{e[1]:.1f}), error={dist:.1f}px {status}")
    
    # Step 3: After compositing onto padded canvas
    print(f"\n{'=' * 50}")
    print("STEP 3: After compositing onto padded canvas")
    print(f"{'=' * 50}")
    
    # Canvas center offset (PADDING shifts the 0-640 space to PADDING to PADDING+640)
    canvas_cx = CENTER_X + PADDING
    canvas_cy = CENTER_Y + PADDING
    
    canvas_with_photo = composite_at_center(canvas.copy(), rotated, canvas_cx, canvas_cy)
    cv2.imwrite('/tmp/step3_composited.png', canvas_with_photo[:,:,:3])
    detected_canvas = detect_corner_colors(canvas_with_photo)
    
    # Expected positions in canvas space
    expected_canvas = []
    for e in expected_rotated:
        expected_canvas.append((e[0] + canvas_cx - rotated_size[0]/2, 
                               e[1] + canvas_cy - rotated_size[1]/2))
    
    print(f"Canvas center offset: ({canvas_cx}, {canvas_cy})")
    print(f"Rotated photo size in canvas: {rotated_size}")
    print("Composited corner positions:")
    for i, (d, e) in enumerate(zip(detected_canvas, expected_canvas)):
        dist = math.sqrt((d[0] - e[0])**2 + (d[1] - e[1])**2) if d and e else None
        status = "✅" if (dist is None or dist < 10) else "❌"
        print(f"  Corner {i}: detected={d}, expected=({e[0]:.1f},{e[1]:.1f}), error={dist:.1f}px {status}")
    
    # Step 4: After perspective warp
    print(f"\n{'=' * 50}")
    print("STEP 4: After perspective warp")
    print(f"{'=' * 50}")
    
    direction = 2  # Top side closer
    M_persp, out_size = create_perspective_matrix(CANVAS_W, CANVAS_H, PERSPECTIVE_STRENGTH, direction)
    
    warped = cv2.warpPerspective(canvas_with_photo, M_persp, out_size,
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    cv2.imwrite('/tmp/step4_warped.png', warped[:,:,:3])
    detected_warped = detect_corner_colors(warped)
    
    # Transform expected corners through perspective
    expected_warped = []
    for e in expected_canvas:
        expected_warped.append(transform_point(e, M_persp))
    
    print(f"Perspective output size: {out_size}")
    print("Warped corner positions:")
    for i, (d, e) in enumerate(zip(detected_warped, expected_warped)):
        dist = math.sqrt((d[0] - e[0])**2 + (d[1] - e[1])**2) if d and e else None
        status = "✅" if (dist is None or dist < 15) else "❌"
        print(f"  Corner {i}: detected={d}, expected=({e[0]:.1f},{e[1]:.1f}), error={dist:.1f}px {status}")
    
    # Step 5: After cropping
    print(f"\n{'=' * 50}")
    print("STEP 5: After cropping")
    print(f"{'=' * 50}")
    
    crop_x1 = CROP_MARGIN
    crop_y1 = CROP_MARGIN
    crop_x2 = out_size[0] - CROP_MARGIN
    crop_y2 = out_size[1] - CROP_MARGIN
    
    cropped = warped[crop_y1:crop_y2, crop_x1:crop_x2]
    cv2.imwrite('/tmp/step5_cropped.png', cropped[:,:,:3] if cropped.size > 0 else warped)
    if cropped.size > 0:
        detected_cropped = detect_corner_colors(cropped)
    else:
        detected_cropped = [None, None, None, None]
    
    # Expected after crop (subtract crop offset)
    expected_cropped = []
    for e in expected_warped:
        expected_cropped.append((e[0] - crop_x1, e[1] - crop_y1))
    
    print(f"Crop region: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
    print("Cropped corner positions:")
    for i, (d, e) in enumerate(zip(detected_cropped, expected_cropped)):
        dist = math.sqrt((d[0] - e[0])**2 + (d[1] - e[1])**2) if d and e else None
        status = "✅" if (dist is None or dist < 15) else "❌"
        print(f"  Corner {i}: detected={d}, expected=({e[0]:.1f},{e[1]:.1f}), error={dist:.1f}px {status}")
    
    # Step 6: After rescaling to 640x640
    print(f"\n{'=' * 50}")
    print("STEP 6: After rescaling to 640x640")
    print(f"{'=' * 50}")
    
    if cropped.size > 0:
        cropped_h, cropped_w = cropped.shape[:2]
        scale_x = FINAL_SIZE / cropped_w if cropped_w > 0 else 1
        scale_y = FINAL_SIZE / cropped_h if cropped_h > 0 else 1
        
        rescaled = cv2.resize(cropped, (FINAL_SIZE, FINAL_SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite('/tmp/step6_rescaled.png', rescaled[:,:,:3])
        detected_rescaled = detect_corner_colors(rescaled)
        
        # Expected after rescale
        expected_rescaled = []
        for e in expected_cropped:
            expected_rescaled.append((e[0] * scale_x, e[1] * scale_y))
        
        print(f"Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
        print("Rescaled corner positions:")
        for i, (d, e) in enumerate(zip(detected_rescaled, expected_rescaled)):
            dist = math.sqrt((d[0] - e[0])**2 + (d[1] - e[1])**2) if d and e else None
            status = "✅" if (dist is None or dist < 15) else "❌"
            print(f"  Corner {i}: detected={d}, expected=({e[0]:.1f},{e[1]:.1f}), error={dist:.1f}px {status}")
        
        # Final summary
        print(f"\n{'=' * 50}")
        print("FINAL SUMMARY")
        print(f"{'=' * 50}")
        
        all_errors = []
        for i, (d, e) in enumerate(zip(detected_rescaled, expected_rescaled)):
            if d and e:
                error = math.sqrt((d[0] - e[0])**2 + (d[1] - e[1])**2)
                all_errors.append(error)
        
        if all_errors:
            avg_error = np.mean(all_errors)
            max_error = max(all_errors)
            print(f"Average corner error: {avg_error:.1f}px")
            print(f"Maximum corner error: {max_error:.1f}px")
            
            if max_error < 15:
                print("✅ PIPELINE PASS - Corners within 15px tolerance")
            else:
                print("❌ PIPELINE FAIL - Corners exceed tolerance")
                print("\nDebug images saved:")
                print("  /tmp/step1_original.png")
                print("  /tmp/step2_rotated.png")
                print("  /tmp/step3_composited.png")
                print("  /tmp/step4_warped.png")
                print("  /tmp/step5_cropped.png")
                print("  /tmp/step6_rescaled.png")
    else:
        print("❌ Cropping resulted in empty image!")


if __name__ == '__main__':
    run_diagnostic()