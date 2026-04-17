#!/usr/bin/env python3
"""
End-to-End Pipeline Test with Colored Markers
=============================================

This integrates colored marker detection INTO the actual generate_dataset pipeline
to verify corner tracking in real generated images.
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path


def place_colored_markers_on_photo(photo, size=15):
    """Place distinct colored markers at each corner."""
    h, w = photo.shape[:2]
    result = photo.copy()
    
    # Colors for corners
    colors = [
        (0, 255, 0),      # TL - Green
        (0, 0, 255),      # TR - Red
        (255, 0, 255),    # BR - Magenta
        (255, 0, 0),      # BL - Blue
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


def detect_colored_markers(photo):
    """Detect colored markers and return positions."""
    img_check = photo[:, :, :3] if len(photo.shape) > 2 else photo
    
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


def apply_perspective_warp(canvas, strength=0.15, direction=2):
    """Apply perspective warp and return warped canvas + transform matrix."""
    h, w = canvas.shape[:2]
    
    max_offset_x = w * strength
    max_offset_y = h * strength
    
    # direction 2 = top side closer
    tl_y = max_offset_y * 0.8
    tr_y = max_offset_y * 0.8
    br_y = -max_offset_y * 0.8
    bl_y = -max_offset_y * 0.8
    
    src_corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)
    
    dst_corners = np.array([
        [0, tl_y],
        [w - 1, tr_y],
        [w - 1 + 0, h - 1 + br_y],
        [0, h - 1 + bl_y]
    ], dtype=np.float32)
    
    min_x = min(c[0] for c in dst_corners)
    min_y = min(c[1] for c in dst_corners)
    
    dst_shifted = dst_corners.copy()
    dst_shifted[:, 0] -= min_x
    dst_shifted[:, 1] -= min_y
    
    M = cv2.getPerspectiveTransform(src_corners, dst_shifted)
    
    max_x = max(c[0] for c in dst_corners)
    max_y = max(c[1] for c in dst_corners)
    out_w = int(max_x - min_x) + 1
    out_h = int(max_y - min_y) + 1
    
    warped = cv2.warpPerspective(canvas, M, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))
    
    return warped, M


def test_generation_pipeline():
    """Test the generation pipeline with colored markers."""
    
    print("=" * 70)
    print("END-TO-END GENERATION PIPELINE TEST")
    print("=" * 70)
    
    # Load test image
    test_image_path = Path("./images")
    sources = list(test_image_path.glob('*.jpg'))
    
    if not sources:
        print("ERROR: No test images found")
        return
    
    photo = cv2.imread(str(sources[0]))
    
    # Test parameters (matching generate_dataset.py)
    PHOTO_W = 250
    PHOTO_H = 200
    CENTER_X = 300  # In 0-640 space
    CENTER_Y = 300
    ROTATION = 20  # degrees
    PADDING = 300
    CANVAS_W = 640 + 2 * PADDING  # 1240
    CANVAS_H = 640 + 2 * PADDING
    CROP_MARGIN = 60
    FINAL_SIZE = 640
    
    print(f"\nParameters:")
    print(f"  Photo: {PHOTO_W}x{PHOTO_H}")
    print(f"  Center: ({CENTER_X}, {CENTER_Y}) in 0-640 space")
    print(f"  Rotation: {ROTATION}°")
    print(f"  Padding: {PADDING}px")
    print(f"  Canvas: {CANVAS_W}x{CANVAS_H}")
    
    # Resize photo
    photo = cv2.resize(photo, (PHOTO_W, PHOTO_H))
    
    # Step 1: Place colored markers BEFORE rotation
    photo_with_markers = place_colored_markers_on_photo(photo.copy())
    
    print(f"\n{'=' * 50}")
    print("Step 1: Markers placed on original photo")
    print("{'=' * 50}")
    
    detected_original = detect_colored_markers(photo_with_markers)
    print("Original marker positions:")
    for i, d in enumerate(detected_original):
        if d:
            print(f"  Corner {i}: {d}")
    
    # Step 2: Rotate photo WITH markers
    rotated, rotated_size = rotate_photo(photo_with_markers, ROTATION)
    
    print(f"\n{'=' * 50}")
    print("Step 2: After rotation")
    print("{'=' * 50}")
    print(f"Rotated size: {rotated_size}")
    
    detected_rotated = detect_colored_markers(rotated)
    print("Rotated marker positions:")
    for i, d in enumerate(detected_rotated):
        if d:
            print(f"  Corner {i}: {d}")
    
    # Step 3: Composite onto padded canvas
    canvas = np.zeros((CANVAS_H, CANVAS_W, 4), dtype=np.uint8)
    canvas[:, :, :3] = 180
    canvas[:, :, 3] = 255
    
    # Canvas center = CENTER + PADDING
    canvas_cx = CENTER_X + PADDING
    canvas_cy = CENTER_Y + PADDING
    
    canvas_with_photo = composite_at_center(canvas, rotated, canvas_cx, canvas_cy)
    
    print(f"\n{'=' * 50}")
    print("Step 3: After compositing onto padded canvas")
    print("{'=' * 50}")
    print(f"Canvas center: ({canvas_cx}, {canvas_cy})")
    
    detected_canvas = detect_colored_markers(canvas_with_photo)
    print("Canvas marker positions:")
    for i, d in enumerate(detected_canvas):
        if d:
            print(f"  Corner {i}: {d}")
    
    # Step 4: Calculate where get_rotated_polygon says corners should be
    print(f"\n{'=' * 50}")
    print("Step 4: Compare detected vs. get_rotated_polygon() calculation")
    print("{'=' * 50}")
    
    # get_rotated_polygon returns corners in PHOTO SPACE (before rotation)
    # But we need corners in CANVAS SPACE (after compositing)
    
    # Original polygon in photo space
    polygon_photo = get_rotated_polygon(PHOTO_W, PHOTO_H, PHOTO_W/2, PHOTO_H/2, 0)
    print("Polygon in photo space (before rotation):")
    for i, pt in enumerate(polygon_photo):
        print(f"  Corner {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # Rotate polygon
    polygon_rotated = get_rotated_polygon(PHOTO_W, PHOTO_H, PHOTO_W/2, PHOTO_H/2, ROTATION)
    print(f"\nPolygon after rotation (in rotated photo space):")
    for i, pt in enumerate(polygon_rotated):
        print(f"  Corner {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # Transform to canvas space (add offset from composite)
    polygon_canvas = polygon_rotated.copy()
    # The rotated photo's top-left is at (canvas_cx - rotated_size[0]/2, canvas_cy - rotated_size[1]/2)
    photo_tl_x = canvas_cx - rotated_size[0] / 2
    photo_tl_y = canvas_cy - rotated_size[1] / 2
    polygon_canvas[:, 0] += photo_tl_x
    polygon_canvas[:, 1] += photo_tl_y
    
    print(f"\nPolygon in canvas space (after compositing):")
    print(f"  Photo top-left: ({photo_tl_x:.1f}, {photo_tl_y:.1f})")
    for i, pt in enumerate(polygon_canvas):
        print(f"  Corner {i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # Compare detected vs. calculated
    print(f"\nComparison (detected vs. calculated in canvas):")
    for i, (det, calc) in enumerate(zip(detected_canvas, polygon_canvas)):
        if det:
            error = math.sqrt((det[0] - calc[0])**2 + (det[1] - calc[1])**2)
            status = "✅" if error < 5 else "❌"
            print(f"  Corner {i}: detected={det}, calculated=({calc[0]:.1f},{calc[1]:.1f}), error={error:.1f}px {status}")
    
    # Step 5: Apply perspective warp
    warped, M_persp = apply_perspective_warp(canvas_with_photo, strength=0.15, direction=2)
    
    print(f"\n{'=' * 50}")
    print("Step 5: After perspective warp")
    print("{'=' * 50}")
    
    detected_warped = detect_colored_markers(warped)
    
    # Transform polygon through perspective
    def transform_point(point, M):
        x, y = point
        denom = M[2, 0] * x + M[2, 1] * y + M[2, 2]
        if abs(denom) < 1e-10:
            return point
        new_x = (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / denom
        new_y = (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / denom
        return (new_x, new_y)
    
    print("Warped marker positions:")
    for i, d in enumerate(detected_warped):
        if d:
            print(f"  Corner {i}: {d}")
    
    # Step 6: Crop and rescale
    crop_x1, crop_y1 = CROP_MARGIN, CROP_MARGIN
    crop_x2 = warped.shape[1] - CROP_MARGIN
    crop_y2 = warped.shape[0] - CROP_MARGIN
    
    cropped = warped[crop_y1:crop_y2, crop_x1:crop_x2]
    
    print(f"\n{'=' * 50}")
    print("Step 6: After cropping")
    print("{'=' * 50}")
    print(f"Crop: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
    
    detected_cropped = detect_colored_markers(cropped)
    
    # Transform polygon through crop
    polygon_cropped = polygon_canvas.copy()
    polygon_cropped[:, 0] -= crop_x1
    polygon_cropped[:, 1] -= crop_y1
    
    print("Cropped marker positions:")
    for i, d in enumerate(detected_cropped):
        if d:
            print(f"  Corner {i}: {d}")
    
    # Rescale to 640x640
    scale = FINAL_SIZE / cropped.shape[1] if cropped.shape[1] > 0 else 1
    rescaled = cv2.resize(cropped, (FINAL_SIZE, FINAL_SIZE), interpolation=cv2.INTER_LINEAR)
    
    print(f"\n{'=' * 50}")
    print("Step 7: After rescaling to 640x640")
    print("{'=' * 50}")
    print(f"Scale factor: {scale:.4f}")
    
    detected_rescaled = detect_colored_markers(rescaled)
    
    # Transform polygon through rescale
    polygon_final = polygon_cropped.copy()
    polygon_final[:, 0] *= scale
    polygon_final[:, 1] *= scale
    
    print("Final marker positions:")
    for i, d in enumerate(detected_rescaled):
        if d:
            print(f"  Corner {i}: {d}")
    
    print(f"\n{'=' * 50}")
    print("FINAL COMPARISON: Detected vs. Calculated")
    print("{'=' * 50}")
    
    all_errors = []
    for i, (det, calc) in enumerate(zip(detected_rescaled, polygon_final)):
        if det:
            error = math.sqrt((det[0] - calc[0])**2 + (det[1] - calc[1])**2)
            all_errors.append(error)
            status = "✅" if error < 10 else "❌"
            print(f"  Corner {i}: detected={det}, calculated=({calc[0]:.1f},{calc[1]:.1f}), error={error:.1f}px {status}")
    
    if all_errors:
        avg_error = np.mean(all_errors)
        max_error = max(all_errors)
        print(f"\nAverage error: {avg_error:.1f}px")
        print(f"Maximum error: {max_error:.1f}px")
        
        if max_error < 15:
            print("✅ PIPELINE PASS - Corners within 15px tolerance")
        else:
            print("❌ PIPELINE FAIL - Corners exceed tolerance")
    
    # Save diagnostic image
    output = rescaled.copy()
    corner_names = ['TL', 'TR', 'BR', 'BL']
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 0, 0)]
    
    for i, (det, calc) in enumerate(zip(detected_rescaled, polygon_final)):
        color = colors[i]
        # Draw detected position
        if det:
            cv2.circle(output, det, 10, color, 2)
            cv2.putText(output, f"Det{i}", (det[0]+12, det[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Draw calculated position
        cv2.circle(output, (int(calc[0]), int(calc[1])), 8, color, -1)
        cv2.putText(output, f"Calc{i}", (int(calc[0])+12, int(calc[1])+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite('/tmp/e2e_pipeline_test.png', output)
    print("\nSaved /tmp/e2e_pipeline_test.png")


if __name__ == '__main__':
    test_generation_pipeline()