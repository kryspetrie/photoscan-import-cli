#!/usr/bin/env python3
"""
PRECISE CORNER TRACKING TEST - No source images needed.
Uses synthetic colored photo with corner markers.
"""

import numpy as np
import cv2
import sys
import os

os.chdir('/Users/krys.petrie/dev/photo-pose-detector')
sys.path.insert(0, 'photo-pose-detector/data_generator')


def get_rotated_polygon_simple(pw, ph, center_x, center_y, rotation):
    """Calculate corners after rotation and placement in one step."""
    if abs(rotation) < 1:
        return np.array([
            [center_x - pw/2, center_y - ph/2],
            [center_x + pw/2, center_y - ph/2],
            [center_x + pw/2, center_y + ph/2],
            [center_x - pw/2, center_y + ph/2]
        ], dtype=np.float32)
    
    photo_center = (pw / 2, ph / 2)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    M[0, 2] += (new_w - pw) / 2
    M[1, 2] += (new_h - ph) / 2
    
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    corners_rot = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        result = M @ pt
        corners_rot[i] = [result[0], result[1]]
    
    rotated_center = M @ np.array([photo_center[0], photo_center[1], 1])
    
    corners = np.zeros_like(corners_rot)
    corners[:, 0] = center_x - rotated_center[0] + corners_rot[:, 0]
    corners[:, 1] = center_y - rotated_center[1] + corners_rot[:, 1]
    
    return corners


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


def apply_global_perspective(canvas, canvas_w, canvas_h, photo_corners=None, crop_margin=60):
    """Apply perspective warp - simplified version."""
    import random
    random.seed(None)
    
    src_corners = np.array([
        [0, 0],
        [canvas_w - 1, 0],
        [canvas_w - 1, canvas_h - 1],
        [0, canvas_h - 1]
    ], dtype=np.float32)
    
    perspective_strength = random.uniform(0.10, 0.15)
    max_offset_x = canvas_w * perspective_strength
    max_offset_y = canvas_h * perspective_strength
    
    direction = random.randint(0, 7)
    
    tl_offset_x = tl_offset_y = tr_offset_x = tr_offset_y = 0
    bl_offset_x = bl_offset_y = br_offset_x = br_offset_y = 0
    
    if direction == 0:
        tl_offset_x = random.uniform(max_offset_x * 0.7, max_offset_x)
        tr_offset_x = random.uniform(max_offset_x * 0.4, max_offset_x * 0.8)
    elif direction == 1:
        tl_offset_x = random.uniform(-max_offset_x, -max_offset_x * 0.7)
        tr_offset_x = random.uniform(-max_offset_x * 0.8, -max_offset_x * 0.4)
    elif direction == 2:
        tl_offset_y = random.uniform(max_offset_y * 0.7, max_offset_y)
        tr_offset_y = random.uniform(max_offset_y * 0.7, max_offset_y)
    elif direction == 3:
        bl_offset_y = random.uniform(max_offset_y * 0.7, max_offset_y)
        br_offset_y = random.uniform(max_offset_y * 0.7, max_offset_y)
    else:
        tl_offset_x = random.uniform(max_offset_x * 0.5, max_offset_x)
        tl_offset_y = random.uniform(max_offset_y * 0.5, max_offset_y)
        tr_offset_x = random.uniform(-max_offset_x, -max_offset_x * 0.5)
        tr_offset_y = random.uniform(max_offset_y * 0.5, max_offset_y)
    
    v_tilt = random.uniform(-max_offset_y * 0.2, max_offset_y * 0.2)
    
    dst_corners = np.array([
        [tl_offset_x, tl_offset_y + v_tilt],
        [canvas_w - 1 + tr_offset_x, tr_offset_y + v_tilt],
        [canvas_w - 1 + br_offset_x, canvas_h - 1 + br_offset_y - v_tilt],
        [bl_offset_x, canvas_h - 1 + bl_offset_y - v_tilt]
    ], dtype=np.float32)
    
    min_x = min(c[0] for c in dst_corners)
    max_x = max(c[0] for c in dst_corners)
    min_y = min(c[1] for c in dst_corners)
    max_y = max(c[1] for c in dst_corners)
    
    out_w = int(max_x - min_x) + 1
    out_h = int(max_y - min_y) + 1
    
    offset_x = -min_x
    offset_y = -min_y
    dst_offset = dst_corners.copy()
    dst_offset[:, 0] += offset_x
    dst_offset[:, 1] += offset_y
    
    M = cv2.getPerspectiveTransform(src_corners, dst_offset)
    
    warped = cv2.warpPerspective(
        canvas, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    warped_photo_corners = []
    if photo_corners is not None:
        for corners in photo_corners:
            ones = np.ones((len(corners), 1))
            corners_h = np.hstack([corners, ones])
            warped_corners = corners_h @ M.T
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:3]
            warped_photo_corners.append(warped_corners)
    
    warped_h, warped_w = warped.shape[:2]
    
    crop_margin_px = crop_margin
    crop_x1 = crop_margin_px
    crop_y1 = crop_margin_px
    crop_x2 = warped_w - crop_margin_px
    crop_y2 = warped_h - crop_margin_px
    
    if crop_x2 > crop_x1 + 200 and crop_y2 > crop_y1 + 200:
        warped = warped[crop_y1:crop_y2, crop_x1:crop_x2]
        
        crop_offset_x = float(crop_margin_px)
        crop_offset_y = float(crop_margin_px)
        
        for corners in warped_photo_corners:
            corners[:, 0] -= crop_offset_x
            corners[:, 1] -= crop_offset_y
        
        dst_offset[:, 0] -= crop_offset_x
        dst_offset[:, 1] -= crop_offset_y
    
    global_corners = dst_offset
    content_bounds = (warped.shape[1], warped.shape[0])
    
    return warped, global_corners, M, content_bounds, warped_photo_corners


def create_photo_with_corners(w, h):
    """Create photo with distinctive corner markers."""
    photo = np.ones((h, w, 3), dtype=np.uint8) * 220
    ms = 30
    
    photo[0:ms, 0:ms] = [0, 0, 255]       # TL - Red
    photo[0:ms, w-ms:w] = [0, 255, 0]      # TR - Green
    photo[h-ms:h, w-ms:w] = [255, 0, 0]    # BR - Blue
    photo[h-ms:h, 0:ms] = [0, 255, 255]    # BL - Yellow
    
    return photo


def detect_corners(img):
    """Detect corner markers in image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    markers = {}
    
    ranges = [
        (0, [0, 100, 100], [15, 255, 255]),      # TL - Red
        (1, [35, 100, 100], [85, 255, 255]),    # TR - Green  
        (2, [100, 100, 100], [130, 255, 255]),  # BR - Blue
        (3, [15, 100, 100], [45, 255, 255])     # BL - Yellow
    ]
    
    for idx, lower, upper in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    markers[idx] = (cx, cy)
    
    return markers


def test_single_angle(angle, pw=300, ph=200, output_size=640):
    """Test single angle and return detailed results."""
    
    CS, PD = 640, 300
    
    # Photo center in padded canvas space
    cx, cy = CS//2 + PD, CS//2 + PD
    
    # Step 1: Calculate corners in photo space after rotation/placement
    corners_placement = get_rotated_polygon_simple(pw, ph, cx, cy, angle)
    
    # Step 2: Create and composite photo
    photo = create_photo_with_corners(pw, ph)
    rotated = rotate_photo(photo, angle)
    rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 255
    
    canvas = np.ones((PD*2+CS, PD*2+CS, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    result = composite_photo_at_center(canvas, rgba, cx, cy)
    
    # Step 3: Apply perspective warp
    crop_margin = 60
    warped, global_corners, transform_matrix, content_bounds, warped_corners = apply_global_perspective(
        result, CS + 2*PD, CS + 2*PD,
        photo_corners=[corners_placement],
        crop_margin=crop_margin
    )
    
    warped_h, warped_w = warped.shape[:2]
    
    # Step 4: Calculate scaling to output size
    scale_x = output_size / warped_w
    scale_y = output_size / warped_h
    
    # Step 5: Scale corners from warped space to output space
    if warped_corners and len(warped_corners) > 0:
        expected_output = warped_corners[0] * np.array([scale_x, scale_y])
    else:
        expected_output = corners_placement * np.array([scale_x, scale_y])
    
    # Step 6: Resize warped image to output size
    final = cv2.resize(warped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    
    # Step 7: Detect actual corners
    detected = detect_corners(final)
    
    # Calculate errors
    errors = []
    for i in range(4):
        if i in detected:
            dx = detected[i][0] - expected_output[i][0]
            dy = detected[i][1] - expected_output[i][1]
            error = np.sqrt(dx**2 + dy**2)
            errors.append((i, error, detected[i], expected_output[i]))
    
    return {
        'angle': angle,
        'warped_size': (warped_w, warped_h),
        'scale': (scale_x, scale_y),
        'expected': expected_output,
        'detected': detected,
        'errors': errors,
        'corners_placement': corners_placement,
        'warped_corners': warped_corners[0] if warped_corners else None
    }


def run_comprehensive_test():
    """Run test across all angles and analyze errors."""
    
    print("="*70)
    print("PRECISE CORNER TRACKING TEST")
    print("="*70)
    print("\nTesting step-by-step corner tracking through pipeline...")
    print("Pipeline: rotation -> placement -> perspective -> crop -> resize -> detect\n")
    
    all_results = []
    
    for angle in range(-60, 91, 15):
        result = test_single_angle(angle)
        all_results.append(result)
        
        if result['errors']:
            max_err = max(e[1] for e in result['errors'])
            avg_err = sum(e[1] for e in result['errors']) / len(result['errors'])
            status = "✅" if max_err < 5 else "⚠️" if max_err < 10 else "❌"
            print(f"  {angle:>4}°: warped={result['warped_size'][0]}x{result['warped_size'][1]}, "
                  f"scale={result['scale'][0]:.4f}, avg={avg_err:.2f}px, max={max_err:.2f}px {status}")
    
    print("\n" + "="*70)
    print("DETAILED ERROR ANALYSIS")
    print("="*70)
    
    # Analyze error patterns
    all_errors = []
    for r in all_results:
        all_errors.extend([e[1] for e in r['errors']])
    
    if all_errors:
        print(f"\n  Overall stats:")
        print(f"    Min error: {min(all_errors):.2f}px")
        print(f"    Max error: {max(all_errors):.2f}px")
        print(f"    Avg error: {sum(all_errors)/len(all_errors):.2f}px")
    
    # Show first failure in detail
    for r in all_results:
        if r['errors'] and max(e[1] for e in r['errors']) > 5:
            print(f"\n  DETAILED breakdown for {r['angle']}°:")
            print(f"    Warped size: {r['warped_size']}")
            print(f"    Scale factor: {r['scale']}")
            print(f"\n    Placement space corners (before perspective):")
            for i, c in enumerate(r['corners_placement']):
                print(f"      Corner {i}: ({c[0]:.1f}, {c[1]:.1f})")
            
            print(f"\n    Warped space corners (after perspective, before crop):")
            if r['warped_corners'] is not None:
                for i, c in enumerate(r['warped_corners']):
                    print(f"      Corner {i}: ({c[0]:.1f}, {c[1]:.1f})")
            
            print(f"\n    Output space corners (expected):")
            for i, c in enumerate(r['expected']):
                print(f"      Corner {i}: ({c[0]:.1f}, {c[1]:.1f})")
            
            print(f"\n    Detected corners:")
            for i, err, det, exp in r['errors']:
                print(f"      Corner {i}: detected={det}, error={err:.2f}px")
            break
    
    print("\n" + "="*70)
    if all_errors and max(all_errors) < 5:
        print("✅ ALL CORNERS WITHIN 5px ACCURACY!")
        return True
    else:
        print("❌ ERRORS EXCEED 5px THRESHOLD")
        return False


if __name__ == '__main__':
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
