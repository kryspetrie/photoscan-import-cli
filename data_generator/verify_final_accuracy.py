#!/usr/bin/env python3
"""CORNER ACCURACY - Track actual marker positions through the entire pipeline."""
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

from generate_dataset import (
    CONFIG, rotate_photo, composite_photo_at_center, apply_global_perspective
)


def get_photo_corners_transformed(pw, ph, center_x, center_y, rotation, scale_to_output=None):
    """
    Calculate where the 4 PHOTO CORNERS end up after:
    1. Rotation around photo center
    2. Placement at (center_x, center_y) in canvas space
    3. (Optional) Scale to output space
    """
    if abs(rotation) < 1:
        # Simple case: no rotation
        corners = np.array([
            [center_x - pw/2, center_y - ph/2],  # TL
            [center_x + pw/2, center_y - ph/2],  # TR
            [center_x + pw/2, center_y + ph/2],  # BR
            [center_x - pw/2, center_y + ph/2]    # BL
        ], dtype=np.float32)
    else:
        # Build rotation matrix around photo center
        photo_center = (pw / 2, ph / 2)
        M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
        
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(ph * sin_a + pw * cos_a)
        new_h = int(ph * cos_a + pw * sin_a)
        
        # Add the translation for canvas expansion
        M[0, 2] += (new_w - pw) / 2
        M[1, 2] += (new_h - ph) / 2
        
        # Corners in PHOTO SPACE
        corners_photo = np.array([
            [0, 0], [pw, 0], [pw, ph], [0, ph]
        ], dtype=np.float32)
        
        # Transform corners through M
        corners_rot = np.zeros_like(corners_photo)
        for i in range(4):
            pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
            result = M @ pt
            corners_rot[i] = [result[0], result[1]]
        
        # Where does the photo center end up in rotated image space?
        rotated_center = M @ np.array([photo_center[0], photo_center[1], 1])
        
        # Offset to place photo center at (center_x, center_y)
        corners = np.zeros_like(corners_rot)
        corners[:, 0] = center_x - rotated_center[0] + corners_rot[:, 0]
        corners[:, 1] = center_y - rotated_center[1] + corners_rot[:, 1]
    
    if scale_to_output is not None:
        corners = corners * scale_to_output
    
    return corners


def create_photo_with_corners(w, h):
    """Create photo with distinctive markers at each corner."""
    photo = np.ones((h, w, 3), dtype=np.uint8) * 220
    ms = 30
    
    # TL - Red
    photo[0:ms, 0:ms] = [0, 0, 255]
    # TR - Green
    photo[0:ms, w-ms:w] = [0, 255, 0]
    # BR - Blue
    photo[h-ms:h, w-ms:w] = [255, 0, 0]
    # BL - Yellow
    photo[h-ms:h, 0:ms] = [0, 255, 255]
    
    return photo


def detect_corners(img):
    """Detect corner markers in image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    markers = {}
    
    ranges = [
        (0, [0, 100, 100], [15, 255, 255]),     # TL - Red
        (1, [35, 100, 100], [85, 255, 255]),    # TR - Green
        (2, [100, 100, 100], [130, 255, 255]),  # BR - Blue
        (3, [15, 100, 100], [45, 255, 255])      # BL - Yellow
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


def run_test():
    CS, PD = 640, 300
    pw, ph = 300, 200
    cx, cy = CS//2+PD, CS//2+PD  # Canvas center in padded space
    OUTPUT_SIZE = 640
    
    np.random.seed(42)
    
    print(f"\n{'='*70}")
    print("CORNER ACCURACY TEST - Using tracked corners through perspective")
    print("="*70)
    print(f"  Canvas: {CS}x{CS} + {PD}px padding = {CS+2*PD}x{CS+2*PD}")
    print(f"  Photo: {pw}x{ph}")
    print(f"  Output: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    
    all_errors = []
    
    for angle in range(-60, 91, 15):
        # Create photo with corner markers
        photo = create_photo_with_corners(pw, ph)
        
        # Rotate photo
        rotated = rotate_photo(photo, angle)
        
        # Add alpha channel
        rgba = cv2.cvtColor(rotated, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
        
        # Create canvas and composite
        canvas = np.ones((PD*2+CS, PD*2+CS, 4), dtype=np.uint8) * 180
        canvas[:, :, 3] = 0
        result = composite_photo_at_center(canvas, rgba, cx, cy)
        
        # Calculate where corners SHOULD be in canvas space (before perspective)
        canvas_scale = OUTPUT_SIZE / (CS + 2*PD)
        corners_before = get_photo_corners_transformed(pw, ph, cx, cy, angle)
        
        # Apply perspective - this will compute the actual transformation
        warped, global_corners, transform_matrix, content_bounds, warped_corners = apply_global_perspective(
            result, CS + 2*PD, CS + 2*PD,
            photo_corners=[corners_before],
            crop_margin=CONFIG['CROP_MARGIN']
        )
        
        # Get output size after perspective
        warped_h, warped_w = warped.shape[:2]
        
        # Calculate scale factors for final output
        scale_x = OUTPUT_SIZE / warped_w
        scale_y = OUTPUT_SIZE / warped_h
        
        # Resize warped image to output size
        final = cv2.resize(warped, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Scale the tracked corners to output space
        # warped_corners already has the corners transformed through perspective
        # Now scale them from warped space to output space
        if warped_corners and len(warped_corners) > 0:
            expected_corners = warped_corners[0] * np.array([scale_x, scale_y])
        else:
            expected_corners = corners_before * np.array([canvas_scale * scale_x, canvas_scale * scale_y])
        
        # Detect actual corner markers in final image
        detected = detect_corners(final)
        
        # Calculate errors
        errors = []
        for i in range(4):
            if i in detected:
                error = np.sqrt(
                    (detected[i][0] - expected_corners[i][0])**2 + 
                    (detected[i][1] - expected_corners[i][1])**2
                )
                errors.append(error)
        
        if errors:
            all_errors.extend(errors)
            mx = max(errors)
            avg = sum(errors) / len(errors)
            status = "✅" if mx < 5 else "⚠️" if mx < 8 else "❌"
            print(f"  {angle:>4}°: detected={len(errors)}/4, avg={avg:.2f}px, max={mx:.2f}px {status}")
    
    print()
    print(f"{'='*70}")
    print("SUMMARY")
    print("="*70)
    if all_errors:
        overall_avg = sum(all_errors) / len(all_errors)
        overall_max = max(all_errors)
        print(f"  Overall average error: {overall_avg:.2f}px")
        print(f"  Overall max error: {overall_max:.2f}px")
        
        ok_5 = sum(1 for e in all_errors if e < 5)
        ok_8 = sum(1 for e in all_errors if e < 8)
        print(f"  Within 5px: {ok_5}/{len(all_errors)} ({100*ok_5/len(all_errors):.0f}%)")
        print(f"  Within 8px: {ok_8}/{len(all_errors)} ({100*ok_8/len(all_errors):.0f}%)")
        
        if ok_5 == len(all_errors):
            print("\n  ✅ ALL CORNERS WITHIN 5px ACCURACY!")
            return True
        elif ok_8 == len(all_errors):
            print("\n  ⚠️  All corners within 8px (acceptable for training)")
            return True
    
    return False


if __name__ == '__main__':
    success = run_test()
    sys.exit(0 if success else 1)