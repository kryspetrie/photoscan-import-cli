#!/usr/bin/env python3
"""FINAL VERIFICATION: Test polygon calculation through full pipeline."""
import numpy as np
import cv2
import sys
import os

os.chdir('/Users/krys.petrie/dev/photo-pose-detector')
sys.path.insert(0, 'photo-pose-detector/data_generator')


def rotate_photo(photo, angle):
    """Rotate a photo by specified degrees (from generate_dataset.py)."""
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
    """Composite photo onto canvas (from generate_dataset.py)."""
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


def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """Calculate polygon corners (FIXED version)."""
    if abs(rotation) < 1:
        return np.array([
            [center_x - width/2, center_y - height/2],
            [center_x + width/2, center_y - height/2],
            [center_x + width/2, center_y + height/2],
            [center_x - width/2, center_y + height/2]
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    canvas_offset = (center_x - width / 2, center_y - height / 2)
    
    corners_photo = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ], dtype=np.float32)
    
    corners_final = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        corners_final[i, 0] = canvas_offset[0] + rotated[0]
        corners_final[i, 1] = canvas_offset[1] + rotated[1]
    
    return corners_final


def detect_corners(img):
    """Detect corner markers in image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    corners = {}
    
    ranges = [
        ([0, 100, 100], [15, 255, 255]),
        ([35, 100, 100], [85, 255, 255]),
        ([100, 100, 100], [130, 255, 255]),
        ([15, 100, 100], [45, 255, 255])
    ]
    
    for idx, (lower, upper) in enumerate(ranges):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 5:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    corners[idx] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    
    return corners


def test_polygon_accuracy():
    """Test polygon calculation through full pipeline."""
    print("="*70)
    print("FINAL VERIFICATION: Polygon Corner Accuracy")
    print("="*70)
    print("\nTesting polygon corners through rotation + perspective + resize pipeline...")
    
    CS, PD = 640, 300
    pw, ph = 300, 200
    cx, cy = CS//2 + PD, CS//2 + PD
    OUTPUT_SIZE = 640
    
    all_errors = []
    
    for angle in range(-60, 91, 15):
        # Create photo with corner markers
        photo = np.ones((ph, pw, 4), dtype=np.uint8) * 180
        photo[:, :, 3] = 255
        
        ms = 7  # Optimal marker size
        photo[0:ms, 0:ms] = [0, 0, 255, 255]           # TL - RED
        photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]     # TR - GREEN
        photo[ph-ms:ph, pw-ms:pw] = [255, 0, 0, 255] # BR - BLUE
        photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]   # BL - YELLOW
        
        # Rotate
        rotated = rotate_photo(photo, angle)
        
        # Get rotated dimensions
        photo_center = (pw / 2, ph / 2)
        M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
        cos_a = abs(M_raw[0, 0])
        sin_a = abs(M_raw[0, 1])
        new_w = int(ph * sin_a + pw * cos_a)
        new_h = int(ph * cos_a + pw * sin_a)
        
        # Create padded canvas and composite
        canvas = np.ones((CS + 2*PD, CS + 2*PD, 4), dtype=np.uint8) * 180
        canvas[:, :, 3] = 0
        result = composite_photo_at_center(canvas, rotated, cx, cy)
        
        # Apply perspective warp (simplified)
        canvas_w, canvas_h = CS + 2*PD, CS + 2*PD
        src_corners = np.array([
            [0, 0], [canvas_w - 1, 0], [canvas_w - 1, canvas_h - 1], [0, canvas_h - 1]
        ], dtype=np.float32)
        
        perspective_strength = 0.12
        max_offset_x = canvas_w * perspective_strength
        max_offset_y = canvas_h * perspective_strength
        
        dst_corners = np.array([
            [max_offset_x * 0.8, max_offset_y * 0.8],
            [canvas_w - 1 - max_offset_x * 0.5, max_offset_y * 0.8],
            [canvas_w - 1, canvas_h - 1],
            [0, canvas_h - 1]
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
            result, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Crop and resize
        crop_margin = 60
        crop_x2 = out_w - crop_margin
        crop_y2 = out_h - crop_margin
        
        if crop_x2 > crop_margin + 200 and crop_y2 > crop_margin + 200:
            warped = warped[crop_margin:crop_y2, crop_margin:crop_x2]
            warped_h, warped_w = warped.shape[:2]
            
            scale_x = OUTPUT_SIZE / warped_w
            scale_y = OUTPUT_SIZE / warped_h
            
            # Calculate polygon in canvas space
            polygon = get_rotated_polygon(pw, ph, cx, cy, angle)
            
            # Transform through perspective
            ones = np.ones((4, 1))
            polygon_h = np.hstack([polygon, ones])
            warped_polygon = polygon_h @ M.T
            warped_polygon = warped_polygon[:, :2] / warped_polygon[:, 2:3]
            
            # Apply crop offset
            warped_polygon[:, 0] -= crop_margin
            warped_polygon[:, 1] -= crop_margin
            
            # Scale to output
            output_polygon = warped_polygon * np.array([scale_x, scale_y])
            
            # Resize and detect
            final = cv2.resize(warped, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            detected = detect_corners(final)
            
            if len(detected) == 4:
                errors = []
                for i in range(4):
                    if i in detected:
                        dist = np.sqrt((output_polygon[i, 0] - detected[i][0])**2 + 
                                      (output_polygon[i, 1] - detected[i][1])**2)
                        errors.append(dist)
                        all_errors.append(dist)
                
                if errors:
                    avg_err = sum(errors) / 4
                    max_err = max(errors)
                    status = "✅" if max_err < 5 else "⚠️"
                    print(f"  {angle:>4}°: avg={avg_err:.2f}px, max={max_err:.2f}px {status}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    
    if all_errors:
        avg_all = sum(all_errors) / len(all_errors)
        max_all = max(all_errors)
        within_5 = sum(1 for e in all_errors if e < 5)
        
        print(f"\n  Total corner measurements: {len(all_errors)}")
        print(f"  Average error: {avg_all:.2f}px")
        print(f"  Maximum error: {max_all:.2f}px")
        print(f"  Within 5px: {within_5}/{len(all_errors)} ({100*within_5/len(all_errors):.0f}%)")
        
        if max_all < 5:
            print("\n  ✅ SUCCESS! All corners within 5px accuracy!")
            print("  ✅ Polygon calculation is correct!")
            return True
        else:
            print("\n  ❌ Some corners exceed 5px error")
            return False
    
    print("\n  ❌ No valid measurements")
    return False


if __name__ == '__main__':
    success = test_polygon_accuracy()
    sys.exit(0 if success else 1)
