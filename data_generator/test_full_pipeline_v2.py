#!/usr/bin/env python3
"""Full pipeline test with proper corner detection."""
import numpy as np
import cv2


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


def detect_colored_corners(img):
    """Detect corners by finding colored regions."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    corners = {}
    
    ranges = [
        ([0, 100, 100], [15, 255, 255]),     # RED - TL
        ([35, 100, 100], [85, 255, 255]),    # GREEN - TR
        ([100, 100, 100], [130, 255, 255]),  # BLUE - BR
        ([15, 100, 100], [45, 255, 255])     # YELLOW - BL
    ]
    
    for idx, (lower, upper) in enumerate(ranges):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 10:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    corners[idx] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    
    return corners


def get_polygon(pw, ph, cx, cy, angle):
    """Calculate polygon corners."""
    if abs(angle) < 1:
        return np.array([
            [cx - pw/2, cy - ph/2],
            [cx + pw/2, cy - ph/2],
            [cx + pw/2, cy + ph/2],
            [cx - pw/2, cy + ph/2]
        ], dtype=np.float32)
    
    photo_center = (pw / 2, ph / 2)
    M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
    canvas_offset = (cx - pw/2, cy - ph/2)
    
    corners_photo = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float32)
    polygon = np.zeros_like(corners_photo)
    
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        polygon[i, 0] = canvas_offset[0] + rotated[0]
        polygon[i, 1] = canvas_offset[1] + rotated[1]
    
    return polygon


def test_full_pipeline():
    """Test polygon calculation through full pipeline."""
    print("="*70)
    print("FULL PIPELINE TEST - All steps with proper detection")
    print("="*70)
    
    CS, PD = 640, 300
    pw, ph = 300, 200
    cx, cy = CS//2 + PD, CS//2 + PD
    OUTPUT_SIZE = 640
    
    results = []
    
    for angle in range(-60, 91, 15):
        # Create photo with corner markers
        photo = np.ones((ph, pw, 4), dtype=np.uint8) * 180
        photo[:, :, 3] = 255
        
        ms = 15
        photo[0:ms, 0:ms] = [0, 0, 255, 255]           # TL - RED
        photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]      # TR - GREEN
        photo[ph-ms:ph, pw-ms:pw] = [255, 0, 0, 255]  # BR - BLUE
        photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]    # BL - YELLOW
        
        # Rotate
        rotated = rotate_photo(photo, angle)
        rot_h, rot_w = rotated.shape[:2]
        
        # Get rotation dimensions
        photo_center = (pw / 2, ph / 2)
        M_raw = cv2.getRotationMatrix2D(photo_center, angle, 1.0)
        cos_a = abs(M_raw[0, 0])
        sin_a = abs(M_raw[0, 1])
        new_w = int(ph * sin_a + pw * cos_a)
        new_h = int(ph * cos_a + pw * sin_a)
        
        # Composite
        canvas = np.ones((CS + 2*PD, CS + 2*PD, 4), dtype=np.uint8) * 180
        canvas[:, :, 3] = 0
        
        canvas_top_left_x = cx - new_w / 2
        canvas_top_left_y = cy - new_h / 2
        
        result = canvas.copy()
        for y in range(rot_h):
            for x in range(rot_w):
                result[int(canvas_top_left_y) + y, int(canvas_top_left_x) + x] = rotated[y, x]
        
        # Perspective warp
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
        
        # Crop
        crop_margin = 60
        crop_x2 = out_w - crop_margin
        crop_y2 = out_h - crop_margin
        
        if crop_x2 > crop_margin + 200 and crop_y2 > crop_margin + 200:
            warped = warped[crop_margin:crop_y2, crop_margin:crop_x2]
            warped_h, warped_w = warped.shape[:2]
            
            # Scale factors
            scale_x = OUTPUT_SIZE / warped_w
            scale_y = OUTPUT_SIZE / warped_h
            
            # Transform polygon through perspective
            polygon = get_polygon(pw, ph, cx, cy, angle)
            ones = np.ones((4, 1))
            polygon_h = np.hstack([polygon, ones])
            warped_polygon = polygon_h @ M.T
            warped_polygon = warped_polygon[:, :2] / warped_polygon[:, 2:3]
            
            # Apply crop offset
            warped_polygon[:, 0] -= crop_margin
            warped_polygon[:, 1] -= crop_margin
            
            # Scale to output
            output_polygon = warped_polygon * np.array([scale_x, scale_y])
            
            # Resize final image
            final = cv2.resize(warped, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            
            # Detect corners
            detected = detect_colored_corners(final)
            
            # Calculate errors
            if len(detected) == 4:
                errors = []
                for i in range(4):
                    if i in detected:
                        dist = np.sqrt((output_polygon[i, 0] - detected[i][0])**2 + 
                                      (output_polygon[i, 1] - detected[i][1])**2)
                        errors.append(dist)
                if len(errors) == 4:
                    avg_err = sum(errors) / 4
                    max_err = max(errors)
                else:
                    avg_err = None
                    max_err = None
            else:
                avg_err = None
                max_err = None
        else:
            avg_err = None
            max_err = None
            detected = {}
        
        results.append((angle, avg_err, max_err, len(detected)))
        
        if avg_err is not None:
            status = "✅" if max_err < 5 else "⚠️" if max_err < 10 else "❌"
            print(f"  {angle:>4}°: avg={avg_err:.2f}px, max={max_err:.2f}px, detected={len(detected)}/4 {status}")
        else:
            print(f"  {angle:>4}°: avg=N/A, detected={len(detected)}/4 ⚠️")
    
    # Summary
    valid = [(a, e, m) for a, e, m, d in results if e is not None]
    if valid:
        all_errors = [e for _, e, _ in valid]
        avg_all = sum(all_errors) / len(all_errors)
        max_all = max(m for _, _, m in valid)
        print(f"\n{'='*60}")
        print(f"SUMMARY: avg={avg_all:.2f}px, max={max_all:.2f}px")
        if max_all < 5:
            print("✅ ALL TESTS PASS - < 5px accuracy through full pipeline!")
        else:
            print("❌ SOME TESTS FAIL - accuracy issues remain")


if __name__ == '__main__':
    test_full_pipeline()
