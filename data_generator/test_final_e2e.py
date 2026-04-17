#!/usr/bin/env python3
"""CORNER ACCURACY - Full pipeline with correct marker position calculation."""
import numpy as np
import cv2
import sys
sys.path.insert(0, '.')

from generate_dataset import (
    CONFIG, rotate_photo, composite_photo_at_center, apply_global_perspective
)

def get_rotation_matrix(w, h, angle):
    """Get rotation matrix for consistent position calculation."""
    if abs(angle) < 1:
        return None, w, h
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return M, new_w, new_h

def get_marker_positions(photo_w, photo_h, center_x, center_y, rotation, marker_offset=20):
    """Calculate where marker centers end up after rotation and placement."""
    half_w = photo_w / 2
    half_h = photo_h / 2
    
    marker_centers = np.array([
        [half_w - marker_offset, half_h - marker_offset],  # TL
        [half_w + marker_offset, half_h - marker_offset],  # TR
        [half_w + marker_offset, half_h + marker_offset],  # BR
        [half_w - marker_offset, half_h + marker_offset]    # BL
    ], dtype=np.float32)
    
    M, new_w, new_h = get_rotation_matrix(photo_w, photo_h, rotation)
    
    positions = []
    for i in range(4):
        marker_pt = np.array([marker_centers[i, 0], marker_centers[i, 1], 1])
        if M is None:
            rotated = marker_pt[:2]
        else:
            rotated = M @ marker_pt
        canvas_x = rotated[0] + (center_x - new_w / 2)
        canvas_y = rotated[1] + (center_y - new_h / 2)
        positions.append([canvas_x, canvas_y])
    
    return np.array(positions, dtype=np.float32)

def detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = {}
    for idx, lo, hi in [
        (0, [0, 50, 50], [15, 255, 255]),
        (1, [35, 50, 50], [85, 255, 255]),
        (2, [100, 50, 50], [130, 255, 255]),
        (3, [15, 50, 50], [45, 255, 255])
    ]:
        mk = cv2.inRange(hsv, np.array(lo), np.array(hi))
        cn, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cn:
            c = max(cn, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                mm = cv2.moments(c)
                m[idx] = (int(mm['m10']/mm['m00']), int(mm['m01']/mm['m00']))
    return m

def create_photo(w, h, marker_size=40):
    p = np.ones((h, w, 3), dtype=np.uint8) * 200
    ms = marker_size
    p[0:ms, 0:ms] = [0, 0, 255]
    p[0:ms, w-ms:w] = [0, 255, 0]
    p[h-ms:h, w-ms:w] = [255, 0, 0]
    p[h-ms:h, 0:ms] = [0, 255, 255]
    return p

def run_test(with_perspective=True):
    CS, PD = 640, 300
    pw, ph = 300, 200
    cx, cy = CS//2+PD, CS//2+PD
    MARKER_SIZE = 40
    MARKER_OFFSET = MARKER_SIZE // 2
    
    np.random.seed(42)
    
    print(f"\n{'='*70}")
    print(f"CORNER ACCURACY - {'WITH' if with_perspective else 'WITHOUT'} PERSPECTIVE")
    print("="*70)
    
    all_errors = []
    
    for angle in range(-60, 91, 15):
        photo = create_photo(pw, ph, MARKER_SIZE)
        rot = rotate_photo(photo, angle)
        rgba = cv2.cvtColor(rot, cv2.COLOR_BGR2BGRA)
        rgba[:,:,3] = 255
        
        canvas = np.ones((PD*2+CS, PD*2+CS, 4), dtype=np.uint8)*180
        canvas[:,:,3] = 0
        res = composite_photo_at_center(canvas, rgba, cx, cy)
        
        if with_perspective:
            # Use marker positions as the corners to track through perspective
            marker_polys = [get_marker_positions(pw, ph, cx, cy, angle, MARKER_OFFSET)]
            wrp, gc, tm, cb, wc = apply_global_perspective(
                res, PD*2+CS, PD*2+CS, photo_corners=marker_polys, crop_margin=60
            )
            hw, hh = wrp.shape[1], wrp.shape[0]
            scale = CS / hw
            final = cv2.resize(wrp, (CS, CS))
            expected = wc[0] * scale
        else:
            # Just resize - marker_polys are in 1240 space, need to scale to 640 space
            res_bgr = cv2.cvtColor(res, cv2.COLOR_BGRA2BGR)
            final = cv2.resize(res_bgr, (CS, CS))
            marker_polys = [get_marker_positions(pw, ph, cx, cy, angle, MARKER_OFFSET)]
            scale = CS / (PD*2+CS)  # 640 / 1240
            expected = marker_polys[0] * scale
        
        det = detect(final)
        
        errors = []
        for i in range(4):
            if i in det:
                e = np.sqrt((det[i][0]-expected[i][0])**2 + (det[i][1]-expected[i][1])**2)
                errors.append(e)
        
        if errors:
            all_errors.extend(errors)
            mx = max(errors)
            status = "✅" if mx < 5 else "⚠️" if mx < 10 else "❌"
            print(f"  {angle:>4}°: detected={len(errors)}/4, max_error={mx:.2f}px {status}")
    
    print()
    if all_errors:
        print(f"  Average error: {sum(all_errors)/len(all_errors):.2f}px")
        print(f"  Max error: {max(all_errors):.2f}px")
        ok_5 = sum(1 for e in all_errors if e < 5)
        print(f"  Within 5px: {ok_5}/{len(all_errors)} ({100*ok_5/len(all_errors):.0f}%)")
        if ok_5 == len(all_errors):
            print("\n  ✅ ALL CORNERS WITHIN 5px!")
        return ok_5 == len(all_errors)
    return False

if __name__ == '__main__':
    run_test(with_perspective=False)
    run_test(with_perspective=True)