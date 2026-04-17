#!/usr/bin/env python3
"""CORNER ACCURACY - Compute marker position from marker's position relative to photo center."""
import numpy as np
import cv2

def rotate_photo(photo, angle):
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
    return cv2.warpAffine(photo, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128, 0))

def get_rotation_matrix(w, h, angle):
    """Get the rotation matrix used by rotate_photo."""
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

CS, PD = 640, 300
pw, ph = 300, 200
cx, cy = CS//2+PD, CS//2+PD  # Canvas center in padded space
MARKER_SIZE = 40
HALF_MARKER = MARKER_SIZE / 2

# Marker center positions relative to PHOTO CENTER (in photo space)
MARKER_CENTERS = {
    0: (HALF_MARKER, HALF_MARKER),    # TL marker center (20, 20)
    1: (pw - HALF_MARKER, HALF_MARKER), # TR marker center (280, 20)
    2: (pw - HALF_MARKER, ph - HALF_MARKER),  # BR marker center (280, 180)
    3: (HALF_MARKER, ph - HALF_MARKER)  # BL marker center (20, 180)
}

np.random.seed(42)

print(f"\n{'='*70}")
print("CORNER ACCURACY - Correct Marker Position Calculation")
print("="*70)
print(f"  Photo size: {pw}x{ph}")
print(f"  Photo center: ({pw/2}, {ph/2})")
print(f"  Markers at photo corners: {MARKER_CENTERS}")

all_errors = []

for angle in range(-60, 91, 15):
    photo = np.ones((ph, pw, 3), dtype=np.uint8) * 200
    ms = MARKER_SIZE
    photo[0:ms, 0:ms] = [0, 0, 255]
    photo[0:ms, pw-ms:pw] = [0, 255, 0]
    photo[ph-ms:ph, pw-ms:pw] = [255, 0, 0]
    photo[ph-ms:ph, 0:ms] = [0, 255, 255]
    
    rot = rotate_photo(photo, angle)
    rgba = cv2.cvtColor(rot, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = 255
    
    canvas = np.ones((PD*2+CS, PD*2+CS, 4), dtype=np.uint8)*180
    canvas[:,:,3] = 0
    
    ph2, pw2 = rgba.shape[:2]
    tlx, tly = int(cx - pw2 / 2), int(cy - ph2 / 2)
    canvas[tly:tly+ph2, tlx:tlx+pw2] = rgba
    
    res_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    final = cv2.resize(res_bgr, (CS, CS))
    
    # Get rotation matrix and new dimensions
    M, new_w, new_h = get_rotation_matrix(pw, ph, angle)
    
    det = detect(final)
    scale = CS / (CS + 2*PD)
    
    errors = []
    for i in range(4):
        if i in det:
            marker_center = MARKER_CENTERS[i]
            
            # Rotate the marker center through the rotation matrix
            # (same transformation as in rotate_photo)
            marker_pt = np.array([marker_center[0], marker_center[1], 1])
            marker_rot = M @ marker_pt if M is not None else marker_pt[:2]
            
            # The rotated photo is placed with its center at (cx, cy)
            # So marker in canvas = rotated_marker + (cx - new_w/2, cy - new_h/2)
            # But actually, the composite places photo with its center at (cx, cy)
            # The rotated photo has its center at (new_w/2, new_h/2) in its own coordinate space
            # And we place it so that this center is at (cx, cy) in canvas space
            
            # In canvas space, the marker position is:
            # marker_canvas = marker_rot + (cx - new_w/2, cy - new_h/2)
            marker_canvas_x = marker_rot[0] + (cx - new_w/2)
            marker_canvas_y = marker_rot[1] + (cy - new_h/2)
            
            # Scale to output space
            expected_x = marker_canvas_x * scale
            expected_y = marker_canvas_y * scale
            
            e = np.sqrt((det[i][0]-expected_x)**2 + (det[i][1]-expected_y)**2)
            errors.append(e)
    
    if errors:
        all_errors.extend(errors)
        mx = max(errors)
        status = "✅" if mx < 5 else "⚠️" if mx < 10 else "❌"
        print(f"  {angle:>4}°: max_error={mx:.2f}px {status}")

print()
print(f"{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
if all_errors:
    overall_avg = sum(all_errors)/len(all_errors)
    overall_max = max(all_errors)
    print(f"  Average error: {overall_avg:.2f}px")
    print(f"  Max error: {overall_max:.2f}px")
    ok_5 = sum(1 for e in all_errors if e < 5)
    ok_8 = sum(1 for e in all_errors if e < 8)
    print(f"  Within 5px: {ok_5}/{len(all_errors)} ({100*ok_5/len(all_errors):.0f}%)")
    print(f"  Within 8px: {ok_8}/{len(all_errors)} ({100*ok_8/len(all_errors):.0f}%)")
    if ok_5 == len(all_errors):
        print("\n  ✅ ALL CORNERS WITHIN 5px!")