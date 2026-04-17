#!/usr/bin/env python3
"""CORNER ACCURACY VERIFICATION - With correct marker offset handling."""
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

def get_polygon(w, h, cx, cy, rot):
    """Get polygon corners for photo corners (not marker centers)."""
    if abs(rot) < 1:
        return np.array([[cx-w/2, cy-h/2], [cx+w/2, cy-h/2], [cx+w/2, cy+h/2], [cx-w/2, cy+h/2]], dtype=np.float32)
    pc = (w/2, h/2)
    M = cv2.getRotationMatrix2D(pc, rot, 1.0)
    ca, sa = abs(M[0,0]), abs(M[0,1])
    nw, nh = int(h*sa + w*ca), int(h*ca + w*sa)
    M[0,2] += (nw-w)/2; M[1,2] += (nh-h)/2
    corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
    rc = np.zeros_like(corners)
    for i in range(4):
        pt = np.array([corners[i,0], corners[i,1], 1])
        rc[i] = M @ pt
    cr = M @ np.array([pc[0], pc[1], 1])
    return rc + np.array([cx-cr[0], cy-cr[1]])

def get_marker_center_offset(marker_size, photo_size, corner_idx, rotation):
    """Get the offset from photo corner to marker center (in PHOTO space, not rotated)."""
    ms = marker_size
    w, h = photo_size
    
    # Marker centers are at (ms/2, ms/2) from photo corners
    # For corner 0 (TL): offset is (ms/2, ms/2)
    # For corner 1 (TR): offset is (-ms/2, ms/2)
    # For corner 2 (BR): offset is (-ms/2, -ms/2)
    # For corner 3 (BL): offset is (ms/2, -ms/2)
    
    offsets = {
        0: (ms/2, ms/2),      # TL
        1: (-ms/2, ms/2),     # TR
        2: (-ms/2, -ms/2),    # BR
        3: (ms/2, -ms/2)      # BL
    }
    
    ox, oy = offsets[corner_idx]
    
    # Return the UNROTATED offset - the polygon is already rotated
    return np.array([ox, oy])

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
cx, cy = CS//2+PD, CS//2+PD
MARKER_SIZE = 40

np.random.seed(42)

print(f"\n{'='*70}")
print("FINAL CORNER ACCURACY VERIFICATION")
print("="*70)
print(f"  Canvas: {CS}x{CS} + {PD}px padding = {CS+2*PD}x{CS+2*PD}")
print(f"  Photo: {pw}x{ph}")
print(f"  Markers: {MARKER_SIZE}x{MARKER_SIZE} at corners")
print(f"  Scale factor: {CS/(CS+2*PD):.4f}")
print()

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
    
    # Get polygon (photo corners in 1240 space)
    poly = get_polygon(pw, ph, cx, cy, angle)
    
    # Scale to 640 space
    scale = CS / (CS + 2*PD)
    poly_scaled = poly * scale
    
    det = detect(final)
    
    errors = []
    for i in range(4):
        if i in det:
            # Expected marker center = polygon corner + unrotated marker offset
            # The polygon gives where (0,0) in photo space ends up
            # The marker is at (ms/2, ms/2) in PHOTO space
            # After rotation, the offset from corner is the same (M is linear)
            # So marker_position = polygon_corner + (ms/2, ms/2), scaled
            
            offset = get_marker_center_offset(MARKER_SIZE, (pw, ph), i, angle)
            
            expected_x = poly_scaled[i][0] + offset[0]
            expected_y = poly_scaled[i][1] + offset[1]
            
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
    print(f"  Overall average error: {overall_avg:.2f}px")
    print(f"  Overall max error: {overall_max:.2f}px")
    ok_5 = sum(1 for e in all_errors if e < 5)
    ok_8 = sum(1 for e in all_errors if e < 8)
    print(f"  Within 5px: {ok_5}/{len(all_errors)} ({100*ok_5/len(all_errors):.0f}%)")
    print(f"  Within 8px: {ok_8}/{len(all_errors)} ({100*ok_8/len(all_errors):.0f}%)")
    print()
    if ok_5 == len(all_errors):
        print(f"  ✅ ALL CORNERS WITHIN 5px ACCURACY!")
    elif ok_8 == len(all_errors):
        print(f"  ⚠️  All corners within 8px (acceptable for training)")