#!/usr/bin/env python3
"""Test to find the systematic offset."""
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

def composite(canvas, photo, cx, cy):
    ph, pw = photo.shape[:2]
    tlx, tly = int(cx - pw / 2), int(cy - ph / 2)
    sx1, sy1, sx2, sy2 = 0, 0, pw, ph
    dx1, dy1, dx2, dy2 = tlx, tly, tlx + pw, tly + ph
    ch, cw = canvas.shape[:2]
    if dx1 < 0: sx1 = -dx1; dx1 = 0
    if dy1 < 0: sy1 = -dy1; dy1 = 0
    if dx2 > cw: sx2 = cw - dx1; dx2 = cw
    if dy2 > ch: sy2 = ch - dy1; dy2 = ch
    cw_out = int(dx2 - dx1); ch_out = int(dy2 - dy1)
    if cw_out <= 0 or ch_out <= 0:
        return canvas
    sx1, sy1 = int(sx1), int(sy1)
    canvas_f = canvas.astype(np.float32) / 255.0
    photo_f = photo[sy1:sy1+ch_out, sx1:sx1+cw_out].astype(np.float32) / 255.0
    alpha = photo_f[:, :, 3:4]
    canvas_f[dy1:dy2, dx1:dx2, :3] = (photo_f[:, :, :3] * alpha + canvas_f[dy1:dy2, dx1:dx2, :3] * (1 - alpha)).astype(np.float32)
    canvas_f[dy1:dy2, dx1:dx2, 3] = np.maximum(canvas_f[dy1:dy2, dx1:dx2, 3], photo_f[:, :, 3])
    return (canvas_f * 255).astype(np.uint8)

def create_photo(w, h, marker_size=40):
    p = np.ones((h, w, 3), dtype=np.uint8) * 200
    ms = marker_size
    p[0:ms, 0:ms] = [0, 0, 255]
    p[0:ms, w-ms:w] = [0, 255, 0]
    p[h-ms:h, w-ms:w] = [255, 0, 0]
    p[h-ms:h, 0:ms] = [0, 255, 255]
    return p

def detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = {}
    for idx, lo, hi in [(0,[0,100,100],[15,255,255]), (1,[35,100,100],[85,255,255]), (2,[100,100,100],[130,255,255]), (3,[15,100,100],[45,255,255])]:
        mk = cv2.inRange(hsv, np.array(lo), np.array(hi))
        cn, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cn:
            c = max(cn, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                mm = cv2.moments(c)
                m[idx] = (int(mm['m10']/mm['m00']), int(mm['m01']/mm['m00']))
    return m

# Test WITHOUT perspective - just resize
CS, PD = 640, 300
pw, ph = 300, 200
cx, cy = CS//2+PD, CS//2+PD

print("TESTING WITHOUT PERSPECTIVE - Simple Resize Only")
print("="*60)

for angle in [0, 45, 90]:
    photo = create_photo(pw, ph, marker_size=40)
    rot = rotate_photo(photo, angle)
    print(f"\n{angle}° rotation:")
    print(f"  Rotated photo size: {rot.shape}")
    
    rgba = cv2.cvtColor(rot, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = 255
    
    canvas = np.ones((PD*2+CS, PD*2+CS, 4), dtype=np.uint8)*180
    canvas[:,:,3] = 0
    res = composite(canvas, rgba, cx, cy)
    
    # Just resize without perspective
    res_bgr = cv2.cvtColor(res, cv2.COLOR_BGRA2BGR)
    final = cv2.resize(res_bgr, (CS, CS))
    
    # Expected corner positions (simple: center ± hw, hh, scaled)
    # For 0°: corners are (470,520), (770,520), (770,720), (470,720) in 1240 space
    # After scale by 640/1240: (242.6, 268.4), etc.
    scale = CS / (CS + 2*PD)  # 640/1240 = 0.516
    hw, hh = pw/2, ph/2
    expected = [
        ((cx - hw) * scale, (cy - hh) * scale),
        ((cx + hw) * scale, (cy - hh) * scale),
        ((cx + hw) * scale, (cy + hh) * scale),
        ((cx - hw) * scale, (cy + hh) * scale),
    ]
    
    det = detect(final)
    for i in range(4):
        if i in det:
            e = np.sqrt((det[i][0]-expected[i][0])**2 + (det[i][1]-expected[i][1])**2)
            print(f"  Corner {i}: detected=({det[i][0]},{det[i][1]}), expected=({expected[i][0]:.1f},{expected[i][1]:.1f}), error={e:.2f}px")
