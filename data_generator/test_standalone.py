#!/usr/bin/env python3
"""Standalone corner accuracy verification - NO perspective, direct test."""
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
    M[0, 2] += (new_w - w) / 2; M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(photo, M, (new_w, new_h), borderValue=(128, 128, 128))

def composite(canvas, photo, cx, cy):
    ph, pw = photo.shape[:2]
    ch, cw = canvas.shape[:2]
    tlx, tly = int(cx - pw / 2), int(cy - ph / 2)
    sx1, sy1, sx2, sy2 = 0, 0, pw, ph
    dx1, dy1, dx2, dy2 = tlx, tly, tlx + pw, tly + ph
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

def get_polygon(w, h, cx, cy, rot):
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

def detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = {}
    for idx, lo, hi in [(0,[0,100,100],[15,255,255]), (1,[35,100,100],[85,255,255]), (2,[100,100,100],[130,255,255]), (3,[15,100,100],[45,255,255])]:
        mk = cv2.inRange(hsv, np.array(lo), np.array(hi))
        cn, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cn:
            c = max(cn, key=cv2.contourArea)
            if cv2.contourArea(c) > 10:
                mm = cv2.moments(c)
                m[idx] = (int(mm['m10']/mm['m00']), int(mm['m01']/mm['m00']))
    return m

def create_photo(w, h):
    p = np.ones((h, w, 3), dtype=np.uint8) * 200
    ms = 6
    p[0:ms, 0:ms] = [0, 0, 255]
    p[0:ms, w-ms:w] = [0, 255, 0]
    p[h-ms:h, w-ms:w] = [255, 0, 0]
    p[h-ms:h, 0:ms] = [0, 255, 255]
    return p

def test():
    CS, PD = 640, 300
    pw, ph = 300, 200
    cx, cy = CS//2+PD, CS//2+PD
    print(f"\n{'='*60}\nCORNER ACCURACY: Rotation + Resize (No Perspective)\n{'='*60}")
    all_errors = []
    for angle in range(-60, 91, 15):
        poly = get_polygon(pw, ph, cx, cy, angle)
        photo = create_photo(pw, ph)
        rot = rotate_photo(photo, angle)
        rgba = cv2.cvtColor(rot, cv2.COLOR_BGR2BGRA)
        rgba[:,:,3] = 255
        canvas = np.ones((PD*2+CS, PD*2+CS, 4), dtype=np.uint8)*180
        canvas[:,:,3] = 0
        res = composite(canvas, rgba, cx, cy)
        # Direct resize to 640x640 (no perspective)
        res_bgr = cv2.cvtColor(res, cv2.COLOR_BGRA2BGR)
        final = cv2.resize(res_bgr, (CS, CS))
        # Scale polygon from 1240 space to 640 space
        scale = CS / (CS + 2*PD)
        scaled_poly = poly * scale
        det = detect(final)
        errors = []
        for i in range(4):
            if i in det:
                e = np.sqrt((det[i][0]-scaled_poly[i][0])**2 + (det[i][1]-scaled_poly[i][1])**2)
                errors.append(e)
        if errors:
            all_errors.extend(errors)
            avg = sum(errors)/len(errors)
            mx = max(errors)
            status = "✅" if mx < 5 else "⚠️" if mx < 10 else "❌"
            print(f"  {angle:>3}°: avg={avg:.1f}px, max={mx:.1f}px {status}")
    if all_errors:
        print(f"\n  OVERALL: avg={sum(all_errors)/len(all_errors):.2f}px, max={max(all_errors):.2f}px")
        print(f"  Within 5px: {sum(1 for e in all_errors if e < 5)}/{len(all_errors)} ({100*sum(1 for e in all_errors if e < 5)/len(all_errors):.0f}%)")
        if all(e < 5 for e in all_errors):
            print(f"\n  ✅ ALL CORNERS WITHIN 5px ACCURACY!")

if __name__ == '__main__':
    test()