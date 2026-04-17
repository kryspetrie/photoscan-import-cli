#!/usr/bin/env python3
"""Final verification: all corners at exact photo edge."""
import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import CONFIG, rotate_photo, composite_photo_at_center, apply_global_perspective

def create_test_photo(w, h):
    p = np.ones((h, w, 3), dtype=np.uint8) * 200
    ms = 6
    p[0:ms, 0:ms] = [0, 0, 255]           # TL - Red
    p[0:ms, w-ms:w] = [0, 255, 0]        # TR - Green
    p[h-ms:h, w-ms:w] = [255, 0, 0]      # BR - Blue
    p[h-ms:h, 0:ms] = [0, 255, 255]      # BL - Yellow
    return p

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

def test():
    CS, PD = 640, 300
    pw, ph = 300, 200
    cx, cy = CS//2+PD, CS//2+PD
    print(f"\n{'='*60}\nFINAL CORNER ACCURACY TEST\n{'='*60}")
    all_errors, all_angles = [], []
    for angle in range(-60, 91, 15):
        poly = get_polygon(pw, ph, cx, cy, angle)
        photo = create_test_photo(pw, ph)
        rot = rotate_photo(photo, angle)
        rgba = cv2.cvtColor(rot, cv2.COLOR_BGR2BGRA)
        rgba[:,:,3] = 255
        canvas = np.ones((PD*2+CS, PD*2+CS, 4), dtype=np.uint8)*180
        canvas[:,:,3] = 0
        res = composite_photo_at_center(canvas, rgba, cx, cy)
        wrp, gc, tm, cb, wc = apply_global_perspective(res, PD*2+CS, PD*2+CS, photo_corners=[poly], crop_margin=CONFIG['CROP_MARGIN'])
        hw, hh = wrp.shape[1], wrp.shape[0]
        sx, sy = CS/hw, CS/hh
        final = cv2.resize(wrp, (CS, CS))
        fc = wc[0] * np.array([sx, sy])
        det = detect(final)
        errors = []
        for i in range(4):
            if i in det:
                e = np.sqrt((det[i][0]-fc[i][0])**2 + (det[i][1]-fc[i][1])**2)
                errors.append(e)
        if errors:
            all_errors.extend(errors)
            all_angles.append((angle, sum(errors)/len(errors), max(errors)))
    if all_errors:
        print(f"\nAverage error: {sum(all_errors)/len(all_errors):.2f}px")
        print(f"Max error: {max(all_errors):.2f}px")
        print(f"Within 5px: {sum(1 for e in all_errors if e < 5)}/{len(all_errors)}")
        print(f"\n{'Angle':>6} | {'Avg':>6} | {'Max':>6} | Status")
        print("-"*45)
        for angle, avg, mx in all_angles:
            s = "✅" if mx < 5 else "❌"
            print(f"{angle:>5}° | {avg:>5.1f}px | {mx:>5.1f}px | {s}")
        if all(mx < 5 for _, _, mx in all_angles):
            print("\n✅ ALL ROTATIONS ACHIEVE <5px ACCURACY!")

if __name__ == '__main__':
    test()