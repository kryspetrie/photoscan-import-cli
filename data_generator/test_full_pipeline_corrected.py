#!/usr/bin/env python3
"""CORNER ACCURACY - Final corrected version."""
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

def apply_perspective(canvas, canvas_w, canvas_h, photo_corners=None, crop_margin=60):
    src = np.array([[0, 0], [canvas_w - 1, 0], [canvas_w - 1, canvas_h - 1], [0, canvas_h - 1]], dtype=np.float32)
    strength = 0.10
    dir_idx = np.random.randint(0, 8)
    max_ox = canvas_w * strength
    max_oy = canvas_h * strength
    ox = [0.0]*4; oy = [0.0]*4
    if dir_idx == 0:
        ox = [np.random.uniform(max_ox*0.6, max_ox) for _ in range(4)]
    elif dir_idx == 1:
        ox = [-np.random.uniform(max_ox*0.5, max_ox) for _ in range(4)]
    elif dir_idx == 2:
        oy = [np.random.uniform(max_oy*0.6, max_oy) if i < 2 else np.random.uniform(-max_oy*0.3, max_oy*0.3) for i in range(4)]
    elif dir_idx == 3:
        oy = [np.random.uniform(-max_oy*0.3, max_oy*0.3) if i < 2 else np.random.uniform(max_oy*0.6, max_oy) for i in range(4)]
    else:
        ox = [np.random.uniform(-max_ox, max_ox) for _ in range(4)]
        oy = [np.random.uniform(-max_oy, max_oy) for _ in range(4)]
    v_tilt = np.random.uniform(-max_oy * 0.3, max_oy * 0.3)
    dst = np.array([
        [ox[0], oy[0] + v_tilt],
        [canvas_w - 1 + ox[1], oy[1] + v_tilt],
        [canvas_w - 1 + ox[2], canvas_h - 1 + oy[2] - v_tilt],
        [ox[3], canvas_h - 1 + oy[3] - v_tilt]
    ], dtype=np.float32)
    min_x = min(c[0] for c in dst); max_x = max(c[0] for c in dst)
    min_y = min(c[1] for c in dst); max_y = max(c[1] for c in dst)
    out_w = int(max_x - min_x) + 1; out_h = int(max_y - min_y) + 1
    offset_x = -min_x; offset_y = -min_y
    dst_offset = dst.copy()
    dst_offset[:, 0] += offset_x; dst_offset[:, 1] += offset_y
    M = cv2.getPerspectiveTransform(src, dst_offset)
    warped = cv2.warpPerspective(canvas, M, (out_w, out_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_photo = []
    if photo_corners is not None:
        for corners in photo_corners:
            ones = np.ones((len(corners), 1))
            corners_h = np.hstack([corners, ones])
            wc = corners_h @ M.T
            warped_photo.append(wc[:, :2] / wc[:, 2:3])
    crop_x1, crop_y1 = crop_margin, crop_margin
    crop_x2, crop_y2 = out_w - crop_margin, out_h - crop_margin
    if crop_x2 > crop_x1 + 200 and crop_y2 > crop_y1 + 200:
        warped = warped[crop_y1:crop_y2, crop_x1:crop_x2]
        for corners in warped_photo:
            corners[:, 0] -= crop_x1; corners[:, 1] -= crop_y1
        dst_offset[:, 0] -= crop_x1; dst_offset[:, 1] -= crop_y1
    return warped, dst_offset, M, (warped.shape[1], warped.shape[0]), warped_photo

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
    """
    Calculate where marker centers end up after rotation and placement.
    
    This is the CORRECTED version that computes marker positions directly
    using the rotation matrix, matching how rotate_photo works.
    
    Args:
        photo_w, photo_h: Original photo dimensions
        center_x, center_y: Where photo center is placed in canvas space
        rotation: Rotation angle in degrees
        marker_offset: Distance from photo corner to marker center (half of marker size)
    
    Returns:
        Array of 4 corner positions in canvas space, one per marker.
    """
    # Marker center positions relative to photo center
    half_w = photo_w / 2
    half_h = photo_h / 2
    
    # Corners: 0=TL, 1=TR, 2=BR, 3=BL
    marker_centers = np.array([
        [half_w - marker_offset, half_h - marker_offset],  # TL marker (at photo corner 0,0)
        [half_w + marker_offset, half_h - marker_offset],  # TR marker
        [half_w + marker_offset, half_h + marker_offset],  # BR marker
        [half_w - marker_offset, half_h + marker_offset]    # BL marker
    ], dtype=np.float32)
    
    # Get rotation matrix
    M, new_w, new_h = get_rotation_matrix(photo_w, photo_h, rotation)
    
    positions = []
    for i in range(4):
        marker_pt = np.array([marker_centers[i, 0], marker_centers[i, 1], 1])
        
        if M is None:
            # No rotation
            rotated = marker_pt[:2]
        else:
            # Apply rotation
            rotated = M @ marker_pt
        
        # Place in canvas by adding offset from photo center placement
        canvas_x = rotated[0] + (center_x - new_w / 2)
        canvas_y = rotated[1] + (center_y - new_h / 2)
        
        positions.append([canvas_x, canvas_y])
    
    return np.array(positions, dtype=np.float32)


# =============================================================================
# MAIN TEST
# =============================================================================

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

CS, PD = 640, 300
pw, ph = 300, 200
cx, cy = CS//2+PD, CS//2+PD
MARKER_SIZE = 40
MARKER_OFFSET = MARKER_SIZE // 2  # 20

np.random.seed(42)

print(f"\n{'='*70}")
print("FINAL CORNER ACCURACY TEST - CORRECTED FUNCTION")
print("="*70)
print(f"  Photo: {pw}x{ph}")
print(f"  Markers: {MARKER_SIZE}x{MARKER_SIZE}")
print(f"  Canvas: {CS}x{CS} + {PD}px padding")

all_errors = []

for angle in range(-60, 91, 15):
    photo = create_photo(pw, ph, MARKER_SIZE)
    rot = rotate_photo(photo, angle)
    rgba = cv2.cvtColor(rot, cv2.COLOR_BGR2BGRA)
    rgba[:,:,3] = 255
    
    canvas = np.ones((PD*2+CS, PD*2+CS, 4), dtype=np.uint8)*180
    canvas[:,:,3] = 0
    
    ph2, pw2 = rgba.shape[:2]
    tlx, tly = int(cx - pw2 / 2), int(cy - ph2 / 2)
    canvas[tly:tly+ph2, tlx:tlx+pw2] = rgba
    
    # Apply perspective
    marker_polys = [get_marker_positions(pw, ph, cx, cy, angle, MARKER_OFFSET)]
    wrp, gc, tm, cb, wc = apply_perspective(canvas, PD*2+CS, PD*2+CS, photo_corners=marker_polys, crop_margin=60)
    
    # Resize
    hw, hh = wrp.shape[1], wrp.shape[0]
    scale = CS / hw
    final = cv2.resize(wrp, (CS, CS))
    fc = wc[0] * scale
    
    det = detect(final)
    
    errors = []
    for i in range(4):
        if i in det:
            e = np.sqrt((det[i][0]-fc[i][0])**2 + (det[i][1]-fc[i][1])**2)
            errors.append(e)
    
    if errors:
        all_errors.extend(errors)
        mx = max(errors)
        status = "✅" if mx < 5 else "⚠️" if mx < 10 else "❌"
        print(f"  {angle:>4}°: detected={len(errors)}/4, max_error={mx:.2f}px {status}")

print()
print(f"{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
if all_errors:
    print(f"  Average error: {sum(all_errors)/len(all_errors):.2f}px")
    print(f"  Max error: {max(all_errors):.2f}px")
    ok_5 = sum(1 for e in all_errors if e < 5)
    print(f"  Within 5px: {ok_5}/{len(all_errors)} ({100*ok_5/len(all_errors):.0f}%)")
    if ok_5 == len(all_errors):
        print("\n  ✅ ALL CORNERS WITHIN 5px ACCURACY!")