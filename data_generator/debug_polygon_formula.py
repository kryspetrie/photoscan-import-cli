#!/usr/bin/env python3
"""Debug the polygon calculation formula."""
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


def composite_photo_at_center(canvas, photo, cx, cy):
    """Composite photo onto canvas with center at (cx, cy)."""
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


def OLD_get_rotated_polygon(pw, ph, center_x, center_y, rotation):
    """Original (WRONG) polygon calculation."""
    if abs(rotation) < 1:
        return np.array([
            [center_x - pw/2, center_y - ph/2],
            [center_x + pw/2, center_y - ph/2],
            [center_x + pw/2, center_y + ph/2],
            [center_x - pw/2, center_y + ph/2]
        ], dtype=np.float32)
    
    photo_center = (pw / 2, ph / 2)
    M = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    M[0, 2] += (new_w - pw) / 2
    M[1, 2] += (new_h - ph) / 2
    
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    corners_rot = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        result = M @ pt
        corners_rot[i] = [result[0], result[1]]
    
    rotated_center = M @ np.array([photo_center[0], photo_center[1], 1])
    
    corners = np.zeros_like(corners_rot)
    corners[:, 0] = center_x - rotated_center[0] + corners_rot[:, 0]
    corners[:, 1] = center_y - rotated_center[1] + corners_rot[:, 1]
    
    return corners


def NEW_get_rotated_polygon(pw, ph, center_x, center_y, rotation):
    """NEW (CORRECT) polygon calculation.
    
    The composite operation:
    1. Rotates the photo around its center
    2. Places the rotated photo's center at (center_x, center_y)
    
    The rotated photo's top-left on canvas = (center_x - new_w/2, center_y - new_h/2)
    
    The rotation matrix M with canvas offset gives: M @ point = rotated_point + canvas_offset
    where canvas_offset = (new_w - pw)/2, (new_h - ph)/2
    
    So: corner_canvas = (center_x - new_w/2, center_y - new_h/2) + (M @ corner_photo - canvas_offset)
    
    Simplifying: M_raw @ corner_photo gives rotation around photo_center with NO canvas offset.
    corner_canvas = (center_x - new_w/2, center_y - new_h/2) + M_raw @ corner_photo
    """
    if abs(rotation) < 1:
        return np.array([
            [center_x - pw/2, center_y - ph/2],
            [center_x + pw/2, center_y - ph/2],
            [center_x + pw/2, center_y + ph/2],
            [center_x - pw/2, center_y + ph/2]
        ], dtype=np.float32)
    
    photo_center = (pw / 2, ph / 2)
    
    # Build rotation matrix WITHOUT canvas offset
    M_raw = cv2.getRotationMatrix2D(photo_center, rotation, 1.0)
    
    cos_a = abs(M_raw[0, 0])
    sin_a = abs(M_raw[0, 1])
    new_w = int(ph * sin_a + pw * cos_a)
    new_h = int(ph * cos_a + pw * sin_a)
    
    # Canvas offset (this is added in rotate_photo)
    canvas_offset_x = (new_w - pw) / 2
    canvas_offset_y = (new_h - ph) / 2
    
    # Canvas top-left position of the rotated photo
    canvas_top_left_x = center_x - new_w / 2
    canvas_top_left_y = center_y - new_h / 2
    
    corners_photo = np.array([
        [0, 0], [pw, 0], [pw, ph], [0, ph]
    ], dtype=np.float32)
    
    corners = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        # M_raw gives rotated position WITHOUT canvas offset
        rotated = M_raw @ pt
        # Add canvas top-left to get canvas position
        corners[i, 0] = canvas_top_left_x + rotated[0]
        corners[i, 1] = canvas_top_left_y + rotated[1]
    
    return corners


def test_polygon_formula():
    """Test both formulas against actual composite."""
    pw, ph = 300, 200
    cx, cy = 620, 620
    angle = -60
    
    print("="*70)
    print(f"TESTING POLYGON FORMULAS: angle={angle}°")
    print("="*70)
    
    # Create photo with markers
    photo = np.zeros((ph, pw, 4), dtype=np.uint8)
    photo[:, :, :3] = 180
    photo[:, :, 3] = 255
    
    ms = 20
    photo[0:ms, 0:ms] = [255, 0, 0, 255]           # TL - Red
    photo[0:ms, pw-ms:pw] = [0, 255, 0, 255]       # TR - Green
    photo[ph-ms:ph, pw-ms:pw] = [0, 0, 255, 255]   # BR - Blue
    photo[ph-ms:ph, 0:ms] = [0, 255, 255, 255]     # BL - Yellow
    
    # Rotate
    rotated = rotate_photo(photo, angle)
    rot_h, rot_w = rotated.shape[:2]
    print(f"\nRotated photo size: {rot_w}x{rot_h}")
    
    # Composite
    canvas = np.ones((1240, 1240, 4), dtype=np.uint8) * 180
    canvas[:, :, 3] = 0
    result = composite_photo_at_center(canvas, rotated, cx, cy)
    
    # Detect actual corners
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    ranges = [
        ([0, 100, 100], [15, 255, 255], "Red"),
        ([35, 100, 100], [85, 255, 255], "Green"),
        ([100, 100, 100], [130, 255, 255], "Blue"),
        ([15, 100, 100], [45, 255, 255], "Yellow")
    ]
    
    actual = {}
    for i, (lower, upper, name) in enumerate(ranges):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                actual[i] = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    
    print("\nActual detected corner positions:")
    for i, pos in actual.items():
        print(f"  {i}: {pos}")
    
    # Calculate using both formulas
    old_corners = OLD_get_rotated_polygon(pw, ph, cx, cy, angle)
    new_corners = NEW_get_rotated_polygon(pw, ph, cx, cy, angle)
    
    print("\n" + "-"*50)
    print("OLD formula errors:")
    total_old = 0
    for i in range(4):
        if i in actual:
            dx = actual[i][0] - old_corners[i][0]
            dy = actual[i][1] - old_corners[i][1]
            dist = np.sqrt(dx**2 + dy**2)
            total_old += dist
            print(f"  Corner {i}: actual={actual[i]}, expected=({old_corners[i][0]:.1f}, {old_corners[i][1]:.1f}), error={dist:.1f}px")
    print(f"  TOTAL: {total_old:.1f}px")
    
    print("\n" + "-"*50)
    print("NEW formula errors:")
    total_new = 0
    for i in range(4):
        if i in actual:
            dx = actual[i][0] - new_corners[i][0]
            dy = actual[i][1] - new_corners[i][1]
            dist = np.sqrt(dx**2 + dy**2)
            total_new += dist
            print(f"  Corner {i}: actual={actual[i]}, expected=({new_corners[i][0]:.1f}, {new_corners[i][1]:.1f}), error={dist:.1f}px")
    print(f"  TOTAL: {total_new:.1f}px")
    
    print("\n" + "="*70)
    if total_new < total_old:
        print("✅ NEW FORMULA IS BETTER!")
        return True
    else:
        print("❌ OLD FORMULA IS BETTER (or same)")
        return False


if __name__ == '__main__':
    success = test_polygon_formula()
