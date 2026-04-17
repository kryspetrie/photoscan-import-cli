#!/usr/bin/env python3
"""
Final verification with CORRECTED rotation formula.

BUG FIX: OpenCV getRotationMatrix2D(angle) performs CLOCKWISE rotation.
To get counterclockwise rotation, we must use NEGATIVE angle.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import CONFIG

# Configuration
CANVAS_SIZE = CONFIG['CANVAS_SIZE']           # 640
PADDING = CONFIG['CANVAS_PADDING']            # 300
PADDED_CANVAS_SIZE = CANVAS_SIZE + 2 * PADDING  # 1240
CROP_MARGIN = CONFIG['CROP_MARGIN']           # 60


def create_synthetic_photo_with_corners(width, height):
    """Create a synthetic photo with distinct colored corner markers."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    s = min(60, width // 6, height // 6)
    
    # CORRECT color placement (using OpenCV coordinate system where y increases down):
    # img[y:y+h, x:x+w] = color
    # img[0:s, 0:s] = top-left
    # img[0:s, w-s:w] = top-right
    # img[h-s:h, w-s:w] = bottom-right
    # img[h-s:h, 0:s] = bottom-left
    
    # Top-Left (0,0) - Blue
    img[0:s, 0:s] = (255, 0, 0)
    
    # Top-Right (w,0) - Yellow
    img[0:s, width-s:width] = (0, 255, 255)
    
    # Bottom-Right (w,h) - Magenta
    img[height-s:height, width-s:width] = (255, 0, 255)
    
    # Bottom-Left (0,h) - Green
    img[height-s:height, 0:s] = (0, 255, 0)
    
    # Center crosshair
    cv2.line(img, (width//2-30, height//2), (width//2+30, height//2), (0, 0, 0), 2)
    cv2.line(img, (width//2, height//2-30), (width//2, height//2+30), (0, 0, 0), 2)
    
    return img


def rotate_photo(photo, angle):
    """Rotate a photo by specified degrees (clockwise in OpenCV)."""
    h, w = photo.shape[:2]
    
    if abs(angle) < 1:
        return photo.copy(), (w, h), None
    
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
    
    return rotated, (new_w, new_h), M


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
        return canvas, (top_left_x, top_left_y)
    
    canvas[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
        photo[src_y1:src_y1+copy_h, src_x1:src_x1+copy_w]
    
    return canvas, (top_left_x, top_left_y)


def apply_global_perspective(canvas, canvas_w, canvas_h, crop_margin=60, seed=None):
    """Apply perspective warp to canvas."""
    if seed is not None:
        random.seed(seed)
    
    max_disp = int(canvas_w * CONFIG['PERSPECTIVE_STRENGTH_MAX'])
    
    tl = (random.randint(-max_disp, 0), random.randint(-max_disp, 0))
    tr = (random.randint(0, max_disp), random.randint(-max_disp, 0))
    bl = (random.randint(-max_disp, 0), random.randint(0, max_disp))
    br = (random.randint(0, max_disp), random.randint(0, max_disp))
    
    src_pts = np.array([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]], dtype=np.float32)
    dst_pts = np.array([
        [tl[0], tl[1]],
        [canvas_w + tr[0], tr[1]],
        [canvas_w + br[0], canvas_h + br[1]],
        [bl[0], canvas_h + bl[1]]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(canvas, M, (canvas_w, canvas_h), 
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(128, 128, 128, 0))
    
    result = warped[crop_margin:crop_margin+CANVAS_SIZE, 
                   crop_margin:crop_margin+CANVAS_SIZE]
    
    return result, (tl, tr, bl, br), M


def get_rotated_polygon(width, height, center_x, center_y, rotation):
    """
    Calculate the 4 corner coordinates of a rotated rectangle.
    
    FIXED: Uses -rotation to match OpenCV's clockwise convention.
    This ensures the polygon matches where the photo is actually placed.
    """
    # Order: LL, UL, UR, LR (counterclockwise starting from bottom-left)
    if abs(rotation) < 1:
        return np.array([
            [center_x - width/2, center_y + height/2],  # LL - bottom-left
            [center_x - width/2, center_y - height/2],  # UL - top-left
            [center_x + width/2, center_y - height/2],  # UR - top-right
            [center_x + width/2, center_y + height/2]   # LR - bottom-right
        ], dtype=np.float32)
    
    photo_center = (width / 2, height / 2)
    # FIX: Use NEGATIVE angle for counterclockwise rotation
    # OpenCV getRotationMatrix2D(angle) does clockwise, so -rotation = counterclockwise
    M_raw = cv2.getRotationMatrix2D(photo_center, -rotation, 1.0)
    
    # Canvas offset: top-left position where photo is placed
    # Photo center at (center_x, center_y), so top-left is at (center_x - rot_w/2, center_y - rot_h/2)
    # But we need to account for the rotation... use the raw matrix approach
    canvas_offset_x = center_x - width / 2
    canvas_offset_y = center_y - height / 2
    
    # Original photo corners in photo space
    # Note: OpenCV uses (x, y) where y increases DOWN
    # LL: (0, height), UL: (0, 0), UR: (width, 0), LR: (width, height)
    corners_photo = np.array([
        [0, height],      # LL - bottom-left
        [0, 0],            # UL - top-left
        [width, 0],        # UR - top-right
        [width, height]   # LR - bottom-right
    ], dtype=np.float32)
    
    corners_final = np.zeros_like(corners_photo)
    for i in range(4):
        pt = np.array([corners_photo[i, 0], corners_photo[i, 1], 1])
        rotated = M_raw @ pt
        corners_final[i, 0] = canvas_offset_x + rotated[0]
        corners_final[i, 1] = canvas_offset_y + rotated[1]
    
    return corners_final


def detect_colored_corner(img, color_name):
    """Detect corner position using color."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if color_name == 'green':
        lower = np.array([35, 100, 100])
        upper = np.array([85, 255, 255])
    elif color_name == 'blue':
        lower = np.array([100, 100, 100])
        upper = np.array([130, 255, 255])
    elif color_name == 'yellow':
        lower = np.array([15, 100, 100])
        upper = np.array([35, 255, 255])
    elif color_name == 'magenta':
        lower = np.array([140, 100, 100])
        upper = np.array([170, 255, 255])
    else:
        return None
    
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            M = cv2.moments(largest)
            if M['m00'] > 0:
                return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return None


def verify_pipeline(photo_w, photo_h, rotation, center_x, center_y, seed, output_dir):
    """Run verification with corrected formula."""
    
    print(f"\n{'='*60}")
    print(f"PHOTO: {photo_w}x{photo_h}, Rot: {rotation}°, Center: ({center_x}, {center_y})")
    print(f"{'='*60}")
    
    # Create photo
    photo = create_synthetic_photo_with_corners(photo_w, photo_h)
    
    # Rotate
    rotated, (rot_w, rot_h), rot_M = rotate_photo(photo, rotation)
    
    # Composite
    canvas = np.zeros((PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE, 3), dtype=np.uint8)
    canvas[:] = (128, 128, 128)
    canvas, top_left = composite_photo_at_center(canvas, rotated, center_x, center_y)
    
    # Perspective warp
    warped, persp_corners, persp_M = apply_global_perspective(
        canvas, PADDED_CANVAS_SIZE, PADDED_CANVAS_SIZE,
        crop_margin=CROP_MARGIN, seed=seed
    )
    
    # Calculate corners
    corners_placement = get_rotated_polygon(photo_w, photo_h, center_x, center_y, rotation)
    
    # Transform through perspective
    corners_persp = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        pt = np.array([corners_placement[i, 0], corners_placement[i, 1], 1.0])
        result = persp_M @ pt
        corners_persp[i, 0] = result[0] / result[2]
        corners_persp[i, 1] = result[1] / result[2]
    
    # Crop
    corners_cropped = corners_persp.copy()
    corners_cropped[:, 0] -= CROP_MARGIN
    corners_cropped[:, 1] -= CROP_MARGIN
    
    # Detect actual corners
    detected = {
        'LL': detect_colored_corner(warped, 'green'),
        'UL': detect_colored_corner(warped, 'blue'),
        'UR': detect_colored_corner(warped, 'yellow'),
        'LR': detect_colored_corner(warped, 'magenta'),
    }
    
    print(f"\nCalculated corners (LL,UL,UR,LR):")
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        print(f"  {name}: ({corners_cropped[i, 0]:.1f}, {corners_cropped[i, 1]:.1f})")
    
    print(f"\nDetected corners:")
    for name, pos in detected.items():
        if pos:
            print(f"  {name}: {pos}")
        else:
            print(f"  {name}: NOT DETECTED")
    
    # Calculate errors
    print(f"\nError analysis:")
    errors = []
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        if detected[name]:
            det = detected[name]
            calc = corners_cropped[i]
            err_x = det[0] - calc[0]
            err_y = det[1] - calc[1]
            err_dist = np.sqrt(err_x**2 + err_y**2)
            errors.append(err_dist)
            status = "✓" if err_dist < 5 else "✗"
            print(f"  {name}: Det=({det[0]},{det[1]}) Calc=({calc[0]:.1f},{calc[1]:.1f}) Error={err_dist:.2f}px {status}")
    
    # Create overlay
    output = warped.copy()
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    
    # Draw detected (circles)
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        if detected[name]:
            cv2.circle(output, detected[name], 20, colors[i], 3)
            cv2.putText(output, f"D:{name}", (detected[name][0]+25, detected[name][1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    # Draw calculated (crosses)
    for i, name in enumerate(['LL', 'UL', 'UR', 'LR']):
        x, y = int(corners_cropped[i, 0]), int(corners_cropped[i, 1])
        cv2.line(output, (x-15, y), (x+15, y), (255, 255, 255), 3)
        cv2.line(output, (x, y-15), (x, y+15), (255, 255, 255), 3)
        cv2.putText(output, f"C:{name}", (x+20, y-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Info
    cv2.putText(output, f"Rot:{rotation}°", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    output_path = output_dir / f"verify_w{photo_w}_h{photo_h}_rot{rotation}_s{seed}.jpg"
    cv2.imwrite(str(output_path), output)
    print(f"\nSaved: {output_path.name}")
    
    return errors


def main():
    output_dir = Path(__file__).parent / "verification_final"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("FINAL VERIFICATION WITH CORRECTED ROTATION")
    print("=" * 60)
    print("Fix: Using -rotation to match OpenCV's clockwise convention")
    print("=" * 60)
    
    test_cases = [
        (300, 200, 0, 620, 620, 42),
        (300, 200, 15, 620, 620, 42),
        (300, 200, -15, 620, 620, 42),
        (300, 200, 25, 620, 620, 42),
        (400, 300, 0, 620, 620, 42),
        (400, 300, 15, 620, 620, 42),
        (300, 400, 0, 620, 620, 42),
        (300, 400, 20, 620, 620, 42),
    ]
    
    all_errors = []
    for w, h, rot, cx, cy, seed in test_cases:
        errors = verify_pipeline(w, h, rot, cx, cy, seed, output_dir)
        all_errors.extend(errors)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if all_errors:
        avg = np.mean(all_errors)
        max_err = np.max(all_errors)
        within_5 = sum(1 for e in all_errors if e < 5)
        print(f"Total: {len(all_errors)} measurements")
        print(f"Average error: {avg:.2f}px")
        print(f"Max error: {max_err:.2f}px")
        print(f"Within 5px: {within_5}/{len(all_errors)} ({100*within_5/len(all_errors):.0f}%)")
        if max_err < 5:
            print("\n✅ SUCCESS! All corners within 5px accuracy")
        else:
            print("\n❌ Some corners exceed 5px")
    print(f"\nOutput: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
